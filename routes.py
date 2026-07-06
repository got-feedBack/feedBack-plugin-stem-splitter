"""Stem Splitter — backend routes, job queue, and engine orchestration.

All work is namespaced under ``/api/plugins/stem_splitter/``. Heavy work (HTTP to
a split/transcribe server, ffmpeg, zip repack, pip installs) runs on a background
worker thread — never inside an ``async def`` handler — so it can't block the
event loop. ``setup()`` only wires; it performs no I/O and imports nothing heavy.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect

import engine_install

INSTRUMENT_STEM_IDS = ["guitar", "bass", "drums", "vocals", "other", "piano"]
_BROADCAST_MIN_INTERVAL = 0.15  # s — throttle progress spam


class JobManager:
    def __init__(self, app: FastAPI, context: dict):
        self.app = app
        self.context = context
        self.config_dir = Path(context["config_dir"])
        self.log = context.get("log") or logging.getLogger("feedBack.plugin.stem_splitter")
        self.meta_db = context.get("meta_db")
        self.get_dlc_dir = context.get("get_dlc_dir")
        self.extract_meta = context.get("extract_meta")
        self.jobs_file = self.config_dir / "stem_splitter_jobs.json"
        self.settings_file = self.config_dir / "stem_splitter.json"

        self.jobs: dict[str, dict] = {}
        self.q: "queue.Queue[str]" = queue.Queue()
        self.paused = threading.Event()
        self.lock = threading.Lock()
        self._clients: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_broadcast = 0.0
        self._cancel: set[str] = set()
        # Last install lifecycle state, echoed into the WS snapshot so a fresh or
        # reconnected settings page recovers the terminal state even if it missed
        # the live install_done event (long silent pip stretches can drop the WS).
        self._install: dict | None = None

        self._load_jobs()
        self._worker = threading.Thread(target=self._worker_loop, name="stem_splitter-worker", daemon=True)
        self._worker.start()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load_jobs(self) -> None:
        try:
            data = json.loads(self.jobs_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, list):
            return
        for j in data:
            if not isinstance(j, dict) or "id" not in j:
                continue
            # Interrupted work becomes queued again (user-initiated intent).
            if j.get("status") in ("running", "queued"):
                j["status"] = "queued"
                j["progress"] = 0.0
                self.jobs[j["id"]] = j
                self.q.put(j["id"])
            elif j.get("status") in ("done", "failed", "canceled"):
                self.jobs[j["id"]] = j

    def _save_jobs(self) -> None:
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with self.lock:
                data = list(self.jobs.values())
            self.jobs_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.warning("stem_splitter: failed to persist jobs: %s", e)

    # ── settings + config ────────────────────────────────────────────────────
    def read_settings(self) -> dict:
        defaults = {
            "split_engine": "auto",      # auto | remote | audio-separator | demucs
            "lyrics_engine": "auto",     # auto | remote | local
            "remote_model": "bs_roformer_sw",
            "whisperx_model": "medium",
            "language": "",
        }
        try:
            data = json.loads(self.settings_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k in defaults:
                    if k in data:
                        defaults[k] = data[k]
        except Exception:
            pass
        return defaults

    def write_settings(self, body: dict) -> None:
        cur = self.read_settings()
        for k in cur:
            if k in body:
                cur[k] = body[k]
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.settings_file.write_text(json.dumps(cur, indent=2), encoding="utf-8")

    def _app_config(self) -> dict:
        """Read the app's own config.json (same dir) for the shared server URL.

        This is the app's config, not another plugin's — reading it is allowed
        and avoids duplicating the server URL setting.
        """
        try:
            data = json.loads((self.config_dir / "config.json").read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _server_url(self) -> str | None:
        cfg = self._app_config()
        url = cfg.get("demucs_server_url")
        if isinstance(cfg.get("whisperx"), dict) and cfg["whisperx"].get("server_url"):
            # whisperx.server_url overrides for lyrics; split still uses demucs_server_url
            pass
        if isinstance(url, str) and url.strip():
            return url.strip().rstrip("/")
        return None

    def _api_key(self) -> str | None:
        cfg = self._app_config()
        for key in ("demucs_api_key", "server_api_key"):
            v = cfg.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        wx = cfg.get("whisperx")
        if isinstance(wx, dict) and isinstance(wx.get("api_key"), str) and wx["api_key"].strip():
            return wx["api_key"].strip()
        return None

    def resolve_split_engine(self) -> tuple[str | None, str]:
        """Return (engine, reason). engine is remote|audio-separator|demucs or
        None if unavailable."""
        s = self.read_settings()
        choice = s.get("split_engine", "auto")
        server_url = self._server_url()
        edir = str(engine_install.engine_dir(self.config_dir))
        import split_stems
        as_ok = split_stems.audio_separator_available(edir)
        demucs_ok = split_stems.demucs_available(edir)

        if choice == "remote":
            return ("remote", "remote (forced)") if server_url else (None, "remote forced but no server configured")
        if choice == "audio-separator":
            return ("audio-separator", "local audio-separator (forced)") if as_ok else (None, "audio-separator not installed")
        if choice == "demucs":
            return ("demucs", "local demucs (forced)") if demucs_ok else (None, "demucs not installed")
        # auto
        if server_url:
            return ("remote", "remote (auto)")
        if as_ok:
            return ("audio-separator", "local audio-separator (auto)")
        if demucs_ok:
            return ("demucs", "local demucs (auto)")
        return (None, "no split engine available — configure a server or install a local engine")

    def resolve_lyrics_engine(self) -> tuple[str | None, str]:
        s = self.read_settings()
        choice = s.get("lyrics_engine", "auto")
        server_url = self._server_url()
        edir = str(engine_install.engine_dir(self.config_dir))
        import sys as _sys
        if edir not in _sys.path:
            _sys.path.insert(0, edir)
        try:
            from lyrics_transcribe import whisperx_available
            wx_ok = whisperx_available()
        except Exception:
            wx_ok = False

        if choice == "remote":
            return ("remote", "remote (forced)") if server_url else (None, "remote forced but no server configured")
        if choice == "local":
            return ("local", "local whisperx (forced)") if wx_ok else (None, "whisperx not installed")
        if server_url:
            return ("remote", "remote (auto)")
        if wx_ok:
            return ("local", "local whisperx (auto)")
        return (None, "no lyrics engine available — configure a server or install whisperx")

    # ── job lifecycle ────────────────────────────────────────────────────────
    def enqueue(self, kind: str, filename: str) -> dict:
        job = {
            "id": uuid.uuid4().hex[:12],
            "kind": kind,
            "filename": filename,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "error": None,
            "created": len(self.jobs),
        }
        with self.lock:
            self.jobs[job["id"]] = job
        self.q.put(job["id"])
        self._save_jobs()
        self.broadcast_snapshot()
        return job

    def _update(self, job_id: str, **fields) -> None:
        with self.lock:
            j = self.jobs.get(job_id)
            if not j:
                return
            j.update(fields)
        self.broadcast_snapshot()

    def _make_progress_cb(self, job_id: str, base: float = 0.0, span: float = 1.0):
        def cb(p: float, message: str):
            frac = max(0.0, min(1.0, base + p * span))
            with self.lock:
                j = self.jobs.get(job_id)
                if j:
                    j["progress"] = frac
                    j["message"] = message
            self.broadcast_snapshot(throttle=True)
        return cb

    def _worker_loop(self) -> None:
        while True:
            job_id = self.q.get()
            while self.paused.is_set():
                time.sleep(0.3)
            with self.lock:
                job = self.jobs.get(job_id)
            if not job or job.get("status") != "queued":
                continue
            if job_id in self._cancel:
                self._cancel.discard(job_id)
                self._update(job_id, status="canceled", message="Canceled")
                self._save_jobs()
                continue
            self._update(job_id, status="running", progress=0.0, message="Starting")
            try:
                self._run_job(job)
                self._update(job_id, status="done", progress=1.0, message="Done")
            except Exception as e:
                self.log.exception("stem_splitter: job %s failed", job_id)
                self._update(job_id, status="failed", error=str(e), message=f"Failed: {e}")
            self._save_jobs()

    def _run_job(self, job: dict) -> None:
        import split_stems
        import transcribe

        filename = job["filename"]
        pak_path = self._resolve_pak(filename)
        cb = self._make_progress_cb(job["id"])
        settings = self.read_settings()
        server_url = self._server_url()
        api_key = self._api_key()
        edir = str(engine_install.engine_dir(self.config_dir))
        mdir = str(engine_install.models_dir(self.config_dir))

        if job["kind"] == "split":
            engine, reason = self.resolve_split_engine()
            if not engine:
                raise RuntimeError(reason)
            self._update(job["id"], message=f"Splitting via {reason}")
            split_stems.split_pak(
                pak_path, engine=engine,
                model=settings.get("remote_model") if engine != "demucs" else None,
                server_url=server_url, api_key=api_key,
                engine_dir=edir, models_dir=mdir, progress_cb=cb,
            )
        elif job["kind"] == "transcribe":
            lyr_engine, lyr_reason = self.resolve_lyrics_engine()
            if not lyr_engine:
                raise RuntimeError(lyr_reason)
            split_engine, _ = self.resolve_split_engine()
            split_kwargs = {
                "engine": split_engine, "server_url": server_url, "api_key": api_key,
                "engine_dir": edir, "models_dir": mdir,
                "model": settings.get("remote_model") if split_engine != "demucs" else None,
            } if split_engine else None
            self._update(job["id"], message=f"Transcribing via {lyr_reason}")
            transcribe.transcribe_pak(
                pak_path, mode=lyr_engine, server_url=server_url, api_key=api_key,
                whisperx_model=settings.get("whisperx_model", "medium"),
                language=settings.get("language") or None,
                engine_dir=edir, split_kwargs=split_kwargs, progress_cb=cb,
            )
        else:
            raise RuntimeError(f"unknown job kind {job['kind']!r}")

        self._reindex(filename, pak_path)

    def _resolve_pak(self, filename: str) -> Path:
        if not self.get_dlc_dir:
            raise RuntimeError("library directory not available")
        dlc = self.get_dlc_dir()
        if not dlc:
            raise RuntimeError("library directory not configured")
        from safepath import safe_join
        target = safe_join(Path(dlc).resolve(), filename)
        if target is None:
            raise RuntimeError(f"unsafe song path: {filename!r}")
        if not Path(target).exists():
            raise FileNotFoundError(f"song not found: {filename}")
        return Path(target)

    def _reindex(self, filename: str, pak_path: Path) -> None:
        """Refresh this one song's row so stem_ids / has_lyrics update, exactly
        as the background scanner would (see server.py update_song_meta)."""
        if not (self.meta_db and self.extract_meta):
            return
        try:
            st = pak_path.stat()
            meta = self.extract_meta(pak_path)
            self.meta_db.put(filename, st.st_mtime, st.st_size, meta)
        except Exception as e:
            self.log.warning("stem_splitter: reindex of %s failed: %s", filename, e)

    # ── broadcast ────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.get("created", 0))
        return {"type": "jobs", "paused": self.paused.is_set(), "jobs": jobs,
                "install": self._install}

    def broadcast_snapshot(self, throttle: bool = False) -> None:
        if throttle:
            now = time.monotonic()
            if now - self._last_broadcast < _BROADCAST_MIN_INTERVAL:
                return
            self._last_broadcast = now
        msg = self.snapshot()
        self._push(msg)

    def _push(self, msg: dict) -> None:
        if not self._loop:
            return
        for q in list(self._clients):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:
                pass

    def push_event(self, msg: dict) -> None:
        self._push(msg)


def setup(app: FastAPI, context: dict) -> None:
    mgr = JobManager(app, context)
    log = mgr.log

    try:
        mgr._loop = asyncio.get_event_loop()
    except Exception:
        mgr._loop = None

    P = "/api/plugins/stem_splitter"

    # ── config / settings ────────────────────────────────────────────────────
    @app.get(f"{P}/config")
    def get_config():
        split_engine, split_reason = mgr.resolve_split_engine()
        lyr_engine, lyr_reason = mgr.resolve_lyrics_engine()
        return {
            "settings": mgr.read_settings(),
            "server_url": mgr._server_url(),
            "split": {"engine": split_engine, "reason": split_reason},
            "lyrics": {"engine": lyr_engine, "reason": lyr_reason},
            "engine_status": engine_install.engine_status(mgr.config_dir),
        }

    @app.post(f"{P}/config")
    async def set_config(req: Request):
        body = await req.json()
        mgr.write_settings(body if isinstance(body, dict) else {})
        return {"ok": True, "settings": mgr.read_settings()}

    # ── engine install (opt-in, heavy) ───────────────────────────────────────
    @app.get(f"{P}/engine_status")
    def get_engine_status():
        return engine_install.engine_status(mgr.config_dir)

    @app.post(f"{P}/install_engine")
    def install_engine(body: dict):
        which = (body or {}).get("which", "all")
        # Run in a thread so the pip install (minutes, GB) never blocks the loop;
        # stream progress over the WS.
        def _run():
            def cb(ev: dict):
                st = {"active": True, "which": which, "line": ev.get("line", ""),
                      "pct": ev.get("pct", 0.0), "phase": ev.get("phase", "")}
                mgr._install = st
                mgr.push_event({"type": "install", **st})
            try:
                status = engine_install.install_engine(mgr.config_dir, which, progress_cb=cb)
                mgr._install = {"active": False, "which": which, "pct": 1.0, "phase": "Done"}
                mgr.push_event({"type": "install_done", "which": which, "status": status})
            except Exception as e:
                mgr._install = {"active": False, "which": which, "pct": 0.0,
                                "phase": "Failed", "error": str(e)}
                mgr.push_event({"type": "install_error", "which": which, "error": str(e)})
        threading.Thread(target=_run, name="stem_splitter-install", daemon=True).start()
        return {"ok": True, "started": which}

    @app.post(f"{P}/uninstall_engine")
    def uninstall_engine():
        return engine_install.uninstall_engine(mgr.config_dir)

    # ── jobs ─────────────────────────────────────────────────────────────────
    @app.get(f"{P}/jobs")
    def get_jobs():
        return mgr.snapshot()

    def _enqueue_many(kind: str, body: dict) -> dict:
        names = body.get("filenames")
        if not names and body.get("filename"):
            names = [body["filename"]]
        names = [n for n in (names or []) if isinstance(n, str) and n]
        if not names:
            return {"error": "no filename(s) provided"}
        created = [mgr.enqueue(kind, n) for n in names]
        return {"ok": True, "enqueued": len(created), "jobs": created}

    @app.post(f"{P}/split")
    def post_split(body: dict):
        return _enqueue_many("split", body or {})

    @app.post(f"{P}/transcribe")
    def post_transcribe(body: dict):
        return _enqueue_many("transcribe", body or {})

    # ── missing detection ────────────────────────────────────────────────────
    def _query(**kwargs):
        if not mgr.meta_db:
            return []
        try:
            songs, _total = mgr.meta_db.query_page(page=0, size=1000, **kwargs)
            return songs or []
        except Exception as e:
            log.warning("stem_splitter: query_page failed: %s", e)
            return []

    @app.get(f"{P}/missing_stems")
    def missing_stems():
        songs = _query(stems_lacks=INSTRUMENT_STEM_IDS)
        return {"songs": [{"filename": s.get("filename"), "title": s.get("title"),
                           "artist": s.get("artist")} for s in songs]}

    @app.get(f"{P}/missing_lyrics")
    def missing_lyrics():
        songs = _query(has_lyrics=0)
        return {"songs": [{"filename": s.get("filename"), "title": s.get("title"),
                           "artist": s.get("artist")} for s in songs]}

    # ── queue controls ───────────────────────────────────────────────────────
    @app.post(f"{P}/pause")
    def pause():
        mgr.paused.set()
        mgr.broadcast_snapshot()
        return {"ok": True, "paused": True}

    @app.post(f"{P}/resume")
    def resume():
        mgr.paused.clear()
        mgr.broadcast_snapshot()
        return {"ok": True, "paused": False}

    @app.delete(f"{P}/jobs/{{job_id}}")
    def delete_job(job_id: str):
        with mgr.lock:
            j = mgr.jobs.get(job_id)
            if j and j.get("status") in ("queued", "running"):
                mgr._cancel.add(job_id)
                if j.get("status") == "queued":
                    j["status"] = "canceled"
                    j["message"] = "Canceled"
            elif j:
                mgr.jobs.pop(job_id, None)
        mgr._save_jobs()
        mgr.broadcast_snapshot()
        return {"ok": True}

    @app.post(f"{P}/cancel_queued")
    def cancel_queued():
        with mgr.lock:
            for j in mgr.jobs.values():
                if j.get("status") == "queued":
                    j["status"] = "canceled"
                    j["message"] = "Canceled"
                    mgr._cancel.add(j["id"])
        mgr._save_jobs()
        mgr.broadcast_snapshot()
        return {"ok": True}

    @app.post(f"{P}/retry_failed")
    def retry_failed():
        with mgr.lock:
            failed = [j for j in mgr.jobs.values() if j.get("status") == "failed"]
        for j in failed:
            mgr.enqueue(j["kind"], j["filename"])
        return {"ok": True, "retried": len(failed)}

    @app.post(f"{P}/clear_finished")
    def clear_finished():
        with mgr.lock:
            for jid in [j["id"] for j in mgr.jobs.values() if j.get("status") in ("done", "failed", "canceled")]:
                mgr.jobs.pop(jid, None)
        mgr._save_jobs()
        mgr.broadcast_snapshot()
        return {"ok": True}

    # ── websocket ────────────────────────────────────────────────────────────
    @app.websocket(f"{P}/events")
    async def events(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue()
        mgr._clients.add(q)
        try:
            await ws.send_json(mgr.snapshot())
            while True:
                msg = await q.get()
                await ws.send_json(msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            mgr._clients.discard(q)

    log.info("stem_splitter: routes registered")

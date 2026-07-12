"""Stem Splitter — backend routes, job queue, and engine orchestration.

All work is namespaced under ``/api/plugins/stem_splitter/``. Heavy work (HTTP to
a split/transcribe server, ffmpeg, zip repack, pip installs) runs on a background
worker thread — never inside an ``async def`` handler — so it can't block the
event loop. ``setup()`` imports nothing heavy and does no network I/O; its only
disk touch is a fast marker check for a deferred engine uninstall (a no-op unless
the user requested an uninstall that was blocked by locked files last session, in
which case it removes the already-orphaned engine dir before anything re-imports it).
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

import demucs_server
import engine_install

INSTRUMENT_STEM_IDS = ["guitar", "bass", "drums", "vocals", "other", "piano"]
_BROADCAST_MIN_INTERVAL = 0.15  # s — throttle progress spam


class JobCanceled(Exception):
    """Raised by a job's cancel checkpoint so an in-flight split/transcribe
    unwinds cleanly and the worker marks it ``canceled`` (not ``failed``)."""


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
        # Same idea for the managed demucs-server lifecycle (install / start /
        # prepare-models), so a reconnecting settings page recovers its state.
        self._server: dict | None = None

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
            # Managed local demucs server. autostart is on by default but is a
            # no-op until the server is actually installed, and start never pulls
            # weights (see demucs_server.start_server).
            "local_server_port": demucs_server.DEFAULT_PORT,
            "local_server_autostart": True,
            "local_server_use_globally": False,
            "local_server_device": "",   # "" = auto | cpu | cuda
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

    def local_server_url(self) -> str | None:
        """URL of the plugin-managed local server, if it's actually running."""
        s = self.read_settings()
        port = int(s.get("local_server_port") or demucs_server.DEFAULT_PORT)
        running, live_port = demucs_server.is_running(self.config_dir, port)
        return demucs_server.url_for(int(live_port or port)) if running else None

    def _server_url(self) -> str | None:
        """Split server URL (demucs/roformer).

        The plugin-managed local server wins whenever it's running — that's the
        "plugin uses it automatically" half of the toggle, and it needs no mutation
        of the app's config.json (so there's nothing to clean up when it stops).
        The app's `demucs_server_url` remains the fallback.
        """
        local = self.local_server_url()
        if local:
            return local
        cfg = self._app_config()
        url = cfg.get("demucs_server_url")
        if isinstance(url, str) and url.strip():
            return url.strip().rstrip("/")
        return None

    def _lyrics_server_url(self) -> str | None:
        """Lyrics server URL. Prefers a dedicated ``whisperx.server_url`` (a
        separate WhisperX host is common); falls back to the split server."""
        cfg = self._app_config()
        wx = cfg.get("whisperx")
        if isinstance(wx, dict):
            u = wx.get("server_url")
            if isinstance(u, str) and u.strip():
                return u.strip().rstrip("/")
        return self._server_url()

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
        # Detect installed engines by DIRECTORY presence, not by importing them:
        # importing torch/demucs/audio-separator loads native DLLs and locks the
        # engine files on Windows, which then makes Uninstall silently no-op. A
        # dir check is enough for gating; the real import still happens at run time.
        inst = engine_install.installed_map(self.config_dir)
        # Gate local engines on torch too: an engine package present without a
        # working torch in the target dir would fail immediately at run time.
        torch_ok = bool(inst.get("torch"))
        as_ok = bool(inst.get("audio-separator")) and torch_ok
        demucs_ok = bool(inst.get("demucs")) and torch_ok

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
        server_url = self._lyrics_server_url()
        # Dir-based detection (see resolve_split_engine) — avoids importing whisperx
        # / torch just to check availability, so viewing settings doesn't lock the
        # engine files against Uninstall.
        inst = engine_install.installed_map(self.config_dir)
        # whisperx also needs torch present in the target dir to run locally.
        wx_ok = bool(inst.get("whisperx")) and bool(inst.get("torch"))

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
    def _song_label(self, filename: str) -> tuple[str | None, str | None]:
        """Best-effort (title, artist) for the queue display. Fast — extract_meta
        only reads the manifest. Falls back to (None, None) so the UI shows the
        filename."""
        try:
            if self.extract_meta and self.get_dlc_dir and self.get_dlc_dir():
                m = self.extract_meta(self._resolve_pak(filename)) or {}
                return (m.get("title") or None, m.get("artist") or None)
        except Exception:
            pass
        return (None, None)

    def enqueue(self, kind: str, filename: str) -> dict:
        title, artist = self._song_label(filename)
        job = {
            "id": uuid.uuid4().hex[:12],
            "kind": kind,
            "filename": filename,
            "title": title,
            "artist": artist,
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

    def _check_cancel(self, job_id: str) -> None:
        """Cancellation checkpoint — raises if the job was asked to cancel."""
        if job_id in self._cancel:
            raise JobCanceled()

    def _make_progress_cb(self, job_id: str, base: float = 0.0, span: float = 1.0):
        def cb(p: float, message: str):
            self._check_cancel(job_id)  # every progress tick is a cancel point
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
                # A job canceled while still queued already had its status flipped;
                # discard its id here so it doesn't leak in `_cancel` forever.
                self._cancel.discard(job_id)
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
            except JobCanceled:
                self._update(job_id, status="canceled", progress=0.0, message="Canceled")
            except Exception as e:
                self.log.exception("stem_splitter: job %s failed", job_id)
                self._update(job_id, status="failed", error=str(e), message=f"Failed: {e}")
            finally:
                self._cancel.discard(job_id)
            self._save_jobs()

    def _run_job(self, job: dict) -> None:
        import split_stems
        import transcribe

        filename = job["filename"]
        pak_path = self._resolve_pak(filename)
        cb = self._make_progress_cb(job["id"])
        cancel_cb = lambda: self._check_cancel(job["id"])  # noqa: E731
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
                engine_dir=edir, models_dir=mdir, progress_cb=cb, cancel_cb=cancel_cb,
            )
        elif job["kind"] == "transcribe":
            lyr_engine, lyr_reason = self.resolve_lyrics_engine()
            if not lyr_engine:
                raise RuntimeError(lyr_reason)
            lyr_server = self._lyrics_server_url()
            split_engine, _ = self.resolve_split_engine()
            split_kwargs = {
                "engine": split_engine, "server_url": server_url, "api_key": api_key,
                "engine_dir": edir, "models_dir": mdir,
                "model": settings.get("remote_model") if split_engine != "demucs" else None,
            } if split_engine else None
            self._update(job["id"], message=f"Transcribing via {lyr_reason}")
            transcribe.transcribe_pak(
                pak_path, mode=lyr_engine, server_url=lyr_server, api_key=api_key,
                whisperx_model=settings.get("whisperx_model", "medium"),
                language=settings.get("language") or None,
                engine_dir=edir, models_dir=mdir, split_kwargs=split_kwargs,
                cancel_cb=cancel_cb, progress_cb=cb,
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
                "install": self._install, "server": self._server}

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

    # ── managed demucs server ────────────────────────────────────────────────
    def run_server_op(self, op: str, fn) -> None:
        """Run a demucs-server lifecycle op (install / start / prepare_models) on a
        daemon thread, streaming progress over the WS. Same contract as the engine
        install, so the settings page reuses the exact same widgets."""
        def _run() -> None:
            def cb(ev: dict) -> None:
                st = {"active": True, "op": op, "line": ev.get("line", ""),
                      "pct": ev.get("pct", 0.0), "phase": ev.get("phase", "")}
                self._server = st
                self.push_event({"type": "server", **st})
            try:
                status = fn(cb)
                self._server = {"active": False, "op": op, "pct": 1.0, "phase": "Done"}
                self.push_event({"type": "server_done", "op": op, "status": status})
            except Exception as e:
                self.log.warning("stem_splitter: server op %s failed: %s", op, e)
                self._server = {"active": False, "op": op, "pct": 0.0,
                                "phase": "Failed", "error": str(e)}
                self.push_event({"type": "server_error", "op": op, "error": str(e)})
            finally:
                # The server's log reader outlives the op. Detach it now so its
                # ongoing output can't keep pushing progress events (which would
                # flip the UI back to "active" and re-disable the controls after
                # we've already reported the op as done).
                demucs_server.clear_stream_cb()
        threading.Thread(target=_run, name=f"stem_splitter-server-{op}", daemon=True).start()

    def needs_server_setup(self) -> dict | None:
        """If the split would go to our managed local server but its weights aren't
        downloaded yet, say so instead of letting the job silently stall on a ~2 GB
        lazy fetch. The UI turns this into a 'download now?' prompt."""
        # Only when the job would ACTUALLY go through the server. A user who forced
        # a local engine (demucs / audio-separator) doesn't need the server's models
        # at all, and must not be blocked just because the server happens to be
        # running without them.
        engine, _reason = self.resolve_split_engine()
        if engine != "remote":
            return None
        if not self.local_server_url():
            return None   # a real remote server: not ours to set up
        if demucs_server.models_downloaded(self.config_dir):
            return None
        return {
            "needs_setup": True,
            "message": "The local demucs server is running, but its models "
                       "haven't been downloaded yet (~2 GB, one time). "
                       "Download them now?",
        }


def setup(app: FastAPI, context: dict) -> None:
    # Finish any uninstall that was deferred last session because the engine's
    # native DLLs were locked. Do this first, before anything can import from the
    # engine dir again.
    _log = context.get("log") or logging.getLogger("feedBack.plugin.stem_splitter")
    try:
        if engine_install.apply_pending_uninstall(Path(context["config_dir"])):
            _log.info("stem_splitter: applied pending engine uninstall on startup")
    except Exception as e:
        # Don't let a cleanup hiccup block plugin load, but leave a diagnostic —
        # otherwise the user sees "installed" after a restart with no explanation.
        _log.warning("stem_splitter: pending engine uninstall failed on startup: %s", e)

    mgr = JobManager(app, context)
    log = mgr.log

    try:
        # setup() is marshalled onto the event-loop thread by the host, so the
        # running loop is the uvicorn loop the worker thread must push onto.
        mgr._loop = asyncio.get_running_loop()
    except RuntimeError:
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

    # ── managed demucs server ────────────────────────────────────────────────
    def _server_opts() -> tuple[int, str]:
        s = mgr.read_settings()
        port = int(s.get("local_server_port") or demucs_server.DEFAULT_PORT)
        device = str(s.get("local_server_device") or "")
        return port, device

    @app.get(f"{P}/server_status")
    def get_server_status():
        return demucs_server.server_status(mgr.config_dir)

    @app.get(f"{P}/server/health")
    def get_server_health():
        """Backend proxy for the 'Test status' button. /health needs no API key."""
        port, _ = _server_opts()
        url = mgr.local_server_url() or demucs_server.url_for(port)
        ok, payload = demucs_server.server_health(url, timeout=4.0)
        return {"ok": ok, "url": url, "health": payload}

    @app.post(f"{P}/server/install")
    def post_server_install():
        mgr.run_server_op("install", lambda cb: demucs_server.install_server(
            mgr.config_dir, progress_cb=cb))
        return {"ok": True, "started": "install"}

    @app.post(f"{P}/server/start")
    def post_server_start():
        port, device = _server_opts()
        # warmup=None -> warm up only if the weights are already on disk, so a
        # start can never trigger the big download.
        mgr.run_server_op("start", lambda cb: demucs_server.start_server(
            mgr.config_dir, port=port, device=device, warmup=None, progress_cb=cb))
        return {"ok": True, "started": "start"}

    @app.post(f"{P}/server/stop")
    def post_server_stop():
        return demucs_server.stop_server(mgr.config_dir)

    @app.post(f"{P}/server/prepare_models")
    def post_server_prepare_models():
        port, device = _server_opts()
        mgr.run_server_op("prepare_models", lambda cb: demucs_server.prepare_models(
            mgr.config_dir, port=port, device=device, progress_cb=cb))
        return {"ok": True, "started": "prepare_models"}

    @app.post(f"{P}/server/uninstall")
    def post_server_uninstall():
        return demucs_server.uninstall_server(mgr.config_dir)

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
        # Warn-and-ask rather than silently stalling on a lazy ~2 GB model fetch.
        # The client re-POSTs with skip_setup_check once the user has agreed (and the
        # models have been prepared).
        if not body.get("skip_setup_check"):
            needs = mgr.needs_server_setup()
            if needs:
                return {**needs, "ok": False, "enqueued": 0}
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
        out: list = []
        page, size = 0, 500
        try:
            while True:
                songs, total = mgr.meta_db.query_page(page=page, size=size, **kwargs)
                if not songs:
                    break
                out.extend(songs)
                if len(songs) < size:
                    break  # short page = last page (correct even if `total` is None)
                if isinstance(total, int) and len(out) >= total:
                    break
                page += 1
                if page > 2000:  # backstop (~1M rows) so a bad `total` can't spin
                    log.warning("stem_splitter: query truncated at %d rows", len(out))
                    break
        except Exception as e:
            log.warning("stem_splitter: query_page failed: %s", e)
        return out

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

    # ── auto-start the managed server ────────────────────────────────────────
    # Entirely on a daemon thread: setup() must never block app launch. And the
    # start itself only warms up models that are ALREADY downloaded, so launching
    # can never kick off the ~2 GB fetch (see demucs_server.start_server).
    def _autostart() -> None:
        try:
            s = mgr.read_settings()
            if not s.get("local_server_autostart"):
                return
            manageable, reason = demucs_server.can_manage(mgr.config_dir)
            if not manageable:
                log.info("stem_splitter: not auto-starting a local server here (%s)", reason)
                return
            if not demucs_server.installed(mgr.config_dir):
                return  # nothing installed -> nothing to start
            port, device = _server_opts()
            running, _ = demucs_server.is_running(mgr.config_dir, port)
            if running:
                log.info("stem_splitter: demucs server already running on %s", port)
                return
            log.info("stem_splitter: auto-starting demucs server on port %s", port)
            mgr.run_server_op("start", lambda cb: demucs_server.start_server(
                mgr.config_dir, port=port, device=device, warmup=None, progress_cb=cb))
        except Exception as e:
            log.warning("stem_splitter: demucs server auto-start failed: %s", e)

    threading.Thread(target=_autostart, name="stem_splitter-autostart", daemon=True).start()

    log.info("stem_splitter: routes registered")

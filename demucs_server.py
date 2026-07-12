"""Managed local demucs server for the Stem Splitter plugin.

Downloads, installs, starts, stops and health-checks
`got-feedBack/feedBack-demucs-server` so a user can get a working split server
without touching a terminal.

Nothing here runs on plugin load, and nothing large is EVER downloaded implicitly:

* ``install_server()``  — only from the explicit "Install server" button (a venv +
  a few GB of wheels).
* ``prepare_models()``  — only from the explicit "Prepare models" button (the
  ~2 GB of model weights).
* ``start_server()``    — cheap. Warms up ONLY when the weights are already on
  disk (that's a RAM load, not a network pull); otherwise starts with
  ``--skip-warmup`` so launching can never trigger a big download.

Layout under ``{config_dir}/demucs-server/``::

    src/      server.py, run_demucs.py, run_roformer.py, requirements.txt
    .venv/    the server's own interpreter (its torch/whisperx pins conflict with
              the plugin's `pip --target engine/` tree, so it gets its own)
    cache/    SLOPSMITH_DEMUCS_CACHE + TORCH_HOME + HF_HOME all point here, so the
              weights are ours to detect and ours to delete on uninstall (instead
              of being orphaned in ~/.cache).
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

import engine_install

log = logging.getLogger("feedBack.plugin.stem_splitter")

ProgressCB = Optional[Callable[[dict], None]]

SOURCE_ZIP_URL = "https://codeload.github.com/got-feedBack/feedBack-demucs-server/zip/refs/heads/main"
# The only files the server actually needs to run.
SOURCE_FILES = ("server.py", "run_demucs.py", "run_roformer.py", "requirements.txt")

DEFAULT_PORT = 7865
DEFAULT_MODEL = "bs_roformer_sw"   # what the plugin splits with; also makes warmup prefetch it

# demucs must be installed --no-deps: it pins torchaudio<2.1 while whisperx needs
# ~2.8, and its full dep set drags in `diffq` (a C-extension with no wheels, which
# is exactly what broke a tester's install). These are demucs' real runtime deps.
_DEMUCS_EXTRAS = ["einops", "julius", "lameenc", "openunmix", "pyyaml", "tqdm", "dora-search"]

# Track a server we started in THIS process, so we can stream its output and reap it.
_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()


# ── paths ────────────────────────────────────────────────────────────────────

def server_dir(config_dir: Path) -> Path:
    return Path(config_dir) / "demucs-server"


def src_dir(config_dir: Path) -> Path:
    return server_dir(config_dir) / "src"


def venv_dir(config_dir: Path) -> Path:
    return server_dir(config_dir) / ".venv"


def cache_dir(config_dir: Path) -> Path:
    return server_dir(config_dir) / "cache"


def state_file(config_dir: Path) -> Path:
    return Path(config_dir) / "stem_splitter_server.json"


def venv_python(config_dir: Path) -> Path:
    v = venv_dir(config_dir)
    return v / "Scripts" / "python.exe" if os.name == "nt" else v / "bin" / "python"


def installed(config_dir: Path) -> bool:
    """Source fetched AND a venv interpreter exists."""
    return (src_dir(config_dir) / "server.py").is_file() and venv_python(config_dir).is_file()


def models_downloaded(config_dir: Path) -> bool:
    """Are the split model weights already on disk?

    This is the gate that decides warmup-vs-skip-warmup at auto-start: warming up
    weights we already have is a RAM load (fine at launch); warming up weights we
    DON'T have would pull ~2 GB (never at launch).

    We key off the roformer checkpoint because ``bs_roformer_sw`` is the model the
    plugin actually splits with.
    """
    roformer = cache_dir(config_dir) / "_roformer-models"
    if roformer.is_dir() and any(roformer.glob("*.ckpt")):
        return True
    # Fall back to a demucs torch-hub checkpoint (TORCH_HOME -> cache/).
    hub = cache_dir(config_dir) / "hub" / "checkpoints"
    return hub.is_dir() and any(hub.glob("*.th"))


# ── download + install (explicit, heavy) ─────────────────────────────────────

def _emit(progress_cb: ProgressCB, line: str, pct: float, phase: str) -> None:
    if progress_cb:
        progress_cb({"line": line, "pct": max(0.0, min(1.0, pct)), "phase": phase})


def download_source(config_dir: Path, progress_cb: ProgressCB = None) -> None:
    """Fetch the server source from GitHub (no `git` required) and extract the
    handful of files it needs."""
    import requests

    sdir = src_dir(config_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    _emit(progress_cb, f"Downloading server source from {SOURCE_ZIP_URL}", 0.02, "Downloading source")

    resp = requests.get(SOURCE_ZIP_URL, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"failed to download server source ({resp.status_code})")

    got = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for member in zf.namelist():
            name = member.rsplit("/", 1)[-1]
            if name in SOURCE_FILES and not member.endswith("/"):
                (sdir / name).write_bytes(zf.read(member))
                got.append(name)
                _emit(progress_cb, f"  extracted {name}", 0.06, "Downloading source")

    missing = [f for f in SOURCE_FILES if f not in got]
    if missing:
        raise RuntimeError(f"server source archive is missing {missing}")
    _emit(progress_cb, f"Source ready ({len(got)} files).", 0.08, "Downloading source")


def install_server(config_dir: Path, progress_cb: ProgressCB = None) -> dict:
    """Download the source, create a venv, and install the server's dependencies.

    Explicit-only (the "Install server" button). Several GB of wheels.
    """
    server_dir(config_dir).mkdir(parents=True, exist_ok=True)
    cache_dir(config_dir).mkdir(parents=True, exist_ok=True)

    download_source(config_dir, progress_cb)

    vpy = venv_python(config_dir)
    if not vpy.is_file():
        _emit(progress_cb, "Creating virtual environment…", 0.10, "venv")
        proc = subprocess.run([sys.executable, "-m", "venv", str(venv_dir(config_dir))],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"failed to create venv: {(proc.stderr or proc.stdout)[:300]}")
    if not vpy.is_file():
        raise RuntimeError(f"venv python not found at {vpy}")

    req = src_dir(config_dir) / "requirements.txt"
    # Steps run in their own pip transactions, sharing the plugin's streaming
    # installer (progress bar + actionable failure hints).
    steps: list[tuple[str, list[str], int]] = [
        ("pip", ["--upgrade", "pip"], 1),
        ("server requirements", ["-r", str(req)], 9),
        # --no-deps is load-bearing: it keeps demucs from dragging in diffq (no
        # wheel, needs a C++ compiler) and from downgrading torchaudio under whisperx.
        ("demucs (no-deps)", ["--no-deps", "demucs"], 1),
        ("demucs runtime deps", list(_DEMUCS_EXTRAS), len(_DEMUCS_EXTRAS)),
    ]
    n = len(steps)
    for i, (label, args, count) in enumerate(steps):
        engine_install.stream_pip(str(vpy), args, label, progress_cb,
                                  base=0.12 + (i / n) * 0.86, span=0.86 / n, pkg_count=count)

    _emit(progress_cb, "Server installed.", 1.0, "Done")
    return server_status(config_dir)


def uninstall_server(config_dir: Path) -> dict:
    """Stop the server and delete its source, venv and downloaded weights."""
    try:
        stop_server(config_dir)
    except Exception as e:
        log.warning("stem_splitter: stop before uninstall failed: %s", e)
    shutil.rmtree(server_dir(config_dir), ignore_errors=True)
    try:
        state_file(config_dir).unlink(missing_ok=True)
    except OSError as e:
        log.warning("stem_splitter: could not clear server state file: %s", e)
    return server_status(config_dir)


# ── lifecycle ────────────────────────────────────────────────────────────────

def _read_state(config_dir: Path) -> dict:
    try:
        data = json.loads(state_file(config_dir).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_state(config_dir: Path, data: dict) -> None:
    try:
        state_file(config_dir).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("stem_splitter: could not write server state file: %s", e)


def url_for(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def server_health(url: str, timeout: float = 3.0) -> tuple[bool, dict]:
    """GET /health. Exempt from the server's API-key auth, so no key needed."""
    import requests
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=timeout)
        if r.status_code == 200:
            payload = r.json()
            return True, payload if isinstance(payload, dict) else {}
        return False, {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


def _server_env(config_dir: Path) -> dict:
    """Subprocess env: keep every weight cache inside our own dir, and make sure
    ffmpeg (a hard prerequisite of the server) is on PATH by reusing the one the
    feedBack app already bundles."""
    env = dict(os.environ)
    cache = cache_dir(config_dir)
    cache.mkdir(parents=True, exist_ok=True)
    env["SLOPSMITH_DEMUCS_CACHE"] = str(cache)
    env["TORCH_HOME"] = str(cache)
    env["HF_HOME"] = str(cache / "huggingface")
    env["HUGGINGFACE_HUB_CACHE"] = str(cache / "huggingface" / "hub")
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        from audio import _ffmpeg_cmd  # feedBack's bundled ffmpeg resolver
        ff = _ffmpeg_cmd()
        if ff:
            ffdir = str(Path(ff).parent)
            env["PATH"] = ffdir + os.pathsep + env.get("PATH", "")
    except Exception as e:
        log.warning("stem_splitter: could not resolve bundled ffmpeg (%s); the "
                    "server needs ffmpeg on PATH", e)
    return env


def is_running(config_dir: Path, port: int | None = None) -> tuple[bool, int | None]:
    """(running, port). True if we own a live child, or if a previously-recorded
    port still answers /health (an orphan from a crashed app)."""
    with _proc_lock:
        p = _proc
    if p is not None and p.poll() is None:
        st = _read_state(config_dir)
        return True, st.get("port", port)

    st = _read_state(config_dir)
    known = port or st.get("port")
    if known:
        ok, _ = server_health(url_for(int(known)), timeout=1.5)
        if ok:
            return True, int(known)
    return False, None


def start_server(config_dir: Path, port: int = DEFAULT_PORT, device: str = "",
                 model: str = DEFAULT_MODEL, warmup: bool | None = None,
                 progress_cb: ProgressCB = None) -> dict:
    """Start the server as a managed subprocess.

    ``warmup=None`` (the default) decides for you: warm up **only if the weights
    are already downloaded**. That way a start is fast and can never trigger the
    ~2 GB fetch, but once you've paid for the download the server comes up warm.
    """
    global _proc

    if not installed(config_dir):
        raise RuntimeError("demucs server is not installed - click 'Install server' first")

    running, live_port = is_running(config_dir, port)
    if running:
        _emit(progress_cb, f"Server already running on port {live_port}.", 1.0, "Running")
        return server_status(config_dir)

    if warmup is None:
        warmup = models_downloaded(config_dir)

    cmd = [str(venv_python(config_dir)), str(src_dir(config_dir) / "server.py"),
           "--port", str(port), "--host", "127.0.0.1", "--model", model]
    if device:
        cmd += ["--device", device]
    if not warmup:
        # No weights on disk yet -> never pull them implicitly at start.
        cmd.append("--skip-warmup")

    _emit(progress_cb, f"Starting server on port {port} "
                       f"({'warming up cached models' if warmup else 'skip-warmup'})…",
          0.05, "Starting")

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc = subprocess.Popen(
        cmd, cwd=str(src_dir(config_dir)), env=_server_env(config_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, creationflags=creationflags,
    )
    with _proc_lock:
        _proc = proc
    _write_state(config_dir, {"pid": proc.pid, "port": port, "started_at": time.time()})

    def _reader() -> None:
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                low = line.lower()
                phase = "Running"
                pct = 0.6
                if "[warmup]" in low:
                    phase = "Warming up"
                    if "ready" in low:
                        pct = 0.9
                _emit(progress_cb, line, pct, phase)
        except Exception:
            pass
        finally:
            rc = proc.poll()
            _emit(progress_cb, f"Server process exited (code {rc}).", 0.0, "Stopped")

    threading.Thread(target=_reader, name="stem_splitter-server-log", daemon=True).start()

    # Wait briefly for the port to bind so the caller gets a truthful status.
    url = url_for(port)
    for _ in range(40):  # ~20s
        if proc.poll() is not None:
            raise RuntimeError(f"server exited immediately (code {proc.returncode}) - check the log")
        ok, _payload = server_health(url, timeout=1.0)
        if ok:
            _emit(progress_cb, f"Server is up at {url}", 1.0, "Running")
            return server_status(config_dir)
        time.sleep(0.5)

    raise RuntimeError(f"server did not answer /health on {url} within 20s")


def stop_server(config_dir: Path) -> dict:
    """Kill the server AND its children (it spawns run_demucs.py / run_roformer.py
    workers - terminating only the parent would orphan them)."""
    global _proc

    with _proc_lock:
        p = _proc
        _proc = None

    pid = None
    if p is not None and p.poll() is None:
        pid = p.pid
    else:
        pid = _read_state(config_dir).get("pid")

    if pid:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                               capture_output=True, text=True)
            else:
                import signal
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except Exception:
                    os.kill(pid, signal.SIGTERM)
                time.sleep(1.0)
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
        except Exception as e:
            log.warning("stem_splitter: failed to kill server pid %s: %s", pid, e)

    if p is not None:
        try:
            p.wait(timeout=5)  # reap
        except Exception:
            pass

    try:
        state_file(config_dir).unlink(missing_ok=True)
    except OSError as e:
        log.warning("stem_splitter: could not clear server state file: %s", e)

    return server_status(config_dir)


def prepare_models(config_dir: Path, port: int = DEFAULT_PORT, device: str = "",
                   model: str = DEFAULT_MODEL, progress_cb: ProgressCB = None) -> dict:
    """The explicit ~2 GB weight download.

    Restarts the server WITH warmup (and with the roformer model as the default,
    so its checkpoint is prefetched too), then polls /health until it's ready.
    """
    _emit(progress_cb, "Preparing models (this downloads ~2 GB once)…", 0.02, "Preparing")
    try:
        stop_server(config_dir)
    except Exception:
        pass

    start_server(config_dir, port=port, device=device, model=model,
                 warmup=True, progress_cb=progress_cb)

    url = url_for(port)
    deadline = time.time() + 60 * 60  # weights on a slow link can take a while
    while time.time() < deadline:
        ok, payload = server_health(url, timeout=5)
        if ok:
            wu = payload.get("warmup") or {}
            states = [str(v) for v in wu.values() if isinstance(v, str)]
            if any(s == "failed" for s in states):
                raise RuntimeError(f"model warmup failed: {wu}")
            if _model_ready(wu, model):
                _emit(progress_cb, "Models ready.", 1.0, "Done")
                return server_status(config_dir)
            _emit(progress_cb, f"warmup: {wu}", 0.6, "Downloading models")
        time.sleep(5)

    raise RuntimeError("timed out waiting for model warmup")


def _model_ready(warmup: dict, model: str) -> bool:
    v = warmup.get(model) or warmup.get("demucs")
    return str(v) in ("ready", "skipped")


def server_status(config_dir: Path) -> dict:
    st = _read_state(config_dir)
    port = int(st.get("port") or DEFAULT_PORT)
    running, live_port = is_running(config_dir, port)
    port = int(live_port or port)
    url = url_for(port)

    health: dict = {}
    models_ready = False
    if running:
        ok, health = server_health(url, timeout=2.0)
        if ok:
            models_ready = _model_ready(health.get("warmup") or {}, DEFAULT_MODEL)

    return {
        "installed": installed(config_dir),
        "running": running,
        "pid": st.get("pid"),
        "port": port,
        "url": url if running else None,
        "health": health,
        "models_downloaded": models_downloaded(config_dir),
        "models_ready": models_ready,
        "server_dir": str(server_dir(config_dir)),
        "disk_bytes": engine_install._dir_size(server_dir(config_dir)),
    }

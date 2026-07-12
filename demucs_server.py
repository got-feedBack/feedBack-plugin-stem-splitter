"""Managed local demucs server for the Stem Splitter plugin.

Downloads, installs, starts, stops and health-checks
`got-feedBack/feedBack-demucs-server` so a user can get a working split server
without touching a terminal.

Nothing here runs on plugin load, and nothing large is EVER downloaded implicitly:

* ``install_server()``  — only from the explicit "Install server" button (a few GB of
  a few GB of wheels).
* ``prepare_models()``  — only from the explicit "Prepare models" button (the
  ~2 GB of model weights).
* ``start_server()``    — cheap. Warms up ONLY when the weights are already on
  disk (that's a RAM load, not a network pull); otherwise starts with
  ``--skip-warmup`` so launching can never trigger a big download.

Layout under ``{config_dir}/demucs-server/``::

    src/      server.py, run_demucs.py, run_roformer.py, requirements.txt, _launch.py
    pylibs/   the server's dependencies (pip --target)
    cache/    SLOPSMITH_DEMUCS_CACHE + TORCH_HOME + HF_HOME all point here, so the
              weights are ours to detect and ours to delete on uninstall (instead
              of being orphaned in ~/.cache).

**Why ``pip --target`` and not a venv.** The packaged app bundles its own Python,
and on Windows that is the python.org *embeddable* distribution, which ships **no
``venv`` and no ``ensurepip``** (see feedback-desktop/scripts/build-windows.sh -
it has to bootstrap pip with get-pip.py for exactly this reason). So
``python -m venv`` is guaranteed to fail on packaged Windows. ``pip install
--target`` needs neither module and works identically on Windows, macOS and Linux,
packaged or dev - it's the same approach the core plugin loader and this plugin's
own engine installer already use.

The deps still land in their own tree, isolated from the app's site-packages, and
the server is launched through a generated ``_launch.py`` that prepends that tree
to ``sys.path``. That in-process injection matters: packaged Windows runs in
isolated ``._pth`` mode where **PYTHONPATH is ignored**, so wiring the deps in via
the environment would silently not work.
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

SOURCE_REPO = "got-feedBack/feedBack-demucs-server"
# Ref to install from. Resolved to an immutable commit SHA before download, so an
# install is reproducible and we can report exactly what's on disk - a bare branch
# zip is a moving target (the same click could install different code next week).
SOURCE_REF = os.environ.get("STEM_SPLITTER_SERVER_REF", "main")
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


def pylibs_dir(config_dir: Path) -> Path:
    """The server's dependency tree (pip --target). Portable across every packaged
    platform, unlike a venv (see the module docstring)."""
    return server_dir(config_dir) / "pylibs"


def launcher_path(config_dir: Path) -> Path:
    return src_dir(config_dir) / "_launch.py"


def cache_dir(config_dir: Path) -> Path:
    return server_dir(config_dir) / "cache"


def state_file(config_dir: Path) -> Path:
    return Path(config_dir) / "stem_splitter_server.json"


def installed(config_dir: Path) -> bool:
    """Source fetched AND the dependency tree is populated."""
    if not (src_dir(config_dir) / "server.py").is_file():
        return False
    p = pylibs_dir(config_dir)
    # uvicorn is the one import the server can't start without.
    return p.is_dir() and (any(p.glob("uvicorn")) or any(p.glob("uvicorn-*.dist-info")))


# ── can we manage a server in this deployment at all? ────────────────────────

def in_container() -> bool:
    """Are we inside Docker/Podman/k8s?

    Not a blocker - the plugin and the server would share the container, so
    127.0.0.1 still resolves and a writable CONFIG_DIR volume persists. It's just
    worth telling the user, since the app container usually has no GPU (CPU-only
    separation is slow) and an unmounted config dir would be ephemeral.
    """
    if os.environ.get("container") or Path("/.dockerenv").exists():
        return True
    try:
        cg = Path("/proc/1/cgroup")
        if cg.exists():
            txt = cg.read_text(errors="ignore")
            if any(m in txt for m in ("docker", "containerd", "kubepods", "podman")):
                return True
    except OSError:
        pass
    return False


def _python_has_pip(python_exe: str) -> bool:
    try:
        p = subprocess.run([python_exe, "-m", "pip", "--version"],
                           capture_output=True, text=True, timeout=60)
        return p.returncode == 0
    except Exception:
        return False


def can_manage(config_dir: Path) -> tuple[bool, str]:
    """Can this deployment install and run a managed local server?

    Deliberately does NOT require ``venv``: the packaged Windows app bundles the
    python.org embeddable distribution, which has no venv/ensurepip at all. We only
    need ``pip`` (present on every packaged platform - the Windows build bootstraps
    it with get-pip.py) and a writable config dir.
    """
    if not _python_has_pip(sys.executable):
        return (False,
                f"This build's Python ({sys.executable}) has no usable 'pip', so the "
                "server's dependencies can't be installed. Run the demucs server "
                "separately and set 'demucs_server_url' instead.")

    try:
        Path(config_dir).mkdir(parents=True, exist_ok=True)
        probe = Path(config_dir) / ".stem_splitter_write_probe"
        probe.write_text("1", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as e:
        return (False, f"The config directory isn't writable ({e}), so the server "
                       "can't be installed here.")

    return (True, "")


def manage_advisory(config_dir: Path) -> str:
    """Non-blocking warnings to surface next to the controls."""
    if in_container():
        return ("The app is running in a container: the server will run inside it "
                "too (usually CPU-only, and it needs a persistent CONFIG_DIR "
                "volume). Running the demucs-server container separately and "
                "pointing 'demucs_server_url' at it is often the better option.")
    return ""


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


def source_meta(config_dir: Path) -> dict:
    """What source is actually installed (ref + commit)."""
    try:
        data = json.loads((server_dir(config_dir) / "source.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_commit(ref: str) -> str | None:
    """Resolve a ref to an immutable commit SHA. None if GitHub can't be reached
    (we then fall back to the branch archive rather than failing the install)."""
    import requests
    try:
        r = requests.get(f"https://api.github.com/repos/{SOURCE_REPO}/commits/{ref}",
                         headers={"Accept": "application/vnd.github.sha"}, timeout=30)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip()
    except Exception as e:
        log.warning("stem_splitter: could not resolve %s@%s to a commit: %s",
                    SOURCE_REPO, ref, e)
    return None


def download_source(config_dir: Path, progress_cb: ProgressCB = None) -> None:
    """Fetch the server source from GitHub (no `git` required) and extract the
    handful of files it needs.

    Pins to a resolved commit so the install is reproducible and reportable; falls
    back to the branch archive only if the SHA can't be resolved.
    """
    import requests

    sdir = src_dir(config_dir)
    sdir.mkdir(parents=True, exist_ok=True)

    commit = _resolve_commit(SOURCE_REF)
    archive = commit or SOURCE_REF
    url = f"https://codeload.github.com/{SOURCE_REPO}/zip/{archive}"
    _emit(progress_cb,
          f"Downloading server source {SOURCE_REF}"
          + (f" @ {commit[:8]}" if commit else " (unpinned - could not resolve a commit)"),
          0.02, "Downloading source")

    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"failed to download server source ({resp.status_code}) from {url}")

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

    try:
        (server_dir(config_dir) / "source.json").write_text(
            json.dumps({"repo": SOURCE_REPO, "ref": SOURCE_REF, "commit": commit,
                        "installed_at": time.time()}, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("stem_splitter: could not record installed source revision: %s", e)

    _emit(progress_cb, f"Source ready ({len(got)} files).", 0.08, "Downloading source")


_LAUNCHER_TEMPLATE = '''\
# Generated by the Stem Splitter plugin - do not edit.
#
# Runs the demucs server with its pip --target dependency tree prepended to
# sys.path. This is done IN-PROCESS on purpose: the packaged Windows app uses the
# python.org embeddable distribution, which runs in isolated `._pth` mode where
# PYTHONPATH is ignored - so injecting the deps via the environment would silently
# do nothing.
import os
import runpy
import sys

PYLIBS = {pylibs!r}
HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PYLIBS)     # server's deps win over the app's site-packages
sys.path.insert(0, HERE)       # so `import run_demucs` etc. resolve
os.chdir(HERE)

server = os.path.join(HERE, "server.py")
sys.argv = [server] + sys.argv[1:]
runpy.run_path(server, run_name="__main__")
'''


def write_launcher(config_dir: Path) -> Path:
    lp = launcher_path(config_dir)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(_LAUNCHER_TEMPLATE.format(pylibs=str(pylibs_dir(config_dir))),
                  encoding="utf-8")
    return lp


def install_server(config_dir: Path, progress_cb: ProgressCB = None) -> dict:
    """Download the source and install the server's dependencies (pip --target).

    Explicit-only (the "Install server" button). Several GB of wheels.
    """
    ok, reason = can_manage(config_dir)
    if not ok:
        raise RuntimeError(reason)

    server_dir(config_dir).mkdir(parents=True, exist_ok=True)
    cache_dir(config_dir).mkdir(parents=True, exist_ok=True)
    target = pylibs_dir(config_dir)
    target.mkdir(parents=True, exist_ok=True)

    download_source(config_dir, progress_cb)

    req = src_dir(config_dir) / "requirements.txt"
    tgt = ["--target", str(target), "--upgrade", "--upgrade-strategy", "only-if-needed"]
    # Each step is its own pip transaction, sharing the plugin's streaming installer
    # (progress bar + actionable failure hints). Run with the app's own interpreter:
    # pip --target needs neither venv nor ensurepip, so this works on the packaged
    # Windows embeddable Python as well as the macOS/Linux standalone builds.
    steps: list[tuple[str, list[str], int]] = [
        ("server requirements", tgt + ["-r", str(req)], 9),
        # --no-deps is load-bearing: it keeps demucs from dragging in diffq (no
        # wheel, needs a C++ compiler) and from downgrading torchaudio under whisperx.
        ("demucs (no-deps)", tgt + ["--no-deps", "demucs"], 1),
        ("demucs runtime deps", tgt + list(_DEMUCS_EXTRAS), len(_DEMUCS_EXTRAS)),
    ]
    n = len(steps)
    for i, (label, args, count) in enumerate(steps):
        engine_install.stream_pip(sys.executable, args, label, progress_cb,
                                  base=0.12 + (i / n) * 0.86, span=0.86 / n, pkg_count=count)

    write_launcher(config_dir)
    _emit(progress_cb, "Server installed.", 1.0, "Done")
    return server_status(config_dir)


def uninstall_server(config_dir: Path) -> dict:
    """Stop the server and delete its source, dependency tree and downloaded weights."""
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
    # The app points PYTHONPATH at its own tree (feedback/, feedback/lib), which
    # would leak its modules - including a *different* server.py - into the demucs
    # server's import path. The launcher sets up sys.path itself, so drop it.
    # (PYTHONHOME stays: the bundled interpreter needs it to find its stdlib.)
    env.pop("PYTHONPATH", None)

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

    # Rewrite the launcher every start: it bakes in the pylibs path, and the config
    # dir can move between installs/platforms.
    write_launcher(config_dir)

    cmd = [sys.executable, str(launcher_path(config_dir)),
           "--port", str(port), "--host", "127.0.0.1", "--model", model]
    if device:
        cmd += ["--device", device]
    if not warmup:
        # No weights on disk yet -> never pull them implicitly at start.
        cmd.append("--skip-warmup")

    _emit(progress_cb, f"Starting server on port {port} "
                       f"({'warming up cached models' if warmup else 'skip-warmup'})…",
          0.05, "Starting")

    # Put the child in its OWN process group / session. This is load-bearing for
    # stop_server(): it kills the process *tree* (the server spawns run_demucs.py /
    # run_roformer.py workers). Without this the child stays in the host app's
    # process group, and killpg() on POSIX would signal the whole feedBack app -
    # i.e. Stop would kill the app itself.
    popen_kwargs: dict = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True  # setsid -> own session + pgid

    proc = subprocess.Popen(
        cmd, cwd=str(src_dir(config_dir)), env=_server_env(config_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, **popen_kwargs,
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


def _posix_kill_tree(pid: int) -> None:
    """TERM then KILL the server's process group, never our own.

    start_server() puts the child in its own session, so its pgid is its own. The
    guard below is a hard safety net: if for any reason the child ended up sharing
    OUR process group, killpg would take down the whole feedBack app - so in that
    case we only ever signal the single pid.
    """
    import signal

    try:
        pgid = os.getpgid(pid)
    except OSError:
        pgid = None

    own_pgid = os.getpgid(0)
    use_group = pgid is not None and pgid != own_pgid
    if pgid is not None and pgid == own_pgid:
        log.warning("stem_splitter: server pid %s shares our process group - "
                    "signalling the pid only, refusing to killpg ourselves", pid)

    def _sig(sig) -> None:
        if use_group:
            os.killpg(pgid, sig)
        else:
            os.kill(pid, sig)

    try:
        _sig(signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as e:
        log.warning("stem_splitter: SIGTERM to server failed: %s", e)

    # Give it a moment to shut down cleanly, then insist.
    for _ in range(20):  # ~2s
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except OSError:
            break
        time.sleep(0.1)

    try:
        _sig(signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


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
                               capture_output=True, text=True, timeout=30)
            else:
                _posix_kill_tree(int(pid))
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

    manageable, manage_reason = can_manage(config_dir)
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
        # False on deployments where a plugin-managed server makes no sense
        # (no pip, or a read-only config dir). The UI disables the section.
        "source": source_meta(config_dir),   # {repo, ref, commit} actually installed
        "manageable": manageable,
        "manage_reason": manage_reason,
        # Non-blocking note (e.g. "you're in a container") shown alongside the controls.
        "advisory": manage_advisory(config_dir),
    }

"""Managed local demucs server for the Stem Splitter plugin.

Downloads, installs, starts, stops and health-checks
`got-feedBack/feedBack-demucs-server` so a user can get a working split server
without touching a terminal.

Nothing here runs on plugin load, and nothing large is EVER downloaded implicitly:

* ``install_server()``  — only from the explicit "Install server" button (a few GB
  of wheels).
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
import re
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
# Default ref. Overridable per-install from the settings' Advanced section (and by
# env for headless/CI use). Resolved to an immutable commit before download.
DEFAULT_SOURCE_REF = os.environ.get("STEM_SPLITTER_SERVER_REF", "main")
SOURCE_REF = DEFAULT_SOURCE_REF   # back-compat alias
# The only files the server actually needs to run.
SOURCE_FILES = ("server.py", "run_demucs.py", "run_roformer.py", "requirements.txt")

DEFAULT_PORT = 7865
DEFAULT_MODEL = "bs_roformer_sw"   # what the plugin splits with; also makes warmup prefetch it

# demucs must be installed --no-deps: it pins torchaudio<2.1 while whisperx needs
# ~2.8, and its full dep set drags in `diffq` (see below). These are its real
# runtime deps.
# sphn is new in demucs 4.1 and `demucs/api.py` imports it at module top level, so a
# --no-deps install without it leaves demucs.api dead (and pip prints a dependency-
# conflict ERROR on every install, which is how real conflicts get lost in the noise).
_DEMUCS_EXTRAS = ["einops", "julius", "lameenc", "openunmix", "pyyaml", "tqdm",
                  "dora-search", "sphn"]

# audio-separator must be --no-deps too, for a subtler reason. Its metadata says:
#
#   Requires-Dist: diffq (>=0.2)        ; sys_platform != "win32"
#   Requires-Dist: diffq-fixed (>=0.2)  ; sys_platform == "win32"
#
# `diffq` is a C-extension whose newest wheels stop at **cp310**. On any Python 3.11+
# pip therefore falls back to its sdist and needs a C compiler — which is precisely
# the failure a Linux/macOS user hits, and the reason the demucs-server's own Docker
# image has never built (its requirements.txt lists audio-separator, so `-r` drags
# diffq in and the compile dies before the Dockerfile's careful --no-deps lines ever
# run). Windows escapes it only by accident: it resolves to `diffq-fixed`, which does
# ship wheels through cp313. Leaving audio-separator's deps to pip means the install
# works on the one platform we happened to test and fails on the other two.
_AUDIO_SEPARATOR = "audio-separator>=0.44.0"
# NOT \b: a word boundary matches before a hyphen too, so `audio-separator-extras`
# (a different distribution) would be dropped as well. Only the exact name.
_AS_REQ_RE = re.compile(r"^\s*audio[-_]separator(?![-_A-Za-z0-9])", re.I)

# audio-separator 0.44's real runtime deps, minus diffq (handled binary-only below)
# and minus everything the main resolve already brings in through requirements.txt
# (torch, librosa, soundfile, numpy, scipy) or _DEMUCS_EXTRAS (einops, julius, pyyaml,
# tqdm). Keep in sync when the pinned audio-separator moves; verify_install()'s import
# check is the backstop that turns a miss into a clear install-time error.
_AS_EXTRAS = [
    "beartype>=0.18.5,<0.19.0",
    "ml_collections",
    "onnx-weekly",
    "onnx2torch-py313>=1.6",
    "onnxruntime>=1.17",
    "pydub>=0.25",
    # Floor, not a bare >=2: old 2.x releases carry HIGH-severity advisories (credential
    # leak on cross-host redirect, verify=False session reuse).
    "requests>=2.32.4",
    "resampy>=0.4",
    "rotary-embedding-torch>=0.6.1,<0.7.0",
    "samplerate==0.1.0",
    "six>=1.16",
    'audioop-lts>=0.2.1; python_version>="3.13"',
]

# diffq, installed BINARY-ONLY so pip can never start a compile. Both distributions
# install the same `diffq` module, so either satisfies the import:
#
#   diffq-fixed  wheels: win_amd64 cp310-313, manylinux cp310-312
#   diffq        wheels: macOS/manylinux/win, but only up to cp310
#
# Try the fork first (it covers modern Pythons), then the original (it covers macOS,
# where the fork ships nothing). If NEITHER has a wheel for this interpreter — macOS
# on 3.11+, Linux on 3.13 — we skip it rather than compile. That is safe for what this
# plugin actually does: diffq is only needed to load QUANTIZED demucs checkpoints.
# `demucs` guards its import (`_check_diffq`) and audio-separator only imports it from
# its bundled demucs ARCHITECTURE — neither is on the bs_roformer_sw path, which is the
# model we split with.
_DIFFQ_CANDIDATES = ("diffq-fixed>=0.2", "diffq>=0.2")

# ── GPU / CUDA ───────────────────────────────────────────────────────────────
# On Windows (and by default everywhere), `pip install torch` from PyPI gives the
# CPU-ONLY wheel: torch.version.cuda is None, so the server reports Device: cpu even
# on a machine with a perfectly good NVIDIA card. CUDA builds live on PyTorch's own
# index and carry a local version tag (e.g. 2.8.0+cu128).
#
# whisperx pins torch ~=2.8, so the CUDA build has to be a 2.8 one, and it must go
# through the SAME single pip resolve as everything else - otherwise we reintroduce
# the split-transaction bug that produced the numpy/numba conflict.
#
# No CUDA Toolkit install is needed: the wheels bundle the CUDA runtime. Only a
# recent-enough NVIDIA driver is required.
TORCH_VERSION = os.environ.get("STEM_SPLITTER_TORCH_VERSION", "2.8.0")
DEFAULT_CUDA_TAG = os.environ.get("STEM_SPLITTER_CUDA_TAG", "cu128")
CUDA_TAG = DEFAULT_CUDA_TAG   # back-compat alias
# Builds PyTorch actually publishes. cu128 needs a recent driver; cu126/cu121 are the
# fallbacks for older ones. Offered in the Advanced section because "which CUDA build
# works" is driver-dependent and we can't reliably guess it.
CUDA_TAGS = ["cu128", "cu126", "cu121"]


def cuda_index(tag: str) -> str:
    return f"https://download.pytorch.org/whl/{tag}"

_gpu_memo: dict = {"t": 0.0, "val": None}
_GPU_TTL = 60.0


def detect_nvidia_gpu() -> dict | None:
    """{name, driver} if an NVIDIA GPU is usable here, else None.

    Uses nvidia-smi (present whenever the driver is), so it works without torch and
    before anything is installed - which is what lets the installer offer a GPU build
    up front.
    """
    now = time.monotonic()
    if _gpu_memo["val"] is not None and now - _gpu_memo["t"] < _GPU_TTL:
        return _gpu_memo["val"] or None

    gpu = None
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if p.returncode == 0 and p.stdout.strip():
            first = p.stdout.strip().splitlines()[0]
            parts = [x.strip() for x in first.split(",")]
            if parts and parts[0]:
                gpu = {"name": parts[0],
                       "driver": parts[1] if len(parts) > 1 else ""}
    except Exception:
        gpu = None   # no driver / not NVIDIA / nvidia-smi absent

    _gpu_memo["t"] = now
    _gpu_memo["val"] = gpu or {}
    return gpu


def install_info(config_dir: Path) -> dict:
    """What the last install actually produced (notably: GPU or CPU torch)."""
    try:
        data = json.loads((server_dir(config_dir) / "install.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

# Track a server we started in THIS process, so we can stream its output and reap it.
_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()

# The progress callback of the op currently in flight (install / start /
# prepare_models), or None when nothing is running.
#
# The server's stdout reader thread outlives start_server() — it streams for the
# whole life of the process. If it kept using the callback captured at spawn, then
# every log line AFTER the op finished (and the exit line, whenever the server is
# eventually stopped) would push another "server" progress event, flipping the UI
# back to active=True and re-disabling the controls long after the op was done.
#
# So the reader emits through whatever callback is active *now*. run_server_op()
# clears this when the op completes, which both stops those zombie events and still
# lets a long op like prepare_models keep streaming the server's warmup lines.
_stream_cb: ProgressCB = None
_stream_lock = threading.Lock()

# Short-lived memo for is_running()'s /health probe: it's on the /config path, so a
# burst of UI calls shouldn't each pay a network round-trip.
_running_memo: dict[tuple[str, int], tuple[float, bool]] = {}
_running_lock = threading.Lock()
_RUNNING_TTL = 3.0  # seconds


def set_stream_cb(cb: ProgressCB) -> None:
    global _stream_cb
    with _stream_lock:
        _stream_cb = cb


def clear_stream_cb() -> None:
    set_stream_cb(None)


def _current_stream_cb() -> ProgressCB:
    with _stream_lock:
        return _stream_cb


def _invalidate_running() -> None:
    """Drop the is_running() memo so a start/stop is reflected immediately."""
    with _running_lock:
        _running_memo.clear()


# Sizing the server dir means walking pylibs/ + cache/ - several GB once the weights
# land. server_status() is polled every 5s by the settings page while models download,
# so recomputing it each time would be a continuous disk scan for a number that barely
# moves. Memoize it (and let the caller skip it entirely).
_disk_memo: dict[str, tuple[float, int]] = {}
_disk_lock = threading.Lock()
_DISK_TTL = 30.0  # seconds


def _dir_size(p: Path) -> int:
    """Own it rather than importing engine_install's private helper - this module
    fully controls how it sizes its own tree, and shouldn't break if that one is
    refactored."""
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue   # vanished mid-walk (an install/uninstall running alongside)
    return total


def _server_disk_bytes(config_dir: Path) -> int:
    key = str(config_dir)
    now = time.monotonic()
    with _disk_lock:
        hit = _disk_memo.get(key)
        if hit and now - hit[0] < _DISK_TTL:
            return hit[1]
    size = _dir_size(server_dir(config_dir))
    with _disk_lock:
        _disk_memo[key] = (now, size)
    return size


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


# pip availability can't change for a running interpreter, but can_manage() is on the
# server_status path - which the settings page polls every 5s while models download.
# Without this we'd spawn a subprocess on every poll.
_pip_ok_memo: dict[str, bool] = {}


def _python_has_pip(python_exe: str) -> bool:
    hit = _pip_ok_memo.get(python_exe)
    if hit is not None:
        return hit
    try:
        p = subprocess.run([python_exe, "-m", "pip", "--version"],
                           capture_output=True, text=True, timeout=60)
        ok = p.returncode == 0
    except Exception:
        ok = False
    _pip_ok_memo[python_exe] = ok
    return ok


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


def download_source(config_dir: Path, ref: str | None = None,
                    progress_cb: ProgressCB = None) -> None:
    """Fetch the server source from GitHub (no `git` required) and extract the
    handful of files it needs.

    Pins to a resolved commit so the install is reproducible and reportable; falls
    back to the branch archive only if the SHA can't be resolved.
    """
    import requests

    sdir = src_dir(config_dir)
    sdir.mkdir(parents=True, exist_ok=True)

    ref = (ref or DEFAULT_SOURCE_REF).strip() or DEFAULT_SOURCE_REF
    commit = _resolve_commit(ref)
    archive = commit or ref
    url = f"https://codeload.github.com/{SOURCE_REPO}/zip/{archive}"
    _emit(progress_cb,
          f"Downloading server source {ref}"
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
            json.dumps({"repo": SOURCE_REPO, "ref": ref, "commit": commit,
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
import signal
import subprocess
import sys
import threading
import time

PYLIBS = {pylibs!r}
HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PYLIBS)     # server's deps win over the app's site-packages
sys.path.insert(0, HERE)       # so `import run_demucs` etc. resolve
os.chdir(HERE)


# ── Die with the app that started us ────────────────────────────────────────
#
# We are spawned DETACHED (own process group / session) so a crash or a Ctrl-C in the app
# cannot take the server down mid-separation. The cost of that is real: nothing then stops
# us when the app exits normally either, and we outlive it — holding the port, ~1 GB of RAM
# once warm, and the GPU. Observed in the wild: a server still listening 36 hours after the
# app was closed (got-feedBack/feedBack-plugin-stem-splitter#12).
#
# An in-process shutdown hook in the APP cannot be relied on: it is killed hard on Windows,
# and a crash runs no handlers at all. So the responsibility is inverted — the server
# watches its parent and exits when the parent goes away. That works for a graceful quit, a
# crash, a task-kill, and a power-user's `taskkill /F` alike.
GRACE = 5.0     # seconds a worker gets on SIGTERM before it is killed outright


def _die():
    """Take the whole process tree with us: a separation in flight has grandchildren
    (run_demucs.py / run_roformer.py), and orphaning THOSE just moves the leak."""
    try:
        if os.name == "nt":
            # /T /F is already tree-wide and forceful; Windows has no SIGTERM to be polite
            # with first.
            subprocess.run(["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                           capture_output=True, timeout=15)
        else:
            # We were started with start_new_session, so this group is OURS alone —
            # signalling it cannot touch the app.
            #
            # Two things this has to get right:
            #
            # 1. Ignore SIGTERM in OURSELVES first. killpg signals every member of the group,
            #    which includes this process — with the default handler we would die on our
            #    own signal and never escalate, which is the entire reason we are here.
            #
            # 2. Escalate. A lone SIGTERM followed by an immediate exit is not a kill, it is a
            #    request: a worker deep inside a torch inference can be slow to act on it, or
            #    ignore it. If we exit at that moment there is nobody left to insist, and the
            #    worker is orphaned — the exact leak this fix exists to close, moved down one
            #    generation.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            pgid = os.getpgid(0)
            try:
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
                pass
            # A fixed grace, deliberately: we cannot poll "is the group empty yet?" because we
            # are IN the group, so it never is. The app is already gone and nothing is waiting
            # on us, so a few seconds of patience costs nobody anything and lets a worker close
            # its files.
            time.sleep(GRACE)
            try:
                os.killpg(pgid, signal.SIGKILL)     # takes us with it, which is the point
            except Exception:
                pass
    except Exception:
        pass
    os._exit(1)


def _watch_parent(ppid):
    if os.name == "nt":
        import ctypes
        SYNCHRONIZE = 0x00100000
        kernel32 = ctypes.windll.kernel32
        h = kernel32.OpenProcess(SYNCHRONIZE, False, ppid)
        if not h:
            _die()                       # parent already gone before we got here
        try:
            kernel32.WaitForSingleObject(h, 0xFFFFFFFF)
        finally:
            # The wait is what this thread exists for, so the handle is only held for the
            # server's lifetime — but "we're about to exit anyway" is exactly the reasoning
            # that leaves handles lying around in code that later gets reused.
            kernel32.CloseHandle(h)
    else:
        # Reparented to init (or to a subreaper) once the parent dies.
        while os.getppid() == ppid:
            time.sleep(2)
    _die()


_ppid = os.environ.get("STEM_SPLITTER_PARENT_PID")
try:
    _ppid = int(_ppid) if _ppid else os.getppid()
except (TypeError, ValueError):
    _ppid = os.getppid()

if _ppid and _ppid > 1:
    threading.Thread(target=_watch_parent, args=(_ppid,), daemon=True).start()


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


# The server does not separate in-process: it shells out to run_demucs.py /
# run_roformer.py. Those GRANDCHILD processes inherit neither _launch.py's sys.path
# injection nor (on packaged Windows, isolated ._pth mode) PYTHONPATH - so without
# this they can't see pylibs/ and die with e.g.
# "ModuleNotFoundError: No module named 'audio_separator'", which surfaces as
# "[warmup] bs_roformer_sw: failed: exit 1".
#
# Patch the same bootstrap into the top of each driver script. Works everywhere,
# regardless of how the child is spawned or which env vars survive.
_DRIVER_SCRIPTS = ("run_demucs.py", "run_roformer.py")
_BOOTSTRAP_MARKER = "# --- stem_splitter path bootstrap ---"
_BOOTSTRAP_TEMPLATE = (
    _BOOTSTRAP_MARKER + "\n"
    "# Injected by the feedBack stem_splitter plugin: the server spawns this script\n"
    "# as a subprocess, which inherits neither the launcher's sys.path nor PYTHONPATH\n"
    "# (ignored by the packaged Windows embeddable Python), so point it at the\n"
    "# plugin-managed dependency tree explicitly.\n"
    "import sys as _ss_sys\n"
    "if {pylibs!r} not in _ss_sys.path:\n"
    "    _ss_sys.path.insert(0, {pylibs!r})\n"
    "# --- end stem_splitter path bootstrap ---\n"
)


def _bootstrap_insert_line(text: str) -> int:
    """0-based line index at which the sys.path bootstrap may be inserted.

    Must land AFTER three things that are only legal at the top of a module:

      * a shebang / encoding cookie (honoured on physical lines 1-2 only);
      * the module docstring — both drivers open with one, and pushing it down turns
        it into a plain expression, silently dropping ``__doc__``;
      * any ``from __future__ import ...`` — these MUST precede every other statement,
        so injecting above one is a hard SyntaxError. Neither driver has one today,
        but a future server revision adding one would break every split with an error
        pointing at a file the user never wrote.

    Parse the module and insert before the first statement that is none of the above.
    Falls back to the shebang-only scan if the source doesn't parse (it's not ours to
    validate — a broken driver should fail on its own terms, not here).
    """
    import ast

    lines = text.splitlines(keepends=True)
    head = 0
    while head < len(lines) and head < 2 and (
            lines[head].startswith("#!") or "coding" in lines[head][:40]):
        head += 1

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return head

    for node in tree.body:
        is_docstring = (isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str))
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if is_docstring or is_future:
            # Skip past it: end_lineno is 1-based and inclusive, so it doubles as the
            # 0-based index of the line after.
            head = max(head, node.end_lineno or head)
            continue
        break

    return min(head, len(lines))


def patch_driver_scripts(config_dir: Path) -> list[str]:
    """Prepend the pylibs sys.path bootstrap to the server's subprocess drivers.
    Idempotent. Returns the scripts it patched."""
    pylibs = str(pylibs_dir(config_dir))
    boot = _BOOTSTRAP_TEMPLATE.format(pylibs=pylibs)
    patched: list[str] = []

    for name in _DRIVER_SCRIPTS:
        f = src_dir(config_dir) / name
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8")
        if _BOOTSTRAP_MARKER in text:
            continue  # already bootstrapped
        lines = text.splitlines(keepends=True)
        head = _bootstrap_insert_line(text)
        f.write_text("".join(lines[:head]) + boot + "".join(lines[head:]),
                     encoding="utf-8")
        patched.append(name)

    return patched


def _requirements_without_audio_separator(config_dir: Path) -> Path:
    """The server's requirements.txt with the ``audio-separator`` line removed.

    It is installed separately with ``--no-deps`` (see _AUDIO_SEPARATOR). Leaving it in
    the ``-r`` file would defeat that entirely: pip resolves the ``-r`` file FIRST, so
    diffq gets dragged in and the source build fails before we ever reach the --no-deps
    step. (That exact ordering is why the upstream Dockerfile's --no-deps lines are dead
    code and its image has never built.)

    Written next to the source rather than mutating requirements.txt in place, so
    re-downloading the source doesn't have to care.
    """
    src = src_dir(config_dir) / "requirements.txt"
    out = src_dir(config_dir) / "requirements.no-audio-separator.txt"
    try:
        lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as e:
        raise RuntimeError(f"the server's requirements.txt is unreadable: {e}") from e

    kept = [ln for ln in lines if not _AS_REQ_RE.match(ln)]
    if len(kept) == len(lines):
        # Upstream may fix this on their side (that's the proper home for it). Not an
        # error - our separate --no-deps install still runs.
        log.info("stem_splitter: requirements.txt no longer pins audio-separator")
    out.write_text("".join(kept), encoding="utf-8")
    return out


# pip's signature for "this package has no wheel for your interpreter/platform". It
# prints BOTH of these lines; matching either is enough, and neither appears for a
# network, index, permission or resolution-conflict failure.
_NO_WHEEL_SIGNS = ("no matching distribution found",
                   "could not find a version that satisfies")


def _is_no_wheel_error(e: Exception) -> bool:
    """Did pip fail specifically because no compatible WHEEL exists?

    The distinction is load-bearing: "no wheel" is expected on some interpreters and we
    carry on without diffq. Anything else (a network blip, a dead index, a bad proxy) is
    a real failure, and treating it as "no wheel" would silently degrade the install —
    the user would only find out much later, via a ModuleNotFoundError from a subprocess.
    """
    text = (getattr(e, "pip_output", "") or "") + "\n" + str(e)
    return any(s in text.lower() for s in _NO_WHEEL_SIGNS)


def _install_diffq(target: Path, progress_cb: ProgressCB = None,
                   base: float = 0.9, span: float = 0.03) -> None:
    """Best-effort, BINARY-ONLY diffq install (see _DIFFQ_CANDIDATES).

    ``--only-binary=:all:`` is the whole point: it makes pip *fail* rather than fall
    back to the sdist, so this can never start a C compile on a user's machine. Both
    candidates are tried, and if neither has a wheel for this interpreter we log it and
    move on — quantized demucs checkpoints become unavailable, bs_roformer_sw (what we
    actually split with) does not care.
    """
    for spec in _DIFFQ_CANDIDATES:
        try:
            engine_install.stream_pip(
                sys.executable,
                ["--target", str(target), "--no-deps", "--only-binary=:all:", spec],
                f"{spec.split('>')[0]} (optional)", progress_cb,
                base=base, span=span, pkg_count=1,
            )
            return
        except RuntimeError as e:
            if not _is_no_wheel_error(e):
                # Only "no wheel exists" is tolerable here. Swallowing everything would
                # turn a transient network/index failure into a silently degraded install
                # that only shows up much later as a mystery ModuleNotFoundError.
                raise
            log.info("stem_splitter: no %s wheel for this interpreter", spec)

    _emit(progress_cb,
          "No diffq wheel exists for this Python — skipping it rather than compiling "
          "from source. Quantized demucs checkpoints won't load; bs_roformer_sw (the "
          "model this plugin splits with) is unaffected.",
          base + span, "Preparing")
    log.warning("stem_splitter: diffq unavailable as a wheel; skipped")


def install_server(config_dir: Path, gpu: bool = False, ref: str | None = None,
                   cuda_tag: str | None = None,
                   progress_cb: ProgressCB = None) -> dict:
    """Download the source and install the server's dependencies (pip --target).

    Explicit-only (the "Install server" button). Several GB of wheels.
    """
    ok, reason = can_manage(config_dir)
    if not ok:
        raise RuntimeError(reason)

    server_dir(config_dir).mkdir(parents=True, exist_ok=True)
    cache_dir(config_dir).mkdir(parents=True, exist_ok=True)
    target = pylibs_dir(config_dir)

    download_source(config_dir, ref=ref, progress_cb=progress_cb)

    # Start from a clean tree. `pip --target` does not treat the target as a real
    # environment: it can't see what's already there when resolving, so re-installing
    # over a previous (possibly inconsistent) tree just layers conflicts. Wheels come
    # from pip's HTTP cache, so this is cheap to redo.
    if target.exists():
        _emit(progress_cb, "Clearing previous dependency tree…", 0.09, "Preparing")
        shutil.rmtree(target, ignore_errors=True)
        # ignore_errors hides the failure, not the consequence: a locked .pyd (a server
        # still running, AV, an indexer) leaves a PARTIAL tree, and pip --target can't
        # see what's already there when it resolves — so the install "succeeds" onto a
        # half-deleted tree and produces a subtly broken environment that fails much
        # later, somewhere unrelated. Fail here, where the cause is still legible.
        leftovers = list(target.iterdir()) if target.is_dir() else []
        if leftovers:
            raise RuntimeError(
                f"Could not clear the existing dependency tree at {target} — "
                f"{len(leftovers)} item(s) remain (e.g. {leftovers[0].name}). Something "
                "is holding files open. Stop the server (and close any app that loaded "
                "it), then try again; if it persists, delete that folder by hand."
            )
    target.mkdir(parents=True, exist_ok=True)

    req = _requirements_without_audio_separator(config_dir)
    tgt = ["--target", str(target)]

    # GPU: pin the CUDA torch build INTO the same resolve rather than installing it
    # in a separate pass. A second transaction can't see what the first one pinned
    # (pip --target doesn't treat the target as an environment), which is exactly how
    # the numpy/numba conflict got in. The +cuXXX local tag only exists on PyTorch's
    # index, so pinning it forces the CUDA wheel instead of PyPI's CPU default.
    tag = (cuda_tag or DEFAULT_CUDA_TAG).strip() or DEFAULT_CUDA_TAG
    gpu_args: list[str] = []
    if gpu:
        gpu_args = [
            f"torch=={TORCH_VERSION}+{tag}",
            f"torchaudio=={TORCH_VERSION}+{tag}",
            "--extra-index-url", cuda_index(tag),
        ]
        _emit(progress_cb,
              f"GPU build requested: torch {TORCH_VERSION}+{tag} "
              f"(~2.5 GB; no CUDA Toolkit needed, the wheel bundles the runtime)",
              0.10, "Preparing")

    # ONE resolve for everything that has dependencies. Splitting these across
    # separate pip transactions is what broke the install: each transaction resolves
    # on its own, so a later step upgraded numpy to 2.5 while numba (pulled in via
    # torchcrepe -> resampy in an earlier step) requires numpy <= 2.4 - producing a
    # tree that only failed at server start with "Numba needs NumPy 2.4 or less".
    # Resolving together lets pip honour every constraint at once.
    #
    # demucs and audio-separator are the exceptions and MUST stay --no-deps (see the
    # notes on _DEMUCS_EXTRAS / _AUDIO_SEPARATOR above — between them they'd drag in a
    # torchaudio<2.1 pin and a diffq source build). Their real deps ride in the main
    # resolve instead, so they add no transitive deps here and can't perturb it.
    steps: list[tuple[str, list[str], int]] = [
        ("server dependencies" + (" (GPU/CUDA)" if gpu else ""),
         tgt + ["-r", str(req), *_AS_EXTRAS, *_DEMUCS_EXTRAS, *gpu_args],
         9 + len(_AS_EXTRAS) + len(_DEMUCS_EXTRAS)),
        ("audio-separator (no-deps)", tgt + ["--no-deps", _AUDIO_SEPARATOR], 1),
        ("demucs (no-deps)", tgt + ["--no-deps", "demucs"], 1),
    ]
    n = len(steps) + 2  # + diffq + the verify step
    for i, (label, args, count) in enumerate(steps):
        engine_install.stream_pip(sys.executable, args, label, progress_cb,
                                  base=0.12 + (i / n) * 0.86, span=0.86 / n, pkg_count=count)

    _install_diffq(target, progress_cb,
                   base=0.12 + (len(steps) / n) * 0.86, span=0.86 / n)

    with _disk_lock:
        _disk_memo.clear()   # the tree just changed materially
    try:
        (server_dir(config_dir) / "install.json").write_text(json.dumps({
            "gpu": bool(gpu),
            "torch": f"{TORCH_VERSION}+{tag}" if gpu else "cpu",
            "cuda_tag": tag if gpu else None,
            "installed_at": time.time(),
        }, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("stem_splitter: could not record install info: %s", e)

    write_launcher(config_dir)
    patched = patch_driver_scripts(config_dir)
    if patched:
        _emit(progress_cb, f"Bootstrapped {', '.join(patched)} to see the dependency tree.",
              0.95, "Preparing")
    verify_install(config_dir, gpu=gpu, cuda_tag=tag, progress_cb=progress_cb)

    _emit(progress_cb, "Server installed.", 1.0, "Done")
    return server_status(config_dir)


# Modules the server imports at start-up. Importing them here turns a broken
# dependency resolve into a clear install-time error instead of a traceback the
# first time the user hits Start.
_VERIFY_IMPORTS = ("fastapi", "uvicorn", "torch", "soundfile",
                   "torchcrepe", "demucs", "audio_separator")


def verify_install(config_dir: Path, gpu: bool = False, cuda_tag: str | None = None,
                   progress_cb: ProgressCB = None) -> None:
    """Import-check the freshly installed tree.

    Catches incompatible-pin breakage (e.g. numba vs numpy) at install time, with
    the offending import named, rather than letting the server die on first start.
    """
    _emit(progress_cb, "Verifying the installed dependencies…", 0.96, "Verifying")

    # The drivers the server shells out to must carry the sys.path bootstrap, or
    # they'll die with ModuleNotFoundError the first time a split runs (which shows
    # up only as "[warmup] bs_roformer_sw: failed: exit 1").
    unpatched = [n for n in _DRIVER_SCRIPTS
                 if (src_dir(config_dir) / n).is_file()
                 and _BOOTSTRAP_MARKER not in (src_dir(config_dir) / n).read_text(encoding="utf-8")]
    if unpatched:
        raise RuntimeError(
            f"the server's subprocess drivers ({', '.join(unpatched)}) are missing the "
            "dependency-path bootstrap - they would fail to import their packages."
        )
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(pylibs_dir(config_dir))!r})\n"
        f"mods = {list(_VERIFY_IMPORTS)!r}\n"
        "bad = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        __import__(m)\n"
        "    except Exception as e:\n"
        "        bad.append('%s: %s' % (m, e))\n"
        "print('VERIFY_FAILED:' + ' | '.join(bad) if bad else 'VERIFY_OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=_server_env(config_dir), timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    if "VERIFY_OK" in out:
        if gpu:
            _verify_cuda(config_dir, cuda_tag=cuda_tag, progress_cb=progress_cb)
        _emit(progress_cb, "Dependencies verified.", 0.99, "Verifying")
        return

    detail = ""
    for line in out.splitlines():
        if line.startswith("VERIFY_FAILED:"):
            detail = line.split(":", 1)[1].strip()
            break
    if not detail:
        detail = out.strip()[-400:]
    raise RuntimeError(
        "the server's dependencies installed but don't import cleanly - "
        f"{detail}. Try 'Uninstall server' and install again; if it persists this is "
        "a dependency-pin conflict worth reporting."
    )


def _scaled(progress_cb: ProgressCB, base: float, span: float) -> ProgressCB:
    """Re-map a sub-step's 0..1 progress into a slice of the overall bar."""
    if progress_cb is None:
        return None

    def cb(ev: dict) -> None:
        pct = base + max(0.0, min(1.0, ev.get("pct", 0.0))) * span
        progress_cb({**ev, "pct": max(0.0, min(1.0, pct))})
    return cb


def setup_server(config_dir: Path, port: int = DEFAULT_PORT, device: str = "",
                 model: str = DEFAULT_MODEL, gpu: bool = False,
                 ref: str | None = None, cuda_tag: str | None = None,
                 progress_cb: ProgressCB = None) -> dict:
    """One-click setup: install the server AND download its models.

    This is what the "Install server" button runs. Installing without the weights
    leaves you with a server that can't actually split until a second, separate
    action - so do the whole job in one explicit click, under one progress stream.
    Still explicit: nothing here runs unless the user asks for it.
    """
    _emit(progress_cb, "Setting up the demucs server (dependencies, then models)…",
          0.0, "Setting up")

    # Dependencies take the bulk of the wall-clock, models the rest.
    install_server(config_dir, gpu=gpu, ref=ref, cuda_tag=cuda_tag,
                   progress_cb=_scaled(progress_cb, 0.0, 0.55))
    prepare_models(config_dir, port=port, device=device, model=model,
                   progress_cb=_scaled(progress_cb, 0.55, 0.45))

    _emit(progress_cb, "Server installed, running and warmed up.", 1.0, "Done")
    return server_status(config_dir)


def uninstall_server(config_dir: Path) -> dict:
    """Stop the server and delete its source, dependency tree and downloaded weights."""
    try:
        stop_server(config_dir)
    except Exception as e:
        log.warning("stem_splitter: stop before uninstall failed: %s", e)
    shutil.rmtree(server_dir(config_dir), ignore_errors=True)
    with _disk_lock:
        _disk_memo.clear()
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
    # Point any process the server spawns at the dependency tree. The embeddable
    # Python ignores this (isolated ._pth mode) - patch_driver_scripts() is what
    # covers that case - but it's correct everywhere else and costs nothing.
    env["PYTHONPATH"] = str(pylibs_dir(config_dir))

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


def _as_port(value, default: int | None = None) -> int | None:
    """Coerce a persisted port to a usable int, else `default`.

    The state file is JSON on disk: a truncated write or a hand-edit can leave
    ``"port": "abc"``. A bare ``int()`` on that raises straight out of
    ``server_status()`` / ``is_running()`` — i.e. out of /config, engine resolution
    and every lifecycle route — 500ing the settings UI with no way for the user to
    recover short of deleting the file by hand.
    """
    try:
        p = int(value)
    except (TypeError, ValueError):
        return default
    return p if 1 <= p <= 65535 else default


def is_running(config_dir: Path, port: int | None = None) -> tuple[bool, int | None]:
    """(running, port). True if we own a live child, or if a previously-recorded
    port still answers /health (an orphan from a crashed app).

    Called from /config and engine resolution, so it must be CHEAP in the common
    case. Two guards:
      * If we never started a server (no state file), don't probe at all — a
        blocking 1.5s /health on every UI call, on a machine that never installed
        the server, is pure latency.
      * Otherwise memoize the probe result briefly.
    """
    with _proc_lock:
        p = _proc
    if p is not None and p.poll() is None:
        st = _read_state(config_dir)
        return True, _as_port(st.get("port"), port)

    st = _read_state(config_dir)
    # NOT `port or ...`: only probe a port we actually started. A garbage value in the
    # state file means we have nothing trustworthy to probe -> not running.
    known = _as_port(st.get("port"))
    if not known:
        return False, None

    key = (str(config_dir), known)
    now = time.monotonic()
    with _running_lock:
        hit = _running_memo.get(key)
        if hit and now - hit[0] < _RUNNING_TTL:
            return (True, int(known)) if hit[1] else (False, None)

    ok, _ = server_health(url_for(int(known)), timeout=1.5)
    with _running_lock:
        _running_memo[key] = (now, ok)
    return (True, int(known)) if ok else (False, None)


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

    # Rewrite the launcher + re-bootstrap the driver scripts on every start: they
    # bake in the pylibs path, the config dir can move, and this repairs an install
    # made before the bootstrap existed without forcing a reinstall.
    write_launcher(config_dir)
    patch_driver_scripts(config_dir)

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

    # The launcher watches THIS pid and exits when it dies (see _LAUNCHER_TEMPLATE). Pass it
    # explicitly rather than letting the child call os.getppid(): if the interpreter is ever
    # reached through a shim, the child's parent is the shim, and the server would sit
    # watching a process that exits immediately - or worse, one that never does.
    env = _server_env(config_dir)
    env["STEM_SPLITTER_PARENT_PID"] = str(os.getpid())

    proc = subprocess.Popen(
        cmd, cwd=str(src_dir(config_dir)), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, **popen_kwargs,
    )
    with _proc_lock:
        _proc = proc
    _write_state(config_dir, {"pid": proc.pid, "port": port, "started_at": time.time()})
    _invalidate_running()

    tail: list[str] = []
    set_stream_cb(progress_cb)

    def _reader() -> None:
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                tail.append(line)
                if len(tail) > 60:
                    tail.pop(0)
                # Emit through the CURRENTLY active op's callback, not the one
                # captured at spawn — see set_stream_cb().
                cb = _current_stream_cb()
                if cb is None:
                    log.debug("stem_splitter[server]: %s", line)
                    continue
                low = line.lower()
                phase = "Running"
                pct = 0.6
                if "[warmup]" in low:
                    phase = "Warming up"
                    if "ready" in low:
                        pct = 0.9
                _emit(cb, line, pct, phase)
        except Exception:
            pass
        finally:
            rc = proc.poll()
            cb = _current_stream_cb()
            if cb is None:
                log.info("stem_splitter: server process exited (code %s)", rc)
            else:
                _emit(cb, f"Server process exited (code {rc}).", 0.0, "Stopped")

    threading.Thread(target=_reader, name="stem_splitter-server-log", daemon=True).start()

    # Wait briefly for the port to bind so the caller gets a truthful status.
    url = url_for(port)
    for _ in range(40):  # ~20s
        if proc.poll() is not None:
            time.sleep(0.2)  # let the reader drain the last lines
            raise RuntimeError(_startup_failure_message(proc.returncode, tail))
        ok, _payload = server_health(url, timeout=1.0)
        if ok:
            _emit(progress_cb, f"Server is up at {url}", 1.0, "Running")
            return server_status(config_dir)
        time.sleep(0.5)

    raise RuntimeError(f"server did not answer /health on {url} within 20s")


def _verify_cuda(config_dir: Path, cuda_tag: str | None = None,
                 progress_cb: ProgressCB = None) -> None:
    """A GPU install that silently landed a CPU wheel is the whole bug we're fixing -
    so prove torch actually sees CUDA rather than trusting the pin."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(pylibs_dir(config_dir))!r})\n"
        "import torch\n"
        "print('CUDA_BUILD:%s|%s' % (torch.version.cuda, torch.cuda.is_available()))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=_server_env(config_dir), timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = next((l for l in out.splitlines() if l.startswith("CUDA_BUILD:")), "")
    built, avail = (line.split(":", 1)[1].split("|") + ["", ""])[:2] if line else ("", "")

    if built in ("", "None"):
        raise RuntimeError(
            "a GPU build was requested but the installed torch has no CUDA support "
            f"(torch.version.cuda is None). The CPU wheel was resolved instead. Try a "
            f"different CUDA build in Advanced (tried: {cuda_tag or DEFAULT_CUDA_TAG}; "
            f"options: {', '.join(CUDA_TAGS)})."
        )
    if avail.strip() != "True":
        _emit(progress_cb,
              f"torch has CUDA {built} but no GPU is visible right now - the server "
              "will fall back to CPU. Check the NVIDIA driver.", 0.98, "Verifying")
        return
    _emit(progress_cb, f"GPU ready: torch CUDA {built} available.", 0.98, "Verifying")


def _startup_failure_message(rc: int | None, tail: list[str]) -> str:
    """Turn an immediate server exit into something actionable.

    The common case by far is a stale or inconsistent dependency tree - e.g. a
    pylibs/ built by an older installer. Starting the server does NOT reinstall
    anything, so the user can restart forever and see the same traceback; say
    plainly that it needs a reinstall.
    """
    blob = "\n".join(tail)
    low = blob.lower()
    msg = f"the server exited immediately (code {rc})"

    if "importerror" in low or "modulenotfounderror" in low:
        offender = ""
        for line in reversed(tail):
            if line.strip().startswith(("ImportError:", "ModuleNotFoundError:")):
                offender = line.strip()
                break
        return (
            f"{msg} because its dependencies don't import: {offender or 'ImportError'}. "
            "This means the installed dependency tree is broken or was built by an older "
            "installer - starting the server does NOT reinstall it. Click 'Uninstall "
            "server', then 'Install server' to rebuild it cleanly."
        )

    if "address already in use" in low or "winerror 10048" in low:
        return (f"{msg}: that port is already in use. Change the port, or stop whatever "
                "is already listening on it.")

    return f"{msg}. Last output:\n" + "\n".join(tail[-12:])


def _pid_cmdline(pid: int) -> str | None:
    """The process's command line, or None if we can't determine it."""
    try:
        if os.name == "nt":
            p = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"],
                capture_output=True, text=True, timeout=20)
            out = (p.stdout or "").strip()
            return out or None
        proc = Path(f"/proc/{int(pid)}/cmdline")
        if proc.exists():
            return proc.read_bytes().replace(b"\\x00", b" ").decode("utf-8", "replace").strip() or None
        p = subprocess.run(["ps", "-p", str(int(pid)), "-o", "args="],
                           capture_output=True, text=True, timeout=20)
        return (p.stdout or "").strip() or None
    except Exception as e:
        log.warning("stem_splitter: could not read cmdline for pid %s: %s", pid, e)
        return None


def _pid_is_our_server(pid: int, config_dir: Path) -> bool:
    """Does this pid actually look like the server WE launched?

    A recorded pid plus a port that answers /health is NOT proof: the state file can be
    stale (server crashed, pid reused) while some other service happens to be on that
    port. Killing on that basis - with taskkill /T, which takes the whole tree - could
    take down an unrelated process. So check the command line really is our launcher
    before signalling anything.
    """
    cmd = _pid_cmdline(pid)
    if not cmd:
        return False   # can't verify -> don't kill
    launcher = str(launcher_path(config_dir))
    return launcher in cmd or ("_launch.py" in cmd and str(server_dir(config_dir)) in cmd)


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
    #
    # Probe the same target we signalled. When killing the group, polling only the
    # parent pid is wrong: uvicorn can exit while its run_demucs.py / run_roformer.py
    # workers are still handling SIGTERM, so the loop would return early and skip the
    # SIGKILL — leaving exactly the orphans this function exists to prevent.
    def _alive() -> bool:
        try:
            if use_group:
                os.killpg(pgid, 0)
            else:
                os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            return True  # e.g. EPERM: it exists but isn't ours to signal

    for _ in range(20):  # ~2s
        if not _alive():
            return
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
        # We own this child and know it's alive - safe to kill by pid.
        pid = p.pid
    else:
        # No live child of ours. The state file's pid may be STALE: the server could
        # have crashed and the OS reused that pid for something else entirely, and
        # killing it (with /T, which takes the whole tree) could take down an
        # unrelated process. Only act on it if the recorded port still answers
        # /health - i.e. something that looks like our server is actually there.
        st = _read_state(config_dir)
        recorded_pid, recorded_port = st.get("pid"), st.get("port")
        if recorded_pid and recorded_port:
            alive, _ = is_running(config_dir)
            if not alive:
                log.info("stem_splitter: recorded server pid %s is not answering on "
                         "port %s - treating it as stale and NOT killing it "
                         "(the pid may have been reused)", recorded_pid, recorded_port)
            elif not _pid_is_our_server(int(recorded_pid), config_dir):
                # Something answers /health on that port, but this pid isn't our
                # launcher - so the state file is stale and the pid belongs to someone
                # else. Killing it (tree-wide) could take down an unrelated process.
                log.warning(
                    "stem_splitter: port %s answers /health but pid %s does not look "
                    "like our server (its command line doesn't reference %s) - "
                    "refusing to kill it. Clearing the stale state file.",
                    recorded_port, recorded_pid, launcher_path(config_dir))
            else:
                pid = recorded_pid

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
    _invalidate_running()

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
    port = _as_port(st.get("port"), DEFAULT_PORT)
    running, live_port = is_running(config_dir, port)
    port = _as_port(live_port, port)
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
        "disk_bytes": _server_disk_bytes(config_dir),
        # False on deployments where a plugin-managed server makes no sense
        # (no pip, or a read-only config dir). The UI disables the section.
        "source": source_meta(config_dir),   # {repo, ref, commit} actually installed
        # GPU picture: what hardware is here vs what the install actually built.
        # A machine with a GPU but a CPU-only torch is the case worth surfacing.
        "gpu_detected": detect_nvidia_gpu(),
        "gpu_build": bool(install_info(config_dir).get("gpu")),
        "install_info": install_info(config_dir),   # {gpu, torch, cuda_tag}
        "defaults": {"ref": DEFAULT_SOURCE_REF, "cuda_tag": DEFAULT_CUDA_TAG,
                     "cuda_tags": CUDA_TAGS, "repo": SOURCE_REPO},
        "manageable": manageable,
        "manage_reason": manage_reason,
        # Non-blocking note (e.g. "you're in a container") shown alongside the controls.
        "advisory": manage_advisory(config_dir),
    }

"""Opt-in local-engine installer for the Stem Splitter plugin.

Nothing here runs on plugin load. It runs ONLY when the user clicks "Download
local engine + models" in the settings, which calls ``install_engine``. The
heavy libraries (torch, demucs / audio-separator, whisperx — multiple GB) are
pip-installed into ``{config_dir}/engine`` (the only writable path), and model
weights land in ``{config_dir}/models``. Both are added to ``sys.path`` / used
as cache dirs at run time by ``split_stems`` / ``transcribe``.

This keeps the guarantee: no model or dependency is ever downloaded unless the
user explicitly asks.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("feedBack.plugin.stem_splitter")

# Progress events: {"line": str, "pct": float 0..1, "phase": str}.
ProgressCB = Optional[Callable[[dict], None]]

# CPU wheels for torch keep the download sane on machines without CUDA. Callers
# who want GPU can install torch themselves into the engine dir.
_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

# Package sets per engine. Kept explicit so the UI can offer them individually.
PKG_SETS: dict[str, list[str]] = {
    "audio-separator": ["torch", "torchaudio", "audio-separator", "onnxruntime"],
    "demucs": ["torch", "torchaudio", "demucs", "soundfile"],
    "whisperx": ["torch", "torchaudio", "whisperx"],
}
PKG_SETS["all"] = sorted({p for s in ("audio-separator", "demucs", "whisperx") for p in PKG_SETS[s]})

# Importable module name per engine (for status probing).
_PROBE_MODULE = {
    "audio-separator": "audio_separator",
    "demucs": "demucs",
    "whisperx": "whisperx",
    "torch": "torch",
}


def engine_dir(config_dir: Path) -> Path:
    return Path(config_dir) / "engine"


def models_dir(config_dir: Path) -> Path:
    return Path(config_dir) / "models"


def _dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _installed_dist(edir: Path, module: str) -> bool:
    """Is a package importable from the engine dir? Cheap check by presence of
    its top-level package/module under the pip --target tree."""
    if not edir.exists():
        return False
    for cand in (edir / module, edir / f"{module}.py"):
        if cand.exists():
            return True
    # dist-info dirs (e.g. audio_separator-x.dist-info) as a fallback signal.
    return any(edir.glob(f"{module.replace('_', '?')}*.dist-info"))


def engine_status(config_dir: Path) -> dict:
    edir = engine_dir(config_dir)
    mdir = models_dir(config_dir)
    installed = {name: _installed_dist(edir, mod) for name, mod in _PROBE_MODULE.items()}
    return {
        "engine_dir": str(edir),
        "models_dir": str(mdir),
        "installed": installed,
        "any_installed": any(installed.values()),
        "engine_bytes": _dir_size(edir),
        "models_bytes": _dir_size(mdir),
    }


def install_engine(config_dir: Path, which: str, progress_cb: ProgressCB = None) -> dict:
    """pip-install the requested engine's packages into ``{config_dir}/engine``.

    Streams pip output line-by-line via ``progress_cb``. Idempotent: pip skips
    already-satisfied requirements. Returns the post-install ``engine_status``.
    """
    if which not in PKG_SETS:
        raise ValueError(f"unknown engine {which!r}; expected one of {sorted(PKG_SETS)}")

    edir = engine_dir(config_dir)
    edir.mkdir(parents=True, exist_ok=True)
    models_dir(config_dir).mkdir(parents=True, exist_ok=True)

    pkgs = PKG_SETS[which]
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(edir),
        "--upgrade",
        "--progress-bar", "off",  # the TTY bar is noise when piped; we derive our own
        "--extra-index-url", _TORCH_INDEX,
        *pkgs,
    ]

    def emit(line: str = "", pct: float = 0.0, phase: str = "") -> None:
        if progress_cb:
            progress_cb({"line": line, "pct": max(0.0, min(1.0, pct)), "phase": phase})

    emit(f"Installing {which}: {', '.join(pkgs)}", 0.02, "Starting")

    # Coarse progress: pip resolves/downloads (Collecting…, ~0–70%), then installs
    # (Installing collected packages…, ~85%), then finishes (Successfully
    # installed, 100%). We can't get true byte-percentages from a piped pip, so we
    # advance the bar off these phase markers + a package counter.
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    total = max(1, len(pkgs))
    collected = 0
    pct = 0.02
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        low = line.lower()
        phase = ""
        if low.startswith("collecting ") or low.startswith("requirement already"):
            collected += 1
            pct = min(0.70, 0.02 + (collected / total) * 0.68)
            phase = "Resolving / downloading"
        elif "downloading " in low:
            phase = "Downloading"
            m = re.search(r"\(([\d.]+\s*[kmg]b)\)", low)
            if m:
                line = line  # size already in the line; UI shows it
        elif low.startswith("installing collected packages"):
            pct = 0.85
            phase = "Installing"
        elif low.startswith("successfully installed"):
            pct = 0.99
            phase = "Finalizing"
        emit(line, pct, phase)

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"pip install failed (exit {rc}) for engine {which!r}")

    emit(f"Done installing {which}.", 1.0, "Done")
    return engine_status(config_dir)


def uninstall_engine(config_dir: Path) -> dict:
    """Delete the installed engine + downloaded models to reclaim disk."""
    for p in (engine_dir(config_dir), models_dir(config_dir)):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    return engine_status(config_dir)

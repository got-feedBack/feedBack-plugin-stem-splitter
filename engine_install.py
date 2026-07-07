"""Opt-in local-engine installer for the Stem Splitter plugin.

Nothing here runs on plugin load. It runs ONLY when the user clicks "Download
local engine + models" in the settings, which calls ``install_engine``. The
heavy libraries (torch, demucs / audio-separator, whisperx - multiple GB) are
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

# Shared PyTorch base, installed ONCE before any engine. All three engines depend
# on torch/torchaudio; installing it per-engine into the same --target tree with
# --upgrade made the versions order-dependent (a later engine could rewrite the
# torch an earlier one needs) and re-downloaded the multi-hundred-MB wheels each
# time. Installing it once, then layering engine-only packages with the default
# only-if-needed upgrade strategy, keeps a single consistent torch for everyone.
_SHARED_PKGS = ["torch", "torchaudio"]

# Engine-specific packages (torch/torchaudio come from _SHARED_PKGS above). Kept
# explicit so the UI can offer each engine individually.
PKG_SETS: dict[str, list[str]] = {
    "audio-separator": ["audio-separator", "onnxruntime"],
    "demucs": ["demucs", "soundfile"],
    "whisperx": ["whisperx"],
}

# "all" installs each engine in ITS OWN pip transaction, in this order, and does
# NOT abort the rest if one fails. demucs is last on purpose: it pulls `diffq`, a
# C-extension with no prebuilt wheels for recent Python, so on a machine without a
# C++ toolchain its wheel build fails - that must not take down audio-separator
# (bs_roformer, the primary engine - pure wheels, no compiler) or whisperx.
_ALL_ORDER = ["audio-separator", "whisperx", "demucs"]

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


def _build_failure_hint(engine: str, output_tail: str) -> str:
    """Turn an opaque install failure into an actionable message.

    Distinguishes true source-build failures (need a C++ toolchain) from
    version/availability failures (no compatible wheel for this Python) so the
    hint points at the right fix.
    """
    low = output_tail.lower()
    build_smell = any(s in low for s in (
        "failed to build", "failed-wheel-build", "diffq",
        "microsoft visual c++", "error: command", "gcc", "clang",
    ))
    version_smell = any(s in low for s in (
        "requires-python", "no matching distribution",
    ))
    # Only surface the diffq-specific guidance when diffq is actually the culprit;
    # other demucs failures fall through to the generic hints below.
    if engine == "demucs" and "diffq" in low:
        return (
            "demucs needs to compile `diffq`, which has no prebuilt wheel for this "
            "Python and requires a C++ build toolchain. On Windows install "
            "\"Microsoft C++ Build Tools\" (Desktop development with C++), then retry "
            "- OR just skip demucs and use audio-separator (bs_roformer_sw), which "
            "needs no compiler."
        )
    if build_smell:
        return (
            f"a dependency of {engine} failed to build from source (no prebuilt wheel "
            "for this Python and no C++ compiler available). Install a C++ build "
            "toolchain, or use a Python version with prebuilt wheels, then retry."
        )
    if version_smell:
        return (
            f"no compatible wheel for {engine} on this Python version. Use a supported "
            "Python (3.10-3.11) and retry - this is a version/availability problem, not "
            "a missing compiler."
        )
    return ""


def _pip_install(config_dir: Path, label: str, pkgs: list[str], progress_cb: ProgressCB,
                 base: float, span: float) -> None:
    """Install one package set in its own pip transaction. Raises ``RuntimeError``
    (with an actionable hint) on failure.

    ``--upgrade-strategy only-if-needed`` (pip's default, made explicit) means a
    layered engine install only touches the shared torch/torchaudio if it actually
    needs a different version - so one engine can't gratuitously rewrite the torch
    another engine already installed.
    """
    edir = engine_dir(config_dir)
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(edir),
        "--upgrade", "--upgrade-strategy", "only-if-needed",
        "--progress-bar", "off",
        "--extra-index-url", _TORCH_INDEX,
        *pkgs,
    ]

    def emit(line: str = "", local: float = 0.0, phase: str = "") -> None:
        if progress_cb:
            pct = base + max(0.0, min(1.0, local)) * span
            progress_cb({"line": line, "pct": max(0.0, min(1.0, pct)),
                         "phase": f"{label}: {phase}" if phase else label})

    emit(f"Installing {label}: {', '.join(pkgs)}", 0.02, "Starting")

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    total = max(1, len(pkgs))
    collected = 0
    local = 0.02
    tail: list[str] = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
        low = line.lower()
        phase = ""
        if low.startswith("collecting ") or low.startswith("requirement already"):
            collected += 1
            local = min(0.70, 0.02 + (collected / total) * 0.68)
            phase = "Resolving / downloading"
        elif "downloading " in low:
            phase = "Downloading"
        elif low.startswith("building wheel") or "building wheels" in low:
            phase = "Building"
        elif low.startswith("installing collected packages"):
            local = 0.85
            phase = "Installing"
        elif low.startswith("successfully installed"):
            local = 0.99
            phase = "Finalizing"
        emit(line, local, phase)

    rc = proc.wait()
    if rc != 0:
        hint = _build_failure_hint(label, "\n".join(tail))
        msg = f"pip install failed (exit {rc}) for {label}"
        if hint:
            msg += " - " + hint
        raise RuntimeError(msg)

    emit(f"Done installing {label}.", 1.0, "Done")


def install_engine(config_dir: Path, which: str, progress_cb: ProgressCB = None) -> dict:
    """pip-install engine package sets into ``{config_dir}/engine``.

    ``which`` is a single engine name or ``"all"``. The shared PyTorch base is
    installed ONCE first, then each engine is installed in its OWN pip transaction
    so one failure (typically demucs' `diffq` wheel build on a machine without a
    C++ compiler) does not abort the others. Streams progress via ``progress_cb``.
    Idempotent. Returns ``engine_status`` augmented with a per-step ``results``
    map. Raises only if EVERY requested engine failed.
    """
    if which != "all" and which not in PKG_SETS:
        raise ValueError(f"unknown engine {which!r}; expected 'all' or one of {sorted(PKG_SETS)}")

    engine_dir(config_dir).mkdir(parents=True, exist_ok=True)
    models_dir(config_dir).mkdir(parents=True, exist_ok=True)

    order = _ALL_ORDER if which == "all" else [which]
    # Shared torch base first (once), then the engine-specific packages. If the
    # base step fails the engine steps still self-heal (pip resolves torch as a
    # dependency), just less efficiently.
    steps: list[tuple[str, list[str]]] = [("pytorch", _SHARED_PKGS)]
    steps += [(eng, PKG_SETS[eng]) for eng in order]
    n = len(steps)
    results: dict[str, str] = {}
    for i, (label, pkgs) in enumerate(steps):
        try:
            _pip_install(config_dir, label, pkgs, progress_cb, base=i / n, span=1 / n)
            results[label] = "ok"
        except Exception as e:  # keep going - a fragile step must not sink the rest
            results[label] = str(e)
            log.warning("stem_splitter: install step %s failed: %s", label, e)
            if progress_cb:
                progress_cb({"line": f"✗ {label} skipped: {e}",
                             "pct": (i + 1) / n, "phase": f"{label}: failed"})

    ok_engines = [e for e in order if results.get(e) == "ok"]
    if not ok_engines:
        prefix = ("all engine installs failed" if which == "all"
                  else f"{which} install failed")
        raise RuntimeError(prefix + ": " +
                           "; ".join(f"{e}: {v}" for e, v in results.items()))
    status = engine_status(config_dir)
    status["results"] = results
    if progress_cb:
        failed = [e for e in order if results.get(e) != "ok"]
        summary = f"Installed: {', '.join(ok_engines)}" + (f" | failed: {', '.join(failed)}" if failed else "")
        progress_cb({"line": summary, "pct": 1.0, "phase": "Done"})
    return status


def uninstall_engine(config_dir: Path) -> dict:
    """Delete the installed engine + downloaded models to reclaim disk."""
    for p in (engine_dir(config_dir), models_dir(config_dir)):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    return engine_status(config_dir)

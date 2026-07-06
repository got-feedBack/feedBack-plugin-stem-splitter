"""Stem-separation orchestration for the Stem Splitter plugin.

Self-contained reimplementation of the concept the retired ``sloppak_convert``
module used to provide — built against feedBack's own pak I/O (``pak_io``) and
its ffmpeg helper (``lib/audio``). Three interchangeable engines behind one
interface:

* **remote** — POST the mix to ``{server_url}/separate`` (default model
  ``bs_roformer_sw``). No local deps.
* **audio-separator** (local, opt-in) — run ``bs_roformer_sw`` locally via the
  ``audio_separator`` package (parity with the remote server).
* **demucs** (local, opt-in) — run a demucs-native model (``htdemucs_6s``).

Local engines and their weights are only present after the user opts into the
in-app download (see ``engine_install``); this module never installs anything —
it imports lazily and raises a clear error if a local engine isn't available.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import pak_io

log = logging.getLogger("feedBack.plugin.stem_splitter")

ProgressCB = Optional[Callable[[float, str], None]]

STEM_SEPARATION_SCHEMA_VERSION = "1.0.0"
DEFAULT_REMOTE_MODEL = "bs_roformer_sw"
DEFAULT_DEMUCS_MODEL = "htdemucs_6s"
# Requested stem set for engines that accept one. bs_roformer variants may
# ignore this and return their own set — we write back whatever comes out.
DEFAULT_STEMS = ("drums", "bass", "vocals", "other", "guitar", "piano")
_STEM_ORDER = ["guitar", "bass", "drums", "vocals", "piano", "other", "full"]

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")


def _sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_").lower()
    return s or "stem"


def _prepend_engine_path(engine_dir: str | None) -> None:
    """Make an opt-in-installed engine importable (pip --target dir)."""
    if engine_dir and engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)


def demucs_available(engine_dir: str | None = None) -> bool:
    _prepend_engine_path(engine_dir)
    try:
        import demucs  # noqa: F401
        return True
    except Exception:
        return False


def audio_separator_available(engine_dir: str | None = None) -> bool:
    _prepend_engine_path(engine_dir)
    try:
        import audio_separator  # noqa: F401
        return True
    except Exception:
        return False


# ── Engines ──────────────────────────────────────────────────────────────────

def _run_remote(mix: Path, out_dir: Path, model: str, server_url: str,
                api_key: str | None, stems: tuple[str, ...],
                progress_cb: ProgressCB) -> Path:
    """POST the mix to ``{server_url}/separate`` and download the stems."""
    import time
    import requests

    server_url = server_url.rstrip("/")
    if progress_cb:
        progress_cb(0.10, f"Uploading to split server ({server_url})")

    content_type = "audio/wav" if mix.suffix.lower() == ".wav" else "audio/ogg"
    params: dict[str, str] = {"model": model}
    if stems:
        params["stems"] = ",".join(stems)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    with open(mix, "rb") as f:
        resp = requests.post(
            f"{server_url}/separate",
            files={"file": (mix.name, f, content_type)},
            params=params, headers=headers, timeout=600,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"split server error ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    stem_urls = data.get("stems") or {}
    if not stem_urls and data.get("job_id"):
        job_id = data["job_id"]
        for _ in range(120):  # up to ~10 min
            time.sleep(5)
            jr = requests.get(f"{server_url}/jobs/{job_id}", timeout=30).json()
            status = jr.get("status")
            if status == "complete":
                stem_urls = jr.get("stems") or {}
                break
            if status == "failed":
                raise RuntimeError(f"split server job failed: {jr.get('error')}")
            if progress_cb:
                progress_cb(0.4, f"Separating on server ({status or 'working'})")
    if not stem_urls:
        raise RuntimeError("split server returned no stems")

    result_dir = out_dir / "remote_stems"
    result_dir.mkdir(parents=True, exist_ok=True)
    for name, url in stem_urls.items():
        if isinstance(url, str) and url.startswith("/"):
            url = f"{server_url}{url}"
        sr = requests.get(url, timeout=180)
        if sr.status_code == 200:
            ext = ".mp3" if ".mp3" in str(url).lower() else ".wav"
            (result_dir / f"{_sanitize(name)}{ext}").write_bytes(sr.content)
    return result_dir


def _run_audio_separator(mix: Path, out_dir: Path, model: str, models_dir: str | None,
                         engine_dir: str | None, progress_cb: ProgressCB) -> Path:
    _prepend_engine_path(engine_dir)
    try:
        from audio_separator.separator import Separator
    except Exception as e:
        raise RuntimeError(
            "local audio-separator engine not installed — use "
            "'Download local engine' in the plugin settings"
        ) from e

    result_dir = out_dir / "as_stems"
    result_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0.15, f"Loading {model} (audio-separator)")

    # bs_roformer_sw ships as a model file the separator resolves by name; the
    # download lands in ``models_dir`` (pre-warmed by the installer).
    sep = Separator(
        output_dir=str(result_dir),
        model_file_dir=str(models_dir) if models_dir else None,
        output_format="WAV",
    )
    sep.load_model(model_filename=_as_model_filename(model))
    if progress_cb:
        progress_cb(0.4, "Separating (local)")
    sep.separate(str(mix))
    return result_dir


def _as_model_filename(model: str) -> str:
    """Map our short model id to an audio-separator model filename."""
    known = {
        "bs_roformer_sw": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    }
    return known.get(model, model if model.endswith((".ckpt", ".onnx", ".pth")) else f"{model}.ckpt")


def _run_demucs_local(mix: Path, out_dir: Path, model: str, engine_dir: str | None,
                      progress_cb: ProgressCB) -> Path:
    _prepend_engine_path(engine_dir)
    if not demucs_available(engine_dir):
        raise RuntimeError(
            "local demucs engine not installed — use 'Download local engine' "
            "in the plugin settings"
        )
    result_dir = out_dir / "demucs_out"
    result_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0.15, f"Running demucs ({model})")
    # Subprocess keeps torch out of the server process and isolates crashes.
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(result_dir), str(mix)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"demucs failed: {(proc.stderr or proc.stdout)[:400]}")
    # demucs writes <out>/<model>/<track>/<stem>.wav
    subdir = result_dir / model
    tracks = [p for p in subdir.glob("*") if p.is_dir()] if subdir.is_dir() else []
    return tracks[0] if tracks else result_dir


# ── Orchestration ────────────────────────────────────────────────────────────

def _encode_ogg(src: Path, out_ogg: Path) -> None:
    """Encode any audio file to Ogg Vorbis using feedBack's ffmpeg helper."""
    if src.suffix.lower() == ".ogg":
        out_ogg.write_bytes(src.read_bytes())
        return
    from audio import _ffmpeg_cmd, _ffmpeg_wav_to_ogg
    ffmpeg = _ffmpeg_cmd()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not available to encode stems")
    proc = _ffmpeg_wav_to_ogg(ffmpeg, src, out_ogg)
    if proc.returncode != 0 or not out_ogg.exists():
        raise RuntimeError(f"ffmpeg ogg encode failed for {src.name}")


def _collect_stem_files(result_dir: Path) -> list[Path]:
    files = [p for p in sorted(result_dir.rglob("*")) if p.suffix.lower() in _AUDIO_EXTS]
    return files


def split_pak(pak_path: Path, *, engine: str, model: str | None = None,
              server_url: str | None = None, api_key: str | None = None,
              engine_dir: str | None = None, models_dir: str | None = None,
              stems: tuple[str, ...] = DEFAULT_STEMS,
              progress_cb: ProgressCB = None) -> list[str]:
    """Split ``pak_path`` into per-instrument stems, write them back into the pak,
    and rewrite the manifest. Returns the list of produced stem ids.

    ``engine`` is one of ``"remote"``, ``"audio-separator"``, ``"demucs"``.
    """
    pak_path = Path(pak_path)
    manifest = pak_io.read_manifest(pak_path)

    with tempfile.TemporaryDirectory(prefix="stemsplit_") as td:
        work = Path(td)
        mix = pak_io.extract_mix(pak_path, manifest, work)

        if engine == "remote":
            if not server_url:
                raise RuntimeError("remote engine selected but no split server configured")
            result_dir = _run_remote(mix, work, model or DEFAULT_REMOTE_MODEL,
                                     server_url, api_key, stems, progress_cb)
        elif engine == "audio-separator":
            result_dir = _run_audio_separator(mix, work, model or DEFAULT_REMOTE_MODEL,
                                               models_dir, engine_dir, progress_cb)
        elif engine == "demucs":
            result_dir = _run_demucs_local(mix, work, model or DEFAULT_DEMUCS_MODEL,
                                           engine_dir, progress_cb)
        else:
            raise RuntimeError(f"unknown split engine: {engine!r}")

        stem_files = _collect_stem_files(result_dir)
        if not stem_files:
            raise RuntimeError("separation produced no stem files")

        if progress_cb:
            progress_cb(0.8, "Encoding stems")

        add_files: dict[str, Path] = {}
        produced: list[dict] = []
        for wav in stem_files:
            stem_id = _sanitize(wav.stem)
            # Some engines suffix the mix name onto the stem file; keep the last token.
            if "_" in stem_id and stem_id.split("_")[-1] in pak_io.ALLOWED_STEM_IDS:
                stem_id = stem_id.split("_")[-1]
            rel = f"stems/{stem_id}.ogg"
            out_ogg = work / f"enc_{stem_id}.ogg"
            _encode_ogg(wav, out_ogg)
            add_files[rel] = out_ogg
            produced.append(pak_io.stem_entry(stem_id, rel, default=True))

        # Spec (updated): the full mix MUST remain in the pak as a fallback after
        # splitting. Keep the combined-mix file and list it as a `full` stem with
        # default:false (present but off) — unless the engine itself produced a
        # `full` stem. Also preserve `original_audio` so downstream players can
        # always fall back to the un-separated mix.
        #
        # IMPORTANT: the fallback keeps its ORIGINAL file verbatim — same bytes,
        # same extension. `mix_rel` points at the existing member (e.g.
        # stems/full.wav) and we neither re-encode nor rename it (it's never in
        # add_files). Only the newly-separated instrument stems are written as
        # .ogg. The feedpak spec allows non-Ogg baseline stems (§5.3.2), so a pack
        # authored with a WAV/FLAC full mix stays WAV/FLAC after a split.
        mix_rel = pak_io.find_mix_relpath(manifest)
        if mix_rel and not any(s["id"] == "full" for s in produced):
            produced.append(pak_io.stem_entry("full", mix_rel, default=False))

        produced.sort(key=lambda s: _STEM_ORDER.index(s["id"]) if s["id"] in _STEM_ORDER else 99)

        # Rewrite manifest: replace stems list, stamp provenance, keep the mix.
        new_manifest = dict(manifest)
        new_manifest["stems"] = produced
        if mix_rel:
            new_manifest["original_audio"] = mix_rel
        # stem_separation provenance (feedpak spec §5.3.1): `engine` is the stable
        # engine id, `model` the engine-specific checkpoint — the two are distinct
        # (engine=demucs, model=htdemucs_6s). The trio is a cache key.
        used_model = model or (DEFAULT_DEMUCS_MODEL if engine == "demucs" else DEFAULT_REMOTE_MODEL)
        engine_id = {"demucs": "demucs", "audio-separator": "audio-separator",
                     "remote": "bs-roformer"}.get(engine, engine)
        new_manifest["stem_separation"] = {
            "engine": engine_id,
            "model": used_model,
            "version": STEM_SEPARATION_SCHEMA_VERSION,
        }
        new_manifest.setdefault("feedpak_version", getattr(pak_io.sloppak_mod, "FEEDPAK_VERSION", "1.2.0"))

        if progress_cb:
            progress_cb(0.92, "Repacking")
        # NB: no `remove=` — the full mix stays in the pak as the fallback.
        pak_io.repack(pak_path, add_files=add_files, manifest=new_manifest)

    return [s["id"] for s in produced]

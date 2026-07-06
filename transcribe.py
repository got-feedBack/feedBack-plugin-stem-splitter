"""Lyrics transcription for the Stem Splitter plugin.

Reuses feedBack's surviving WhisperX primitives (``lib/lyrics_transcribe.py``)
and writes results back into the pak via ``pak_io``. If the pak has no isolated
vocal stem yet, it splits first (via ``split_stems``) to obtain one.

Provenance follows the feedpak spec §7.1 / §7.1.1: ``lyrics_source: transcribed``
plus a ``lyric_transcription: {engine, model, version}`` block.
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import pak_io
import split_stems

log = logging.getLogger("feedBack.plugin.stem_splitter")

ProgressCB = Optional[Callable[[float, str], None]]

# feedpak spec §7.1 lyrics_source vocabulary is {authored, transcribed, user}.
# NB: the running lib/sloppak.py still validates against the older
# {xml, notechart, whisperx, user} set and will fall back to "xml" for this
# value (cosmetic — has_lyrics is unaffected). Widening that enum is the clean
# server-side fix; tracked separately.
LYRICS_SOURCE = "transcribed"
LYRICS_JSON_NAME = "lyrics.json"


def _vocals_relpath(manifest: dict) -> str | None:
    for stem in manifest.get("stems") or []:
        if isinstance(stem, dict) and str(stem.get("id", "")).lower() == "vocals":
            f = stem.get("file")
            if isinstance(f, str) and f.strip():
                return f.strip()
    return None


def transcribe_pak(pak_path: Path, *, mode: str, server_url: str | None = None,
                   api_key: str | None = None, whisperx_model: str = "medium",
                   language: str | None = None, min_word_score: float = 0.35,
                   force: bool = False, engine_dir: str | None = None,
                   split_kwargs: dict | None = None,
                   progress_cb: ProgressCB = None) -> bool:
    """Transcribe lyrics for ``pak_path`` and write them back into the pak.

    ``mode`` is ``"remote"`` or ``"local"``. If no vocals stem exists, splits
    first using ``split_kwargs`` (passed to ``split_stems.split_pak``). Returns
    True if lyrics were written, False if skipped (already present & not forced,
    or the vocal stem was silent).
    """
    pak_path = Path(pak_path)
    manifest = pak_io.read_manifest(pak_path)

    if manifest.get("lyrics") and not force:
        log.info("stem_splitter: %s already has lyrics — skipping", pak_path.name)
        return False

    # Ensure a vocals stem exists (split first if needed).
    vocals_rel = _vocals_relpath(manifest)
    if not vocals_rel:
        # Auto-split to produce a vocal stem — but only if a split engine is
        # actually available. Without one, fail with a clear message instead of a
        # TypeError from split_pak's missing required `engine` argument.
        if not split_kwargs or not split_kwargs.get("engine"):
            raise RuntimeError(
                "this song has no vocal stem yet and no split engine is available "
                "to create one — configure a split server or install a local engine"
            )
        if progress_cb:
            progress_cb(0.05, "No vocal stem — splitting first")
        skw = dict(split_kwargs)
        split_stems.split_pak(pak_path, progress_cb=lambda p, m: progress_cb(0.05 + p * 0.5, m) if progress_cb else None, **skw)
        manifest = pak_io.read_manifest(pak_path)
        vocals_rel = _vocals_relpath(manifest)
        if not vocals_rel:
            raise RuntimeError("split did not produce a 'vocals' stem — cannot transcribe")

    from lyrics_transcribe import (
        transcribe_vocals_remote, transcribe_vocals_local, whisperx_available,
        vocals_has_signal, LYRIC_TRANSCRIPTION_ENGINE, LYRIC_TRANSCRIPTION_SCHEMA_VERSION,
    )

    with tempfile.TemporaryDirectory(prefix="stemsplit_lyr_") as td:
        work = Path(td)
        data = pak_io.read_member_bytes(pak_path, vocals_rel)
        if data is None:
            raise RuntimeError(f"vocals stem {vocals_rel!r} not found in pak")
        vocals = work / "vocals.ogg"
        vocals.write_bytes(data)

        # Silence gate — Whisper hallucinates on near-silent input.
        try:
            if not vocals_has_signal(vocals):
                log.info("stem_splitter: vocal stem of %s is near-silent — skipping", pak_path.name)
                return False
        except Exception:
            pass  # gate is best-effort (needs soundfile/numpy)

        if progress_cb:
            progress_cb(0.6, "Transcribing lyrics")

        if mode == "remote":
            if not server_url:
                raise RuntimeError("remote lyrics selected but no server configured")
            lyrics = transcribe_vocals_remote(
                vocals, server_url, language=language, api_key=api_key,
                min_word_score=min_word_score,
                progress_cb=lambda p, s, m: progress_cb(0.6 + p * 0.3, m) if progress_cb else None,
            )
        else:
            if engine_dir and engine_dir not in sys.path:
                sys.path.insert(0, engine_dir)
            if not whisperx_available():
                raise RuntimeError(
                    "local whisperx not installed — use 'Download local engine' "
                    "in the plugin settings, or configure a remote server"
                )
            lyrics = transcribe_vocals_local(
                vocals, model_size=whisperx_model, language=language,
                min_word_score=min_word_score,
                progress_cb=lambda p, s, m: progress_cb(0.6 + p * 0.3, m) if progress_cb else None,
            )

        if not lyrics:
            log.info("stem_splitter: no lyrics produced for %s", pak_path.name)
            return False

        lyrics_file = work / LYRICS_JSON_NAME
        lyrics_file.write_text(json.dumps(lyrics, separators=(",", ":")), encoding="utf-8")

        new_manifest = dict(manifest)
        new_manifest["lyrics"] = LYRICS_JSON_NAME
        new_manifest["lyrics_source"] = LYRICS_SOURCE
        new_manifest["lyric_transcription"] = {
            "engine": LYRIC_TRANSCRIPTION_ENGINE,
            "model": whisperx_model if mode == "local" else "remote",
            "version": LYRIC_TRANSCRIPTION_SCHEMA_VERSION,
        }
        new_manifest.setdefault("feedpak_version", getattr(pak_io.sloppak_mod, "FEEDPAK_VERSION", "1.2.0"))

        if progress_cb:
            progress_cb(0.95, "Repacking")
        pak_io.repack(pak_path, add_files={LYRICS_JSON_NAME: lyrics_file}, manifest=new_manifest)

    return True

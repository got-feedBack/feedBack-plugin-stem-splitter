"""Re-align a pak's EXISTING lyrics to its vocal stem.

The words are already right; the timings aren't. That happens constantly: lyrics pasted from the
web, imported from a Guitar Pro file, or transcribed against a different mix — correct text,
drifting or wholly wrong timestamps. Until now the only repair was "Transcribe lyrics", which
throws the words away and lets Whisper guess them again — so you fixed the timing by corrupting
the text, and a mis-heard line ("hold me closer, Tony Danza") is a worse outcome than a late one.

This is the other half of the same machinery, and the endpoint it uses is the one the plugin
spent months calling by mistake:

    /transcribe   "what are the lyrics, and when is each sung?"     (no text)
    /align        "here are the lyrics — when is each word sung?"   (needs text)  ← this

So a re-align never invents a word. It sends the lyrics you already have and keeps every one of
them, replacing only the timings.

**The manifest is not touched.** `lyrics_source` is a closed vocabulary in the feedpak spec
(§7.1: authored | transcribed | user), and re-aligning does not change where the words came
from — authored lyrics stay authored. A pak whose timings were repaired is not a pak whose
provenance changed, and inventing a value to say so would be a spec change, not a plugin change.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Callable, Optional

import pak_io

log = logging.getLogger("feedBack.plugin.stem_splitter.realign")

ProgressCB = Optional[Callable[[float, str], None]]

LYRICS_JSON_NAME = "lyrics.json"

# How much of a server error body to keep. Long enough for a FastAPI validation body or the last
# line of a traceback — the parts that actually say what went wrong.
_MAX_ERR_BODY = 4000


def _err_body(resp) -> str:
    text = (getattr(resp, "text", "") or "").strip()
    if len(text) <= _MAX_ERR_BODY:
        return text
    # Head AND tail: on a traceback the exception is the LAST line, and a head-only cut throws
    # away the answer while keeping the preamble.
    marker = f"\n… [truncated, {len(text)} chars total] …\n"
    budget = max(0, _MAX_ERR_BODY - len(marker))
    head = budget * 2 // 3
    tail = budget - head
    return text[:head].rstrip() + marker + text[len(text) - tail:].lstrip()


def lyrics_to_text(tokens: list[dict]) -> str:
    """sloppak syllable tokens -> the plain text /align wants.

    The on-disk shape (feedpak spec §2.3) is not words: it is syllables, carrying two suffixes
    on `w` — `-` joins to the next syllable, `+` ends a line. Both are SUFFIXES on real
    syllables, never standalone tokens. So "to-geth-er+" is one word ending a line, and naively
    splitting on tokens would send the aligner three words that aren't words.

    Rebuild real words (drop the `-` joins), and real lines (break on `+`).
    """
    lines: list[list[str]] = [[]]
    word = ""
    for tok in tokens or []:
        if not isinstance(tok, dict):
            continue
        raw = str(tok.get("w") or "")
        if not raw:
            continue
        ends_line = raw.endswith("+")
        if ends_line:
            raw = raw[:-1]
        joins = raw.endswith("-")
        if joins:
            raw = raw[:-1]

        word += raw
        if not joins:                      # the word is complete
            if word:
                lines[-1].append(word)
            word = ""
        if ends_line:
            if word:                       # a line that ended mid-word: keep the fragment
                lines[-1].append(word)
                word = ""
            lines.append([])

    if word:
        lines[-1].append(word)
    return "\n".join(" ".join(ln) for ln in lines if ln)


def segments_to_lyrics(segments: list[dict]) -> list[dict]:
    """/align word-granularity output -> sloppak tokens.

    The server marks the FIRST word of each line with `new_line`. sloppak marks the LAST syllable
    of a line with a `+` suffix. So the break has to be moved back one token — writing `+` on the
    word the server flagged would end every line one word early, and the renderer would show the
    break in the wrong place on every single line.
    """
    words: list[dict] = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text") or "").strip()
        start, end = seg.get("start"), seg.get("end")
        if not text or start is None or end is None:
            continue
        words.append({
            "t": round(float(start), 3),
            "d": round(max(0.0, float(end) - float(start)), 3),
            "w": text,
            "_nl": bool(seg.get("new_line")),
        })

    out: list[dict] = []
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        text = w["w"]
        if nxt is not None and nxt["_nl"]:
            text += "+"                    # this word is the LAST of its line
        out.append({"t": w["t"], "d": w["d"], "w": text})
    return out


def _vocals_relpath(manifest: dict) -> str | None:
    """The vocal stem's path, matched exactly as transcribe.py matches it.

    Case-insensitive and trimmed on purpose: a pak written by another tool can carry `"Vocals"`,
    and matching case-sensitively here would mean the same manifest transcribes fine and then
    re-aligns with "this song has no vocal stem" — two features disagreeing about what is
    plainly there, which reads as a broken pak rather than a broken plugin.
    """
    for stem in manifest.get("stems") or []:
        if isinstance(stem, dict) and str(stem.get("id", "")).strip().lower() == "vocals":
            f = stem.get("file")
            if isinstance(f, str) and f.strip():
                return f.strip()
    return None


def align_vocals_remote(vocals: Path, text: str, server_url: str, *,
                        language: str | None = None, api_key: str | None = None,
                        timeout: int = 300, progress_cb: ProgressCB = None) -> list[dict]:
    """POST the stem AND the lyrics to `/align`, ask for word granularity."""
    import requests

    server_url = server_url.rstrip("/")
    if progress_cb:
        progress_cb(0.65, f"Aligning against the vocal stem ({server_url})")

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Everything the server reads is a FORM field — `text`, `language` and `granularity` all come
    # from Form(...), and a query param is silently ignored. (That mistake, in the other
    # direction, is what made remote transcription 422 for months: #17.)
    form = {"text": text, "granularity": "word"}
    if language:
        form["language"] = language

    try:
        with open(vocals, "rb") as f:
            resp = requests.post(
                f"{server_url}/align",
                files={"file": (vocals.name, f, "audio/ogg")},
                data=form,
                headers=headers or None,
                timeout=timeout,
            )
    except requests.RequestException as e:
        raise RuntimeError(f"could not reach the WhisperX server at {server_url}: {e}") from e
    except OSError as e:
        raise RuntimeError(f"could not read the vocal stem {vocals.name}: {e}") from e

    # Read the body, THEN hand the connection back to the pool. Raising with the response still
    # open holds it out of the pool until GC, and a batch re-align does this once per song.
    try:
        if resp.status_code != 200:
            raise RuntimeError(
                f"WhisperX server error ({resp.status_code}): {_err_body(resp)}")

        try:
            data = resp.json()
        except ValueError as e:
            # A proxy's HTML error page, or an empty body, with a 200 on it. `resp.json()` alone
            # raises a decode error that names a byte offset and tells the user nothing; show
            # them what the server actually said.
            raise RuntimeError(
                f"the server answered 200 but not JSON ({e}): {_err_body(resp)}") from e

        segments = data.get("segments") if isinstance(data, dict) else None
        if not isinstance(segments, list):
            raise RuntimeError(f"the server returned no segments: {_err_body(resp)}")
    finally:
        resp.close()

    return segments_to_lyrics(segments)


def realign_pak(pak_path: Path, *, server_url: str | None = None, api_key: str | None = None,
                language: str | None = None,
                cancel_cb: Optional[Callable[[], None]] = None,
                progress_cb: ProgressCB = None) -> bool:
    """Re-time ``pak_path``'s existing lyrics against its vocal stem. Words are never changed.

    Returns True if the lyrics were rewritten. Raises if there is nothing to re-align (no lyrics,
    or no vocal stem) — those are user-visible mistakes, not silent no-ops: a button that
    "succeeds" and does nothing is worse than one that says why it can't.
    """
    pak_path = Path(pak_path)
    manifest = pak_io.read_manifest(pak_path)

    lyrics_rel = manifest.get("lyrics")
    if not lyrics_rel:
        raise RuntimeError(
            "this song has no lyrics to re-align — use 'Transcribe lyrics' to create some"
        )

    vocals_rel = _vocals_relpath(manifest)
    if not vocals_rel:
        # Deliberately NOT splitting here. Re-align is the cheap, safe repair — a few seconds
        # against a stem you already have. Silently kicking off a multi-minute GPU split because
        # a menu item was clicked is not what anybody asked for; "Split stems" is right there.
        raise RuntimeError(
            "this song has no vocal stem to align against — run 'Split stems' first"
        )

    if cancel_cb:
        cancel_cb()

    if not server_url:
        raise RuntimeError(
            "re-aligning needs a demucs/WhisperX server — configure one in the plugin settings"
        )

    raw = pak_io.read_member_bytes(pak_path, str(lyrics_rel))
    if raw is None:
        raise RuntimeError(f"lyrics file {lyrics_rel!r} is named in the manifest but not in the pak")
    try:
        tokens = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise RuntimeError(f"the pak's {lyrics_rel} is not readable JSON: {e}") from e
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError(f"the pak's {lyrics_rel} holds no lyrics")

    text = lyrics_to_text(tokens)
    if not text.strip():
        raise RuntimeError("the existing lyrics have no words in them")

    if progress_cb:
        progress_cb(0.1, f"Re-aligning {len(tokens)} syllables")

    with tempfile.TemporaryDirectory(prefix="stemsplit_realign_") as td:
        work = Path(td)
        data = pak_io.read_member_bytes(pak_path, vocals_rel)
        if data is None:
            raise RuntimeError(f"vocals stem {vocals_rel!r} not found in pak")
        vocals = work / "vocals.ogg"
        vocals.write_bytes(data)

        if cancel_cb:
            cancel_cb()

        lyrics = align_vocals_remote(
            vocals, text, server_url, language=language, api_key=api_key,
            progress_cb=progress_cb,
        )
        if not lyrics:
            # The words went in and nothing came back with a timestamp. Do NOT write that:
            # replacing a song's lyrics with an empty file, on a "re-align" click, would destroy
            # the one thing the user was trying to keep.
            raise RuntimeError(
                "the aligner produced no timings — the vocal stem may be silent, or the lyrics "
                "may be in a different language than the audio"
            )

        if cancel_cb:
            cancel_cb()

        if progress_cb:
            progress_cb(0.95, "Repacking")

        out = work / LYRICS_JSON_NAME
        out.write_text(json.dumps(lyrics, separators=(",", ":")), encoding="utf-8")
        # Write to the path the manifest already names, and pass no manifest: the words, their
        # source, and every other key are exactly as they were. Only the timings moved.
        pak_io.repack(pak_path, add_files={str(lyrics_rel): out})

    log.info("stem_splitter: re-aligned %d syllables in %s", len(lyrics), pak_path.name)
    return True

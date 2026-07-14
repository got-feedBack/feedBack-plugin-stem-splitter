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


def segments_to_words(segments: list[dict]) -> list[dict]:
    """/align word-granularity output -> a plain timed word list. Nothing sloppak-shaped yet."""
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
        })
    return words


def _token_words(tokens: list[dict]) -> list[tuple[str, list[int]]]:
    """Group the pak's SYLLABLE tokens into words: [(comparable_word, [token indices]), ...]."""
    words: list[tuple[str, list[int]]] = []
    buf, idx = "", []
    for i, tok in enumerate(tokens):
        if not isinstance(tok, dict):
            continue
        raw = str(tok.get("w") or "")
        if not raw:
            continue
        body = raw[:-1] if raw.endswith("+") else raw
        joins = body.endswith("-")
        if joins:
            body = body[:-1]
        buf += body
        idx.append(i)
        if not joins:
            key = _words(buf)
            words.append((key[0] if key else "", list(idx)))
            buf, idx = "", []
    if idx:
        key = _words(buf)
        words.append((key[0] if key else "", list(idx)))
    return words


def retime_tokens(tokens: list[dict], aligned: list[dict]) -> list[dict]:
    """Rebuild the pak's lyrics from ITS OWN tokens, changing only `t` and `d`.

    This is the whole feature, and the earlier version got it subtly wrong: it built the new
    lyrics out of the words the SERVER returned. Forced alignment legitimately drops a word it
    can't place — so every dropped word was silently deleted from the user's song. A re-align
    that quietly loses "dancer" from the chorus has not kept the promise; it has broken it in a
    way nobody notices until they play the song.

    So the output is the user's tokens, in the user's order, with the user's exact text —
    including the `-` joins and `+` line breaks, which means the line structure they authored
    survives too, rather than being replaced by wherever the aligner decided a line was. Only the
    numbers move.

    An unaligned word keeps its place and is given a plausible span, interpolated across the gap
    between the neighbours that DID align. It is better to sing a word slightly early than to
    lose it.
    """
    groups = _token_words(tokens)
    got = [dict(w, _k=(_words(w["w"])[:1] or [""])[0]) for w in aligned]

    # Match aligned words onto the original words, in order (the same subsequence rule the guard
    # enforces): the aligner may skip, never invent or reorder.
    spans: list[tuple[float, float] | None] = [None] * len(groups)
    gi = 0
    for w in got:
        while gi < len(groups) and groups[gi][0] != w["_k"]:
            gi += 1
        if gi >= len(groups):
            break
        spans[gi] = (float(w["t"]), float(w["t"]) + float(w["d"]))
        gi += 1

    anchored = [i for i, s in enumerate(spans) if s is not None]
    if not anchored:
        raise RuntimeError("no word in your lyrics could be matched to the aligner's output")

    # Fill the unaligned words by interpolation, so they keep their place in the song.
    _AVG = 0.35          # seconds; only used at the very edges, where there is nothing to divide
    for i in range(len(groups)):
        if spans[i] is not None:
            continue
        prev = max((a for a in anchored if a < i), default=None)
        nxt = min((a for a in anchored if a > i), default=None)
        if prev is not None and nxt is not None:
            # Share the gap between the neighbours among the unaligned run.
            run = [j for j in range(prev + 1, nxt) if spans[j] is None]
            start, end = spans[prev][1], spans[nxt][0]
            width = max(0.0, end - start) / max(1, len(run))
            for k, j in enumerate(run):
                spans[j] = (start + k * width, start + (k + 1) * width)
        elif nxt is not None:                       # unaligned words before the first anchor
            run = [j for j in range(0, nxt) if spans[j] is None]
            end = spans[nxt][0]
            width = min(_AVG, end / max(1, len(run))) if end > 0 else 0.0
            for k, j in enumerate(run):
                spans[j] = (max(0.0, end - (len(run) - k) * width),
                            max(0.0, end - (len(run) - k - 1) * width))
        else:                                       # ...and after the last one
            run = [j for j in range(prev + 1, len(groups)) if spans[j] is None]
            start = spans[prev][1]
            for k, j in enumerate(run):
                spans[j] = (start + k * _AVG, start + (k + 1) * _AVG)

    # Split each word's span across its syllables, proportional to their length — a long syllable
    # gets a long slice, which is closer to how they are actually sung than an even split.
    out = [dict(t) for t in tokens if isinstance(t, dict)]
    for (_key, idxs), span in zip(groups, spans):
        start, end = span
        total = max(0.0, end - start)
        sizes = [max(1, len(str(out[i].get("w") or "").rstrip("+").rstrip("-"))) for i in idxs]
        span_total = sum(sizes)
        cursor = start
        for i, size in zip(idxs, sizes):
            share = total * (size / span_total) if span_total else 0.0
            out[i]["t"] = round(cursor, 3)
            out[i]["d"] = round(share, 3)
            cursor += share
    return out


def _words(text: str) -> list[str]:
    """Comparable words: case- and punctuation-insensitive.

    The aligner is allowed to hand back "Closer," for "closer" — that is not a changed lyric, and
    refusing over it would make the guard below fire on every real song.
    """
    out: list[str] = []
    for w in (text or "").split():
        w = w.strip(".,!?;:\"'()[]-—…").lower()
        if w:
            out.append(w)
    return out


# How much of the original the aligner has to actually time before we believe it. Forced
# alignment legitimately drops the odd word it can't place (a shout, a word buried under a
# cymbal), so demanding 100% would refuse good results. But a handful of words out of a whole
# song is not an alignment, it is noise — and writing THAT back would gut the lyrics while
# reporting success.
_MIN_COVERAGE = 0.5


def _verify_words_survived(original: str, aligned: list[dict]) -> None:
    """The promise of this feature is that it does not touch your words. Check it.

    Forced alignment returns OUR text with timings, so the words coming back must be the words
    that went in. What it may legitimately do is DROP one it couldn't place. What it must never
    do is invent one, reorder them, or hand back a tenth of the song — and if any of that
    happens, the right answer is to refuse, because we are about to overwrite the user's lyrics
    with whatever this is.
    """
    want = _words(original)
    got = _words(" ".join(str(t.get("w") or "").rstrip("+").rstrip("-") for t in aligned))
    if not got:
        raise RuntimeError("the aligner returned no words at all")

    # Every returned word must appear, in order, in the original: a subsequence. That admits
    # dropped words and refuses invented or reordered ones.
    i = 0
    for w in got:
        while i < len(want) and want[i] != w:
            i += 1
        if i >= len(want):
            raise RuntimeError(
                "the aligner returned words that are not in your lyrics, so this is not a "
                "re-align — refusing to overwrite them. (Use 'Transcribe lyrics' if you meant "
                "to replace the words.)"
            )
        i += 1

    coverage = len(got) / max(1, len(want))
    if coverage < _MIN_COVERAGE:
        raise RuntimeError(
            f"the aligner only placed {len(got)} of {len(want)} words "
            f"({coverage:.0%}) — refusing to overwrite your lyrics with a partial result. "
            f"The vocal stem may not match these lyrics, or the language may be wrong."
        )


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


# Match split_stems._run_remote: the content-type follows the FILE, not a guess. A pak from
# another tool can carry a .wav or .flac vocals stem, and telling a server (or a proxy in front
# of it) that a wav is an ogg is how you get a rejected upload or a mis-decoded one.
_CONTENT_TYPES = {
    ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
    ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4", ".aac": "audio/aac",
}


def _content_type(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def align_vocals_remote(vocals: Path, text: str, server_url: str, *,
                        language: str | None = None, api_key: str | None = None,
                        timeout: int = 300, progress_cb: ProgressCB = None) -> list[dict]:
    """POST the stem AND the lyrics to `/align`. Returns the server's TIMED WORDS.

    Deliberately not sloppak tokens: the pak's lyrics get rebuilt from the pak's OWN tokens (see
    retime_tokens). Nothing the server says about the text is written to disk — only its numbers.
    """
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
                files={"file": (vocals.name, f, _content_type(vocals))},
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

    return segments_to_words(segments)


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
        # Keep the pak's own suffix. Writing a .wav out as "vocals.ogg" mislabels it to the
        # server and to any proxy in between, and makes the content-type above a lie.
        suffix = Path(str(vocals_rel)).suffix or ".ogg"
        vocals = work / f"vocals{suffix}"
        vocals.write_bytes(data)

        if cancel_cb:
            cancel_cb()

        aligned = align_vocals_remote(
            vocals, text, server_url, language=language, api_key=api_key,
            progress_cb=progress_cb,
        )
        if not aligned:
            # The words went in and nothing came back with a timestamp. Do NOT write that:
            # replacing a song's lyrics with an empty file, on a "re-align" click, would destroy
            # the one thing the user was trying to keep.
            raise RuntimeError(
                "the aligner produced no timings — the vocal stem may be silent, or the lyrics "
                "may be in a different language than the audio"
            )

        # The whole promise, checked rather than assumed: a server that hands back different
        # words — normalized, re-tokenized, or simply misbehaving — must not get anywhere near
        # the user's lyrics.
        _verify_words_survived(text, aligned)

        # Rebuild from the ORIGINAL tokens. The server's words are used for their timings and
        # nothing else, so a word the aligner failed to place keeps its text, its place in the
        # line, and an interpolated span — rather than being quietly deleted from the song.
        lyrics = retime_tokens(tokens, aligned)

        if cancel_cb:
            cancel_cb()

        if progress_cb:
            progress_cb(0.95, "Repacking")

        out = work / LYRICS_JSON_NAME
        out.write_text(json.dumps(lyrics, separators=(",", ":")), encoding="utf-8")
        # Write to the path the manifest already names, and pass no manifest: the words, their
        # source, and every other key are exactly as they were. Only the timings moved.
        pak_io.repack(pak_path, add_files={str(lyrics_rel): out})

    log.info("stem_splitter: re-aligned %d of %d syllables in %s (%d words timed by the aligner)",
             len(lyrics), len(tokens), pak_path.name, len(aligned))
    return True

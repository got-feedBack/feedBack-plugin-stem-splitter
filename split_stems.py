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

import importlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

# NB: ``pak_io`` (which imports the core ``sloppak`` lib) is imported lazily inside
# ``split_pak`` — keeping this module import-clean means its pure helpers
# (``_normalize_stem_id``, ``_sanitize``) can be unit-tested without the host.

log = logging.getLogger("feedBack.plugin.stem_splitter")

ProgressCB = Optional[Callable[[float, str], None]]
# Optional cancellation checkpoint: a no-arg callable the caller supplies that
# RAISES when the job has been asked to cancel. Engines invoke it at safe points
# (loop iterations, around the heavy step) so a cancel actually interrupts an
# in-flight separation instead of only taking effect for still-queued jobs.
CancelCB = Optional[Callable[[], None]]

STEM_SEPARATION_SCHEMA_VERSION = "1.0.0"
DEFAULT_REMOTE_MODEL = "bs_roformer_sw"
DEFAULT_DEMUCS_MODEL = "htdemucs_6s"
# Requested stem set for engines that accept one. bs_roformer variants may
# ignore this and return their own set — we write back whatever comes out.
DEFAULT_STEMS = ("drums", "bass", "vocals", "other", "guitar", "piano")
_STEM_ORDER = ["guitar", "bass", "drums", "vocals", "piano", "other", "full"]

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")

# Remote-split job limits. The server allows a roformer job up to 30 min, so give
# it headroom rather than the old 10-min cap (which failed long splits mid-flight).
_JOB_TIMEOUT = 35 * 60
# The server returns 503 once MAX_CONCURRENT separations are in flight; a batch
# split will hit that routinely, so back off and retry instead of erroring.
_BUSY_RETRIES = 6
_BUSY_BASE_BACKOFF = 5
_BUSY_MAX_BACKOFF = 60


def _sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_").lower()
    return s or "stem"


# Canonical feedpak stem ids the v3 library filter understands. Separators label
# their outputs inconsistently — demucs writes bare "drums.wav"/"guitar.wav",
# while audio-separator (BS-Roformer-SW etc.) writes "<mix>_(Guitar)_<model>.flac"
# — so map the label onto a canonical id instead of trusting the raw filename
# (which otherwise yields ids like "mix_guitar_bs_roformer_sw" the library ignores).
_STEM_ALIASES = {
    "vocals": "vocals", "vocal": "vocals", "voice": "vocals",
    "drums": "drums", "drum": "drums",
    "bass": "bass",
    "guitar": "guitar", "guitars": "guitar",
    "piano": "piano", "keys": "piano", "keyboard": "piano",
    "other": "other",
    # A 2-stem model (should one be configured) emits an instrumental companion
    # to vocals; map it to the catch-all "other" so it is still a recognized stem.
    "instrumental": "other", "instruments": "other", "instrument": "other",
    "music": "other", "accompaniment": "other",
    "no_vocals": "other", "novocals": "other",
}


def _normalize_stem_id(raw_name: str) -> str | None:
    """Best-effort map a separator's output stem label to a canonical stem id.

    Prefers audio-separator's parenthesised ``<base>_(<Label>)_<model>`` label,
    exactly as slopsmith-demucs-server does (``server.py`` ``_run_roformer``);
    falls back to a whole-token scan (longest alias first, so ``no_vocals`` beats
    a bare ``vocals``) for demucs' bare names and the server's clean keys.
    Returns None when nothing maps — the caller keeps a sanitized fallback id.
    """
    m = re.search(r"_\(([^)]+)\)_", raw_name)
    if m:
        lbl = re.sub(r"[^a-z0-9]+", "_", m.group(1).lower()).strip("_")
        if lbl in _STEM_ALIASES:
            return _STEM_ALIASES[lbl]
    s = re.sub(r"[^a-z0-9]+", "_", raw_name.lower()).strip("_")
    for alias in sorted(_STEM_ALIASES, key=len, reverse=True):
        if re.search(rf"(^|_){re.escape(alias)}(_|$)", s):
            return _STEM_ALIASES[alias]
    return None


def _origin(url: str) -> tuple[str, str, int | None] | None:
    """(scheme, host, port) with the DEFAULT port made explicit, or None if unparseable.

    Comparing raw netlocs is wrong: `https://h` and `https://h:443` are the same origin
    but different strings, so a server that hands back an explicit default port would
    have its own download URL treated as third-party and the API key withheld — breaking
    an authenticated download for no reason. Normalize instead of string-matching.

    Uses .hostname (not .netloc), which also strips any `user:pass@` — so
    `http://good.example.com@evil.com/` correctly resolves to host `evil.com`.
    """
    from urllib.parse import urlsplit
    try:
        s = urlsplit(str(url))
        port = s.port                       # raises ValueError on a bad port
    except ValueError:
        return None
    scheme = (s.scheme or "").lower()
    host = (s.hostname or "").lower()
    if port is None:
        port = {"http": 80, "https": 443}.get(scheme)
    return (scheme, host, port)


def _same_origin(url: str, server_url: str) -> bool:
    """Is `url` on the same scheme+host+port as the configured server?

    Used to decide whether the API key may be attached — never send credentials to
    a host the server merely *pointed* at.
    """
    from urllib.parse import urlparse
    try:
        a, b = urlparse(str(url)), urlparse(server_url)
    except ValueError:
        return False
    # "Relative" means exactly one thing here: an absolute PATH that the caller
    # rewrites onto server_url (it only does that for a leading "/"). Anything else is
    # not ours, and the key must not ride along:
    #   * "https:evil.com/steal"  -> scheme set, netloc empty. NOT relative.
    #   * "//evil.com/steal"      -> netloc set (scheme inherited). NOT relative.
    #   * "::::" / "download/x"   -> no scheme, no netloc, but we never rewrite these
    #                                onto server_url, so they aren't ours either.
    if not a.scheme and not a.netloc:
        return str(url).startswith("/") and not str(url).startswith("//")
    oa, ob = _origin(url), _origin(server_url)
    return oa is not None and ob is not None and oa == ob


def _module_cmd(engine_dir: str | None, module: str, argv: list[str]) -> list[str]:
    """Build a command that runs ``python -m <module>`` in a subprocess that can
    actually SEE the plugin's engine packages.

    A plain ``-m demucs`` cannot: the engine is a ``pip --target`` tree that we add
    to *this* process's ``sys.path``, and a child does not inherit sys.path. Setting
    ``PYTHONPATH`` doesn't fix it either — the packaged Windows app bundles the
    python.org embeddable distribution, which runs in isolated ``._pth`` mode where
    **PYTHONPATH is ignored**. So inject the path in-process via ``-c``, which works
    on every platform, packaged or dev.
    """
    code = (
        "import sys, runpy\n"
        "e = sys.argv.pop(1)\n"
        "if e:\n"
        "    sys.path.insert(0, e)\n"
        "runpy.run_module(%r, run_name='__main__', alter_sys=True)\n" % module
    )
    return [sys.executable, "-c", code, engine_dir or "", *argv]


def _prepend_engine_path(engine_dir: str | None) -> None:
    """Make an opt-in-installed engine importable (pip --target dir).

    Invalidate import caches whenever an engine dir is given (even one already
    on sys.path): if a split was attempted BEFORE the engine was installed (the
    natural first-run flow — try to split, get told to install, install, try
    again), the failed import poisons
    ``sys.path_importer_cache`` with a negative finder for the then-nonexistent
    engine dir, and every later in-process import fails until the app restarts.
    ``importlib.invalidate_caches()`` is the documented requirement when
    packages are installed at runtime; it is cheap and only runs on the split
    path, never per frame.
    """
    if engine_dir and engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    if engine_dir:
        importlib.invalidate_caches()


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


_MAX_REDIRECTS = 5
_REDIRECT_CODES = (301, 302, 303, 307, 308)


def _redact_url(url: str) -> str:
    """scheme://host/path — drop the query and fragment before logging.

    The URLs we refuse to authenticate to are attacker- or third-party-chosen, and a
    redirect target is very often a PRE-SIGNED url (S3 &X-Amz-Signature=…, a bare
    ?token=…). Logging one verbatim writes someone's credential into the app log, which
    is exactly the leak we just refused to commit over the wire.
    """
    from urllib.parse import urlparse
    try:
        u = urlparse(str(url))
    except ValueError:
        return "<unparseable url>"
    if not u.scheme and not u.netloc:
        return (u.path or "") + ("?…" if u.query else "")
    return f"{u.scheme}://{u.netloc}{u.path}" + ("?…" if u.query else "")


def _get_authed(url: str, server_url: str, headers: dict | None, timeout: float):
    """GET `url`, following redirects BY HAND so the API key can never ride off-origin.

    ``requests`` drops the ``Authorization`` header on a cross-host redirect, but it
    knows nothing about the ``X-API-Key`` header this server authenticates with — it
    would happily forward it. So a compromised or malicious split server could return a
    perfectly on-origin download URL that 302s to a host it controls and harvest the
    user's key. ``_same_origin`` alone doesn't catch that: it only ever sees the first
    hop.

    Redirects are therefore disabled and re-evaluated per hop: the key is attached only
    while the current URL is still on the server's origin.

    Returns ``(response, final_url)`` — the caller needs the final URL because the stem
    file extension is derived from it.
    """
    import requests
    from urllib.parse import urljoin

    for _ in range(_MAX_REDIRECTS + 1):
        hop_headers = headers if _same_origin(url, server_url) else None
        if headers and hop_headers is None:
            log.warning("stem_splitter: %s is off-origin from %s - requesting without "
                        "the API key", _redact_url(url), server_url)
        resp = requests.get(url, headers=hop_headers, timeout=timeout,
                            allow_redirects=False)
        location = resp.headers.get("location") if resp.status_code in _REDIRECT_CODES \
            else None
        if not location:
            return resp, url
        resp.close()
        url = urljoin(url, location)   # absolute, so the next _same_origin check is real

    raise RuntimeError(f"split server sent more than {_MAX_REDIRECTS} redirects")


# ── Engines ──────────────────────────────────────────────────────────────────

# How much of a server error body to keep.
#
# To be precise about what this did and didn't cause: the truncation a user actually reported
# (#16) was the UI's ellipsis, not this cap — a bare FastAPI 422 body is ~139 chars and fit
# inside 300 fine. This is the truncation BEHIND that one, and it bites the errors that carry
# the most diagnosis, because those are the long ones: a multi-field validation body, a 500
# whose traceback answers on its LAST line, an HTML error page from a reverse proxy. 300 chars
# keeps the header and throws away the answer.
#
# The cap still exists so a server answering with a 2 MB HTML page can't push a novel into the
# job record — which is persisted to disk and re-read on every load. 4000 chars holds any real
# API error whole, and _err_body() says so when it has to cut.
_MAX_ERR_BODY = 4000


def _err_body(resp) -> str:
    """The server's error body, whole if it plausibly is one, and marked when it isn't."""
    text = resp.text or ""
    if len(text) <= _MAX_ERR_BODY:
        return text.strip()
    return text[:_MAX_ERR_BODY].strip() + f"\n… [truncated, {len(text)} chars total]"


def _run_remote(mix: Path, out_dir: Path, model: str, server_url: str,
                api_key: str | None, stems: tuple[str, ...],
                progress_cb: ProgressCB, cancel_cb: CancelCB = None) -> Path:
    """POST the mix to ``{server_url}/separate`` and download the stems."""
    import requests

    server_url = server_url.rstrip("/")
    if progress_cb:
        progress_cb(0.10, f"Uploading to split server ({server_url})")

    content_type = "audio/wav" if mix.suffix.lower() == ".wav" else "audio/ogg"
    params: dict[str, str] = {"model": model}
    if stems:
        params["stems"] = ",".join(stems)
    # The server authenticates with X-API-Key (or an ?api_key= query param) - NOT
    # an Authorization: Bearer header. Only /health, /docs and /openapi.json are
    # exempt, so /jobs and /download need the key too.
    headers = {"X-API-Key": api_key} if api_key else None

    # The server caps concurrent separations (MAX_CONCURRENT) and answers 503 when
    # it's saturated - which a batch split WILL hit. Back off and retry instead of
    # failing the job.
    resp = None
    for attempt in range(_BUSY_RETRIES):
        if cancel_cb:
            cancel_cb()
        with open(mix, "rb") as f:
            resp = requests.post(
                f"{server_url}/separate",
                files={"file": (mix.name, f, content_type)},
                params=params, headers=headers, timeout=600,
            )
        if resp.status_code != 503:
            break
        if attempt == _BUSY_RETRIES - 1:
            # Last attempt: don't sleep (nobody is going to retry after it) and don't
            # close the response - the error path below needs to read resp.text.
            break
        # We're going to retry: release the connection back to the pool instead of
        # holding it open across the backoff (a batch split would otherwise pin one
        # connection per in-flight retry).
        resp.close()
        wait = min(_BUSY_MAX_BACKOFF, _BUSY_BASE_BACKOFF * (2 ** attempt))
        if progress_cb:
            progress_cb(0.12, f"Split server busy - retrying in {wait}s")
        # Sleep in slices and check for cancellation between them: with backoff up
        # to a minute, a single sleep(wait) would leave a user's cancel unnoticed
        # for that whole time.
        deadline = time.time() + wait
        while time.time() < deadline:
            if cancel_cb:
                cancel_cb()  # raises to abort
            time.sleep(min(0.5, max(0.0, deadline - time.time())))

    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp is not None else "no response"
        body = ""
        if resp is not None:
            body = _err_body(resp)
            # Read the body first, then hand the connection back to the pool. Raising
            # with the response still open holds it out of the pool until GC — and a
            # batch that exhausts the retries on 503 does this once per song.
            resp.close()
        raise RuntimeError(f"split server error ({code}): {body}")

    data = resp.json()
    stem_urls = data.get("stems") or {}
    if not stem_urls and data.get("job_id"):
        job_id = data["job_id"]
        # Wall-clock deadline, not an iteration count: the server allows a roformer
        # job up to 30 min, so the old 120x5s (=10 min) cap failed long splits that
        # were actually still running.
        deadline = time.time() + _JOB_TIMEOUT
        while time.time() < deadline:
            time.sleep(5)
            if cancel_cb:
                cancel_cb()  # raises to abort a canceled job between polls
            # Same redirect discipline as the stem downloads: the poll carries the key,
            # so the server must not be able to bounce it to a host of its choosing.
            jresp, _ = _get_authed(f"{server_url}/jobs/{job_id}", server_url,
                                   headers, timeout=30)
            if jresp.status_code != 200:
                # A 404/500/HTML error page would blow up .json() with an opaque
                # decode error; surface what actually happened instead.
                raise RuntimeError(
                    f"split server job poll failed ({jresp.status_code}): "
                    f"{_err_body(jresp)}")
            try:
                jr = jresp.json()
            except ValueError as e:
                raise RuntimeError(
                    f"split server returned a non-JSON job response: {_err_body(jresp)}"
                ) from e
            status = jr.get("status")
            if status == "complete":
                stem_urls = jr.get("stems") or {}
                break
            if status == "failed":
                raise RuntimeError(f"split server job failed: {jr.get('error')}")
            if progress_cb:
                pct = jr.get("progress")
                detail = f"{pct}%" if isinstance(pct, (int, float)) else (status or "working")
                progress_cb(0.4, f"Separating on server ({detail})")
        else:
            raise RuntimeError(
                f"split server job timed out after {_JOB_TIMEOUT // 60} min")
    if not stem_urls:
        raise RuntimeError("split server returned no stems")

    result_dir = out_dir / "remote_stems"
    result_dir.mkdir(parents=True, exist_ok=True)
    for name, url in stem_urls.items():
        if isinstance(url, str) and url.startswith("/"):
            url = f"{server_url}{url}"
        # The stem URLs come from the server's RESPONSE, and each hop of a redirect chain
        # is attacker-choosable from there. _get_authed re-checks the origin per hop and
        # only attaches the key while we're still talking to the configured server.
        sr, final_url = _get_authed(url, server_url, headers, timeout=180)
        # Skipping a failed download silently produced a pak that LOOKS split but is
        # missing stems, and the job still reported success. Fail loudly instead so the
        # user can retry.
        if sr.status_code != 200:
            raise RuntimeError(
                # Redacted: this url can be a pre-signed one, and the message lands in
                # the job error, the UI and the log.
                f"stem download failed for '{name}': HTTP {sr.status_code} from "
                f"{_redact_url(final_url)}"
            )
        # Trust the URL's own extension: roformer emits .flac, demucs .wav.
        # (Hardcoding .wav mislabels flac stems.) Use the FINAL url - a redirect is
        # what actually names the file. Strip BOTH the query and any fragment first -
        # ".flac#frag" would otherwise not match _AUDIO_EXTS and fall back to .wav.
        clean = str(final_url).split("?", 1)[0].split("#", 1)[0]
        suffix = Path(clean).suffix.lower()
        ext = suffix if suffix in _AUDIO_EXTS else ".wav"
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
    """Map our short model id to an audio-separator checkpoint filename.

    MUST match the demucs server's ``ROFORMER_MODELS`` mapping so the local
    audio-separator engine loads the same checkpoint and produces the same
    6-stem output as the remote server (the plugin's stated local-parity goal) —
    not a different stock 2-stem checkpoint.
    """
    known = {
        "bs_roformer_sw": "BS-Roformer-SW.ckpt",  # 6-stem, matches slopsmith-demucs-server
    }
    return known.get(model, model if model.endswith((".ckpt", ".onnx", ".pth")) else f"{model}.ckpt")


def _run_demucs_local(mix: Path, out_dir: Path, model: str, engine_dir: str | None,
                      models_dir: str | None = None, progress_cb: ProgressCB = None,
                      cancel_cb: CancelCB = None) -> Path:
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
    # Pin demucs' downloaded model weights inside the plugin-managed models dir
    # (via TORCH_HOME) so engine_status accounts for them and Uninstall reclaims
    # them — otherwise they land in ~/.cache/torch and leak past an uninstall.
    env = dict(os.environ)
    if models_dir:
        # Assign, don't setdefault: the packaged app already exports TORCH_HOME for
        # itself, so setdefault would be a no-op there and demucs' weights would land
        # in the app's cache — where engine_status can't see them and Uninstall can't
        # reclaim them. This env is the subprocess's only, so the app is unaffected.
        env["TORCH_HOME"] = models_dir
    # Subprocess keeps torch out of the server process and isolates crashes.
    # Popen + poll (not subprocess.run) so a cancel can terminate it mid-run;
    # output drains to a temp file to avoid a full-PIPE deadlock.
    # NB: -m demucs alone would fail — the child can't see the pip --target engine
    # tree (see _module_cmd).
    cmd = _module_cmd(engine_dir, "demucs",
                      ["-n", model, "-o", str(result_dir), str(mix)])
    with tempfile.TemporaryFile() as logf:  # binary — child writes raw bytes to the fd
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            while proc.poll() is None:
                if cancel_cb:
                    try:
                        cancel_cb()
                    except BaseException:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                            proc.wait()  # reap so we don't leave a zombie
                        raise
                time.sleep(0.3)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()  # reap
        if proc.returncode != 0:
            logf.seek(0, os.SEEK_END)          # read only the last ~400 bytes,
            logf.seek(max(0, logf.tell() - 400))  # not the whole log, on the error path
            tail = logf.read().decode("utf-8", "replace")
            raise RuntimeError(f"demucs failed: {tail}")
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


def _merge_stem_entries(existing: list[dict], produced: list[dict],
                        replace_stems: set[str] | None) -> list[dict]:
    """Merge freshly separated stem entries into a pak's existing stems list.

    ``replace_stems=None`` replaces everything the engine produced (the original
    behaviour). With a set, only those ids take the new entry; every other
    existing entry — a stem the user deliberately replaced, or a custom stem the
    engine knows nothing about ("click", "backing", …) — is preserved VERBATIM:
    same file reference, same default flag, untouched bytes. Custom ids outside
    the produced set are preserved in both modes; the old wholesale replacement
    silently delisted them, which is exactly the clobbering issue #11 is about.

    The ``full`` fallback entry is never dropped — not by special-case
    re-appending, but by construction: the merge starts from every existing
    entry and only ever overwrites ids that were selected, so an existing
    ``full`` (like any other unselected entry) survives untouched.
    """
    by_id: dict[str, dict] = {}
    for e in existing:
        sid = str(e.get("id") or "")
        if sid:
            by_id[sid] = dict(e)
    for pe in produced:
        sid = pe["id"]
        if replace_stems is None or sid in replace_stems or sid not in by_id:
            # New id (not previously in the pak) always lands — an unselected id
            # only protects an EXISTING stem; there is nothing to protect when
            # the pak never had one.
            by_id[sid] = pe
    merged = list(by_id.values())
    merged.sort(key=lambda s: _STEM_ORDER.index(s["id"]) if s["id"] in _STEM_ORDER else 99)
    return merged


def split_pak(pak_path: Path, *, engine: str, model: str | None = None,
              server_url: str | None = None, api_key: str | None = None,
              engine_dir: str | None = None, models_dir: str | None = None,
              stems: tuple[str, ...] = DEFAULT_STEMS,
              replace_stems: set[str] | None = None,
              progress_cb: ProgressCB = None, cancel_cb: CancelCB = None) -> list[str]:
    """Split ``pak_path`` into per-instrument stems, write them back into the pak,
    and rewrite the manifest. Returns the ids of every stem in the pak after
    the merge — replaced AND preserved entries alike (the manifest's final
    stems list), not just the files written this run.

    ``engine`` is one of ``"remote"``, ``"audio-separator"``, ``"demucs"``.

    ``replace_stems``: on a re-split, the ids whose existing stems may be
    overwritten. ``None`` replaces all engine outputs. The engines always
    separate the full set — selection is applied at write-back, so an
    unselected stem's file and manifest entry are left untouched (issue #11:
    users replace or add stems on purpose; a re-split must not clobber them).
    """
    import pak_io  # lazy — pulls in the core sloppak lib (see module docstring)

    pak_path = Path(pak_path)
    manifest = pak_io.read_manifest(pak_path)

    with tempfile.TemporaryDirectory(prefix="stemsplit_") as td:
        work = Path(td)
        mix = pak_io.extract_mix(pak_path, manifest, work)

        if engine == "remote":
            if not server_url:
                raise RuntimeError("remote engine selected but no split server configured")
            result_dir = _run_remote(mix, work, model or DEFAULT_REMOTE_MODEL,
                                     server_url, api_key, stems, progress_cb, cancel_cb)
        elif engine == "audio-separator":
            result_dir = _run_audio_separator(mix, work, model or DEFAULT_REMOTE_MODEL,
                                               models_dir, engine_dir, progress_cb)
        elif engine == "demucs":
            result_dir = _run_demucs_local(mix, work, model or DEFAULT_DEMUCS_MODEL,
                                           engine_dir, models_dir, progress_cb, cancel_cb)
        else:
            raise RuntimeError(f"unknown split engine: {engine!r}")

        stem_files = _collect_stem_files(result_dir)
        if not stem_files:
            raise RuntimeError("separation produced no stem files")

        if progress_cb:
            progress_cb(0.8, "Encoding stems")

        existing_stems = [e for e in (manifest.get("stems") or []) if isinstance(e, dict)]
        existing_ids = {str(e.get("id") or "") for e in existing_stems}

        add_files: dict[str, Path] = {}
        produced: list[dict] = []
        seen_ids: set[str] = set()
        for wav in stem_files:
            stem_id = _normalize_stem_id(wav.stem)
            if stem_id is None:
                # Unrecognized label — keep a sanitized token so no audio is lost
                # (it just won't match the v3 stem filter).
                sid = _sanitize(wav.stem)
                if "_" in sid and sid.split("_")[-1] in pak_io.ALLOWED_STEM_IDS:
                    sid = sid.split("_")[-1]
                stem_id = sid
            if stem_id in seen_ids:
                # Two outputs mapped to the same id (e.g. a 2-stem model's
                # instrumental collapsing onto "other"): keep the first.
                continue
            seen_ids.add(stem_id)
            if (replace_stems is not None and stem_id not in replace_stems
                    and stem_id in existing_ids):
                # Protected: the pak already has this stem and the user chose
                # not to replace it. Don't encode, don't write the file.
                continue
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
        if (mix_rel and "full" not in existing_ids
                and not any(s["id"] == "full" for s in produced)):
            produced.append(pak_io.stem_entry("full", mix_rel, default=False))

        # Rewrite manifest: merge the new entries over the existing list (an
        # unselected or custom stem keeps its entry verbatim), stamp provenance,
        # keep the mix.
        new_manifest = dict(manifest)
        new_manifest["stems"] = _merge_stem_entries(existing_stems, produced, replace_stems)
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

    return [s["id"] for s in new_manifest["stems"]]

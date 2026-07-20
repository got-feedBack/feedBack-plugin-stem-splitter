"""Pak read + repack helpers for the Stem Splitter plugin.

feedBack's ``lib/sloppak.py`` is read-only and there is no library API to *write*
a pak (add a stem file, add ``lyrics.json``, rewrite the manifest). This module
supplies that write side, self-contained, modelled on the repack pattern in
``lib/songmeta.py:_rewrite_zip_manifest`` (backup + temp + atomic replace, and —
crucially — preserving each surviving entry's ``compress_type`` so the already
Ogg-compressed stems aren't re-deflated).

A pak exists in two interchangeable forms (see ``lib/sloppak.py`` docstring):
a zip **file** (``.feedpak`` / legacy ``.sloppak``), or a **directory** whose
name ends in that suffix. Both are handled here.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import yaml

# Core lib is on sys.path (same as songmeta.py's ``import sloppak``).
import sloppak as sloppak_mod

_MANIFEST_NAMES = ("manifest.yaml", "manifest.yml")

# Interoperability set — the ids the v3 library filter understands
# (server.py ``_ALLOWED_STEM_IDS``). We map produced stems onto these where we can.
ALLOWED_STEM_IDS = {"full", "guitar", "bass", "drums", "vocals", "piano", "other"}


def is_zip_form(path: Path) -> bool:
    """A pak file (zip) vs a pak directory."""
    return Path(path).is_file()


def read_manifest(path: Path) -> dict:
    """Parsed manifest dict for a pak (dir or zip form)."""
    return sloppak_mod.load_manifest(Path(path)) or {}


def _manifest_member_name(names) -> str:
    for cand in _MANIFEST_NAMES:
        if cand in names:
            return cand
    return "manifest.yaml"


def find_mix_relpath(manifest: dict) -> str | None:
    """Pak-relative path of the pre-separation full mix.

    Prefers manifest ``original_audio``; else a stem entry with id ``full``;
    else the conventional ``stems/full.ogg`` (the caller verifies it exists).
    """
    orig = manifest.get("original_audio")
    if isinstance(orig, str) and orig.strip():
        return orig.strip()
    for stem in manifest.get("stems") or []:
        if isinstance(stem, dict) and str(stem.get("id", "")).lower() == "full":
            f = stem.get("file")
            if isinstance(f, str) and f.strip():
                return f.strip()
    return "stems/full.ogg"


def read_member_bytes(path: Path, relpath: str) -> bytes | None:
    """Read one pak member's bytes without unpacking the whole archive."""
    path = Path(path)
    relpath = relpath.replace("\\", "/").lstrip("/")
    if is_zip_form(path):
        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                return zf.read(relpath)
        except (KeyError, zipfile.BadZipFile, OSError):
            return None
    member = path / relpath
    if member.is_file():
        return member.read_bytes()
    return None


def extract_mix(path: Path, manifest: dict, dest_dir: Path) -> Path:
    """Write the full-mix audio to ``dest_dir`` and return its path.

    Raises ``FileNotFoundError`` if no mix can be located in the pak.
    """
    rel = find_mix_relpath(manifest)
    data = read_member_bytes(path, rel) if rel else None
    if data is None:
        raise FileNotFoundError(
            f"no full-mix audio found in pak (looked for {rel!r}); "
            "cannot split a pak that has no combined stem"
        )
    suffix = Path(rel).suffix or ".ogg"
    out = Path(dest_dir) / f"mix{suffix}"
    out.write_bytes(data)
    return out


def stem_entry(stem_id: str, file_rel: str, default: bool = True) -> dict:
    # `default` is a YAML boolean in this tree (see docs/sloppak-hand-editing.md):
    # true = the Stems-plugin fader starts un-muted. The FIRST stem listed is what
    # the base <audio> plays, so callers must not list `full` first.
    return {"id": stem_id, "file": file_rel, "default": bool(default)}


def dump_manifest(manifest: dict) -> str:
    return yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)


def repack(path: Path, *, add_files: dict[str, Path] | None = None,
           remove: set[str] | None = None, manifest: dict | None = None,
           keep_backup: bool = False) -> None:
    """Rewrite a pak, adding/replacing members, removing members, and (optionally)
    replacing the manifest.

    ``add_files`` maps pak-relative path -> local source file to insert.
    ``remove`` is a set of pak-relative paths to drop.
    ``manifest`` (if given) is dumped to manifest.yaml.
    ``keep_backup``: preserve a pre-rewrite copy even after success — the zip form's
    ``.feedpak.bak`` survives instead of being cleaned up, and the directory form writes a
    one-time ``<member>.bak`` beside each replaced file. For callers that overwrite data the
    user cannot regenerate (re-align rewrites authored lyric timings).

    Zip form: backup ``.bak`` + build ``.tmp`` + atomic ``replace`` (mirrors
    songmeta). Directory form: write/remove files in place with a one-time
    ``manifest.yaml.bak``.
    """
    path = Path(path)
    add_files = {k.replace("\\", "/").lstrip("/"): Path(v) for k, v in (add_files or {}).items()}
    remove = {r.replace("\\", "/").lstrip("/") for r in (remove or set())}

    if is_zip_form(path):
        _repack_zip(path, add_files, remove, manifest, keep_backup=keep_backup)
    else:
        _repack_dir(path, add_files, remove, manifest, keep_backup=keep_backup)


def _repack_zip(path: Path, add_files: dict[str, Path], remove: set[str], manifest: dict | None,
                *, keep_backup: bool = False) -> None:
    backup = path.with_name(path.name + ".bak")
    created_backup = False
    if not backup.exists():
        shutil.copy2(path, backup)
        created_backup = True
    out_tmp = path.with_name(path.name + ".tmp")

    replaced = set(add_files.keys())
    with zipfile.ZipFile(str(path), "r") as zin:
        names = zin.namelist()
        manifest_name = _manifest_member_name(names)
        with zipfile.ZipFile(str(out_tmp), "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                name = item.filename
                if name in _MANIFEST_NAMES and manifest is not None:
                    continue
                if name in remove or name in replaced:
                    continue
                # Preserve original compression (Ogg stays stored, not re-deflated).
                zout.writestr(item, zin.read(name))
            # New / replacement members. Ogg is already compressed → STORE it.
            for rel, src in add_files.items():
                comp = zipfile.ZIP_STORED if rel.lower().endswith(".ogg") else zipfile.ZIP_DEFLATED
                zout.writestr(rel, src.read_bytes(), compress_type=comp)
            if manifest is not None:
                zout.writestr(manifest_name, dump_manifest(manifest))
    out_tmp.replace(path)
    # The atomic replace succeeded, so the new pak is safely in place and the
    # backup is no longer needed. Removing it keeps a batch run from leaving one
    # `.feedpak.bak` per processed song on disk. (The backup only guards a crash
    # mid-repack, i.e. everything before this line.) A keep_backup caller wants
    # the opposite: the pre-rewrite copy IS the product — the undo for a
    # plausible-but-wrong rewrite — so it stays. And only a backup THIS run
    # created may be cleaned up: a later split/transcribe on the same pak must
    # not delete the undo a re-align deliberately left behind.
    if not keep_backup and created_backup:
        try:
            backup.unlink()
        except OSError:
            pass


def _repack_dir(path: Path, add_files: dict[str, Path], remove: set[str], manifest: dict | None,
                *, keep_backup: bool = False) -> None:
    for rel in remove:
        target = path / rel
        if target.is_file():
            target.unlink()
    for rel, src in add_files.items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != target.resolve():
            # Same one-time policy as manifest.yaml.bak below: the first rewrite
            # preserves the user's original; later rewrites don't clobber it.
            if keep_backup and target.is_file():
                bak = target.with_name(target.name + ".bak")
                if not bak.exists():
                    shutil.copy2(target, bak)
            shutil.copy2(src, target)
    if manifest is not None:
        mf = path / "manifest.yaml"
        if not mf.exists() and (path / "manifest.yml").exists():
            mf = path / "manifest.yml"
        if mf.exists():
            bak = mf.with_name(mf.name + ".bak")
            if not bak.exists():
                shutil.copy2(mf, bak)
        mf.write_text(dump_manifest(manifest), encoding="utf-8")

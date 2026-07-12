"""Unit tests for ``demucs_server._requirements_without_audio_separator``.

audio-separator has to be installed with ``--no-deps`` because its metadata requires
``diffq`` on non-Windows — a C-extension whose newest wheels stop at cp310, so on any
Python 3.11+ pip falls back to the sdist and needs a compiler. That's the failure a
Linux/macOS user hits, and the reason the upstream demucs-server Docker image has never
built.

But ``--no-deps`` on the standalone install only helps if audio-separator is ALSO absent
from the ``-r requirements.txt`` file, because pip resolves ``-r`` first — diffq gets
dragged in and the compile dies before the ``--no-deps`` step ever runs. (That exact
ordering is what makes the upstream Dockerfile's --no-deps lines dead code.)

So this filter is load-bearing: if it silently stops matching, the whole cross-platform
fix reverts with no other symptom on Windows, where it happens to work anyway.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import demucs_server as ds  # noqa: E402


class RequirementsFilter(unittest.TestCase):
    def _run(self, body: str) -> str:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td)
            ds.src_dir(cfg).mkdir(parents=True, exist_ok=True)
            (ds.src_dir(cfg) / "requirements.txt").write_text(body, encoding="utf-8")
            return ds._requirements_without_audio_separator(cfg).read_text(encoding="utf-8")

    def test_drops_the_audio_separator_pin(self):
        out = self._run("fastapi>=0.109.1\naudio-separator>=0.44.0\nlibrosa>=0.10.0\n")
        self.assertNotIn("audio-separator", out)
        self.assertIn("fastapi>=0.109.1", out)
        self.assertIn("librosa>=0.10.0", out)

    def test_underscore_spelling(self):
        # PyPI normalises - and _; a requirements file may use either.
        self.assertNotIn("audio_separator", self._run("audio_separator>=0.44.0\nfastapi\n"))

    def test_leading_whitespace_and_no_pin(self):
        out = self._run("  audio-separator\nfastapi\n")
        self.assertNotIn("audio-separator", out)
        self.assertIn("fastapi", out)

    def test_keeps_a_comment_mentioning_it(self):
        # Upstream's file has an explanatory comment above the pin. Dropping the
        # comment is harmless, but the regex anchors on the requirement, not the word.
        out = self._run("# audio-separator is fragile on slim\nfastapi\n")
        self.assertIn("# audio-separator is fragile on slim", out)

    def test_does_not_match_a_similarly_named_package(self):
        out = self._run("audio-separator-extras\naudio-separatorx\nfastapi\n")
        # \b after the name means a hyphenated CONTINUATION is a different project and
        # must survive. Only the exact distribution is dropped.
        self.assertIn("audio-separator-extras", out)
        self.assertIn("audio-separatorx", out)

    def test_absent_is_not_an_error(self):
        # Upstream is the proper home for this fix; when they land it, we must not break.
        out = self._run("fastapi\nlibrosa\n")
        self.assertEqual(out, "fastapi\nlibrosa\n")

    def test_unreadable_requirements_raises_clearly(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td)
            ds.src_dir(cfg).mkdir(parents=True, exist_ok=True)   # no requirements.txt
            with self.assertRaises(RuntimeError) as cm:
                ds._requirements_without_audio_separator(cfg)
            self.assertIn("requirements.txt", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

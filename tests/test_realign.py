"""Re-align: keep the words, fix the timings. (#20)

The whole value of this feature is the promise that it does NOT touch your lyrics. Everything
here defends that promise, because the failure modes all look like "it worked" until you read the
song:

* the sloppak on-disk shape is SYLLABLES, not words, carrying two suffixes (`-` joins to the next
  syllable, `+` ends a line). Send those to the aligner naively and "to-geth-er+" becomes three
  words that aren't words, and the alignment is garbage built from garbage;

* the server marks the FIRST word of a line (`new_line`); sloppak marks the LAST syllable of a
  line (`+`). Copy the flag straight across and every line breaks one word early — in every
  song, forever, and it looks like a rendering bug;

* an aligner that returns nothing must NOT be written back. Replacing a song's lyrics with an
  empty file, on a click that promised to preserve them, is the single worst thing this code
  could do.
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import realign  # noqa: E402


class SyllablesAreNotWords(unittest.TestCase):
    """`w` carries suffixes: `-` joins to the next syllable, `+` ends a line (feedpak §2.3)."""

    def test_joined_syllables_rebuild_one_word(self):
        tokens = [{"t": 0.0, "d": 0.2, "w": "to-"},
                  {"t": 0.2, "d": 0.2, "w": "geth-"},
                  {"t": 0.4, "d": 0.2, "w": "er"}]
        self.assertEqual(realign.lyrics_to_text(tokens), "together",
                         "sending 'to', 'geth', 'er' as three words aligns nonsense")

    def test_the_plus_suffix_is_a_line_break(self):
        tokens = [{"t": 0.0, "d": 0.2, "w": "hello"},
                  {"t": 0.3, "d": 0.2, "w": "world+"},
                  {"t": 1.0, "d": 0.2, "w": "second"},
                  {"t": 1.3, "d": 0.2, "w": "line"}]
        self.assertEqual(realign.lyrics_to_text(tokens), "hello world\nsecond line")

    def test_both_suffixes_at_once(self):
        # "+" and "-" can land on the same syllable: a word that both joins and ends a line.
        tokens = [{"t": 0.0, "d": 0.2, "w": "for-"},
                  {"t": 0.2, "d": 0.2, "w": "ev-"},
                  {"t": 0.4, "d": 0.2, "w": "er+"},
                  {"t": 1.0, "d": 0.2, "w": "and"}]
        self.assertEqual(realign.lyrics_to_text(tokens), "forever\nand")

    def test_junk_tokens_are_skipped_not_crashed_on(self):
        tokens = [{"t": 0.0, "d": 0.1, "w": "real"}, "nonsense", {"w": ""}, None, {"no_w": 1}]
        self.assertEqual(realign.lyrics_to_text(tokens), "real")


class TheLineBreakMovesBackOneToken(unittest.TestCase):
    """The server flags the FIRST word of a line; sloppak flags the LAST syllable of one.

    Copy the flag across as-is and every line breaks one word early — in every song."""

    def test_the_plus_lands_on_the_word_before_the_break(self):
        segments = [
            {"start": 0.0, "end": 0.4, "text": "hello", "new_line": True},
            {"start": 0.5, "end": 0.9, "text": "world"},
            {"start": 2.0, "end": 2.4, "text": "second", "new_line": True},
            {"start": 2.5, "end": 2.9, "text": "line"},
        ]
        out = realign.segments_to_lyrics(segments)
        self.assertEqual([t["w"] for t in out], ["hello", "world+", "second", "line"],
                         "the `+` belongs on the LAST word of a line, not the first")

    def test_timings_are_start_and_duration(self):
        out = realign.segments_to_lyrics([{"start": 1.25, "end": 1.75, "text": "x"}])
        self.assertEqual(out, [{"t": 1.25, "d": 0.5, "w": "x"}])

    def test_a_backwards_segment_does_not_produce_a_negative_duration(self):
        out = realign.segments_to_lyrics([{"start": 2.0, "end": 1.0, "text": "x"}])
        self.assertEqual(out[0]["d"], 0.0)

    def test_untimed_segments_are_dropped(self):
        segments = [{"text": "no times"}, {"start": 1.0, "end": 2.0, "text": "kept"}]
        self.assertEqual([t["w"] for t in realign.segments_to_lyrics(segments)], ["kept"])


class TheRoundTripPreservesTheWords(unittest.TestCase):
    def test_text_out_equals_text_in(self):
        """The point of the feature: same words, new timings."""
        original = [
            {"t": 0.0, "d": 0.3, "w": "hold"},
            {"t": 0.4, "d": 0.3, "w": "me"},
            {"t": 0.8, "d": 0.5, "w": "clos-"},
            {"t": 1.3, "d": 0.4, "w": "er+"},
            {"t": 2.0, "d": 0.4, "w": "tiny"},
            {"t": 2.5, "d": 0.6, "w": "dancer"},
        ]
        text = realign.lyrics_to_text(original)
        self.assertEqual(text, "hold me closer\ntiny dancer")

        # What the server gives back for that text, at word granularity, with new timings.
        segments = [
            {"start": 10.0, "end": 10.3, "text": "hold", "new_line": True},
            {"start": 10.4, "end": 10.6, "text": "me"},
            {"start": 10.7, "end": 11.2, "text": "closer"},
            {"start": 12.0, "end": 12.4, "text": "tiny", "new_line": True},
            {"start": 12.5, "end": 13.1, "text": "dancer"},
        ]
        out = realign.segments_to_lyrics(segments)

        self.assertEqual(realign.lyrics_to_text(out), text,
                         "a re-align must not change a single word — that is the entire promise")
        self.assertEqual(out[0]["t"], 10.0, "and the timings must actually be the new ones")


class ItRefusesRatherThanDestroys(unittest.TestCase):
    """Every one of these would otherwise be a silent way to lose the user's lyrics."""

    def _pak(self, manifest, lyrics=None):
        m = mock.patch.object(realign.pak_io, "read_manifest", return_value=manifest)
        raw = json.dumps(lyrics).encode() if lyrics is not None else None
        r = mock.patch.object(realign.pak_io, "read_member_bytes", return_value=raw)
        return m, r

    def test_no_lyrics_is_an_error_not_a_no_op(self):
        m, r = self._pak({"stems": [{"id": "vocals", "file": "stems/vocals.ogg"}]})
        with m, r, self.assertRaises(RuntimeError) as e:
            realign.realign_pak("song.sloppak", server_url="http://s")
        self.assertIn("Transcribe", str(e.exception), "tell them which button DOES apply")

    def test_no_vocal_stem_says_so_instead_of_silently_splitting(self):
        # Kicking off a multi-minute GPU split because a menu item was clicked is not what
        # anybody asked for. "Split stems" is right there.
        m, r = self._pak({"lyrics": "lyrics.json", "stems": []})
        with m, r, self.assertRaises(RuntimeError) as e:
            realign.realign_pak("song.sloppak", server_url="http://s")
        self.assertIn("Split stems", str(e.exception))

    def test_an_empty_alignment_is_never_written_back(self):
        """The worst possible bug in this file: replacing a song's lyrics with nothing, on a
        click that promised to keep them."""
        manifest = {"lyrics": "lyrics.json",
                    "stems": [{"id": "vocals", "file": "stems/vocals.ogg"}]}
        m, r = self._pak(manifest, lyrics=[{"t": 0, "d": 1, "w": "word"}])
        with m, r, \
             mock.patch.object(realign, "align_vocals_remote", return_value=[]), \
             mock.patch.object(realign.pak_io, "repack") as repack:
            with self.assertRaises(RuntimeError) as e:
                realign.realign_pak("song.sloppak", server_url="http://s")
        repack.assert_not_called()
        self.assertIn("no timings", str(e.exception))

    def test_the_manifest_is_not_rewritten(self):
        """`lyrics_source` is a closed vocabulary in the feedpak spec (authored | transcribed |
        user), and re-aligning does not change where the words CAME from. Authored lyrics stay
        authored: a pak whose timings were repaired is not a pak whose provenance changed."""
        manifest = {"lyrics": "lyrics.json", "lyrics_source": "authored",
                    "stems": [{"id": "vocals", "file": "stems/vocals.ogg"}]}
        m, r = self._pak(manifest, lyrics=[{"t": 0, "d": 1, "w": "word"}])
        aligned = [{"t": 5.0, "d": 0.5, "w": "word"}]
        with m, r, \
             mock.patch.object(realign, "align_vocals_remote", return_value=aligned), \
             mock.patch.object(realign.pak_io, "repack") as repack:
            self.assertTrue(realign.realign_pak("song.sloppak", server_url="http://s"))

        _, kwargs = repack.call_args
        self.assertIsNone(kwargs.get("manifest"),
                          "re-align must not rewrite the manifest — it changes timings, not "
                          "provenance, and lyrics_source is the spec's to define, not ours")
        self.assertIn("lyrics.json", kwargs["add_files"])

    def test_a_server_is_required(self):
        manifest = {"lyrics": "lyrics.json",
                    "stems": [{"id": "vocals", "file": "stems/vocals.ogg"}]}
        m, r = self._pak(manifest, lyrics=[{"t": 0, "d": 1, "w": "w"}])
        with m, r, self.assertRaises(RuntimeError) as e:
            realign.realign_pak("song.sloppak", server_url=None)
        self.assertIn("server", str(e.exception))


class TheRequestIsTheContract(unittest.TestCase):
    """#17 was a request bug that no mapper test could see. Same lesson, applied up front."""

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"segments": [{"start": 1.0, "end": 2.0, "text": "hi", "new_line": True}]}

    def test_it_posts_the_text_and_asks_for_word_granularity(self, ):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            vocals = Path(td) / "vocals.ogg"
            vocals.write_bytes(b"audio")
            with mock.patch("requests.post", return_value=self._Resp()) as post:
                realign.align_vocals_remote(vocals, "hello world", "http://s:7865", language="en")

        url = post.call_args.args[0]
        form = post.call_args.kwargs["data"]
        self.assertTrue(url.endswith("/align"), "re-align is FORCED ALIGNMENT — it has the words")
        self.assertEqual(form["text"], "hello world", "the lyrics must actually be sent")
        self.assertEqual(form["granularity"], "word")
        self.assertEqual(form["language"], "en")
        # Everything the server reads is Form(...); a query param is silently ignored. That
        # mistake, in the other direction, is what made transcription 422 for months (#17).
        self.assertFalse(post.call_args.kwargs.get("params"))


if __name__ == "__main__":
    unittest.main()

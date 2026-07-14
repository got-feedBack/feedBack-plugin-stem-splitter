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
from pathlib import Path
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


class TheLyricsAreRebuiltFromTheUsersOwnTokens(unittest.TestCase):
    """The output is the user's tokens with new numbers. It is never the server's tokens.

    The earlier version built the new lyrics out of the words the SERVER returned — and forced
    alignment legitimately drops a word it can't place, so every dropped word was silently deleted
    from the song. A re-align that quietly loses "dancer" from the chorus hasn't kept the promise;
    it has broken it in the way nobody notices until they play the song. (Caught by CodeRabbit.)
    """

    def test_a_word_the_aligner_dropped_is_still_in_the_lyrics(self):
        tokens = [{"t": 0, "d": 1, "w": "hold"}, {"t": 1, "d": 1, "w": "me"},
                  {"t": 2, "d": 1, "w": "closer"}]
        aligned = [{"t": 10.0, "d": 0.4, "w": "hold"},      # "me" was not placed
                   {"t": 11.0, "d": 0.6, "w": "closer"}]
        out = realign.retime_tokens(tokens, aligned)

        self.assertEqual([x["w"] for x in out], ["hold", "me", "closer"],
                         "the dropped word must survive — losing it is the bug this guards")
        self.assertEqual(out[0]["t"], 10.0)
        self.assertEqual(out[2]["t"], 11.0)
        # ...and it lands between its neighbours rather than at zero.
        self.assertGreaterEqual(out[1]["t"], out[0]["t"])
        self.assertLessEqual(out[1]["t"], out[2]["t"])

    def test_the_authored_line_structure_survives(self):
        # The `+` breaks are the user's. The aligner's idea of where a line ends is irrelevant.
        tokens = [{"t": 0, "d": 1, "w": "hold"}, {"t": 1, "d": 1, "w": "me+"},
                  {"t": 2, "d": 1, "w": "closer"}]
        aligned = [{"t": 5.0, "d": 0.3, "w": "hold"}, {"t": 5.4, "d": 0.3, "w": "me"},
                   {"t": 6.0, "d": 0.5, "w": "closer"}]
        out = realign.retime_tokens(tokens, aligned)
        self.assertEqual([x["w"] for x in out], ["hold", "me+", "closer"])

    def test_a_word_split_into_syllables_shares_its_span(self):
        tokens = [{"t": 0, "d": 1, "w": "to-"}, {"t": 1, "d": 1, "w": "geth-"},
                  {"t": 2, "d": 1, "w": "er"}]
        aligned = [{"t": 4.0, "d": 0.9, "w": "together"}]
        out = realign.retime_tokens(tokens, aligned)

        self.assertEqual([x["w"] for x in out], ["to-", "geth-", "er"], "syllables are preserved")
        self.assertAlmostEqual(out[0]["t"], 4.0, places=2)
        # contiguous, and covering the aligned word's span
        self.assertAlmostEqual(out[2]["t"] + out[2]["d"], 4.9, places=1)
        for a, b in zip(out, out[1:]):
            self.assertAlmostEqual(a["t"] + a["d"], b["t"], places=2,
                                   msg="syllables of one word must not overlap or leave gaps")

    def test_timings_are_the_new_ones(self):
        tokens = [{"t": 99.0, "d": 9.0, "w": "hello"}]
        out = realign.retime_tokens(tokens, [{"t": 1.0, "d": 0.5, "w": "hello"}])
        self.assertEqual(out[0], {"t": 1.0, "d": 0.5, "w": "hello"})

    def test_nothing_matching_at_all_is_refused(self):
        with self.assertRaises(RuntimeError):
            realign.retime_tokens([{"t": 0, "d": 1, "w": "hello"}],
                                  [{"t": 1, "d": 1, "w": "goodbye"}])

    def test_junk_in_the_lyrics_does_not_shift_the_timings_onto_the_wrong_words(self):
        """lyrics_to_text() already tolerates junk entries, so a pak can contain them.

        The token indices are recorded against the ORIGINAL list. Filter the junk out before
        applying them and every index after it shifts by one — the timings land on the wrong
        words, and the result is a corrupted chart that still LOOKS like a chart."""
        tokens = [{"t": 0, "d": 1, "w": "hold"},
                  "junk",                                  # not a dict
                  {"t": 1, "d": 1, "w": "me"},
                  None,
                  {"t": 2, "d": 1, "w": "closer"}]
        aligned = [{"t": 10.0, "d": 0.4, "w": "hold"},
                   {"t": 11.0, "d": 0.3, "w": "me"},
                   {"t": 12.0, "d": 0.6, "w": "closer"}]
        out = realign.retime_tokens(tokens, aligned)

        self.assertEqual([x["w"] for x in out], ["hold", "me", "closer"],
                         "the junk is dropped from the output")
        self.assertEqual([x["t"] for x in out], [10.0, 11.0, 12.0],
                         "and every word keeps ITS OWN timing — not its neighbour's")

    def test_wordless_dict_tokens_are_not_written_back(self):
        """`{}` and `{"w": ""}` are not lyrics. lyrics_to_text() already ignores them, so a pak
        can be carrying them — and passing them through would re-emit invalid tokens, untimed, as
        if we had produced them."""
        tokens = [{"t": 0, "d": 1, "w": "hold"},
                  {},                                   # dict junk
                  {"w": ""},                            # ...and more of it
                  {"t": 2, "d": 1, "w": "closer"}]
        out = realign.retime_tokens(tokens, [{"t": 5.0, "d": 0.3, "w": "hold"},
                                             {"t": 6.0, "d": 0.4, "w": "closer"}])
        self.assertEqual([x["w"] for x in out], ["hold", "closer"])
        self.assertTrue(all(str(x.get("w") or "").strip() for x in out),
                        "no wordless token may reach the pak")

    def test_segments_to_words_drops_untimed_and_keeps_timings(self):
        out = realign.segments_to_words([
            {"text": "no times"},
            {"start": 1.25, "end": 1.75, "text": "kept"},
            {"start": 2.0, "end": 1.0, "text": "backwards"},   # must not go negative
        ])
        self.assertEqual([w["w"] for w in out], ["kept", "backwards"])
        self.assertEqual(out[0], {"t": 1.25, "d": 0.5, "w": "kept"})
        self.assertEqual(out[1]["d"], 0.0)


class TheWordsAreVerifiedNotAssumed(unittest.TestCase):
    """The feature's promise is "your words are safe". Checked, not trusted.

    Forced alignment hands back OUR text with timings — so if different words come back, something
    is wrong (a misbehaving server, a proxy, an endpoint that isn't /align), and we are one repack
    away from overwriting the user's lyrics with them."""

    def _aligned(self, words):
        return [{"t": float(i), "d": 0.5, "w": w} for i, w in enumerate(words)]

    def test_the_same_words_pass(self):
        realign._verify_words_survived("hold me closer", self._aligned(["hold", "me", "closer"]))

    def test_punctuation_and_case_are_not_changes(self):
        # "Closer," for "closer" is not a changed lyric. Refusing over it would fire on every song.
        realign._verify_words_survived("hold me closer", self._aligned(["Hold", "me", "Closer,"]))

    def test_line_and_join_suffixes_are_ignored(self):
        realign._verify_words_survived("hold me closer", self._aligned(["hold", "me+", "closer"]))

    def test_a_dropped_word_is_tolerated(self):
        # The aligner legitimately fails to place the odd word (a shout, a word under a cymbal).
        realign._verify_words_survived(
            "hold me closer tiny dancer", self._aligned(["hold", "me", "closer", "dancer"]))

    def test_an_invented_word_is_refused(self):
        """The one that matters: a server returning words we never sent is not aligning, and
        writing that back would destroy the lyrics this feature exists to protect."""
        with self.assertRaises(RuntimeError) as e:
            realign._verify_words_survived(
                "hold me closer tiny dancer",
                self._aligned(["hold", "me", "closer", "tony", "danza"]))
        self.assertIn("not in your lyrics", str(e.exception))
        self.assertIn("Transcribe", str(e.exception), "point at the button that DOES replace words")

    def test_reordered_words_are_refused(self):
        with self.assertRaises(RuntimeError):
            realign._verify_words_survived(
                "hold me closer", self._aligned(["closer", "me", "hold"]))

    def test_a_mostly_empty_alignment_is_refused(self):
        # Two words out of ten is not an alignment; writing it back would gut the song while
        # reporting success.
        original = "one two three four five six seven eight nine ten"
        with self.assertRaises(RuntimeError) as e:
            realign._verify_words_survived(original, self._aligned(["one", "ten"]))
        self.assertIn("only placed", str(e.exception))

    def test_nothing_at_all_is_refused(self):
        with self.assertRaises(RuntimeError):
            realign._verify_words_survived("hold me closer", [])

    def test_the_guard_runs_before_the_repack(self):
        """A guard that fires after the write is not a guard."""
        manifest = {"lyrics": "lyrics.json",
                    "stems": [{"id": "vocals", "file": "stems/vocals.ogg"}]}
        lyrics = json.dumps([{"t": 0, "d": 1, "w": "hello"}]).encode()
        with mock.patch.object(
                realign.pak_io, "read_manifest", return_value=manifest),              mock.patch.object(
                realign.pak_io, "read_member_bytes", return_value=lyrics),              mock.patch.object(
                realign, "align_vocals_remote",
                return_value=[{"t": 0.0, "d": 1.0, "w": "goodbye"}]),              mock.patch.object(
                realign.pak_io, "repack") as repack:
            with self.assertRaises(RuntimeError):
                realign.realign_pak("song.sloppak", server_url="http://s")
        repack.assert_not_called()


class WhatLandsInThePakKeepsEveryWord(unittest.TestCase):
    """End to end, through realign_pak: read the bytes that would actually be written.

    The unit tests prove retime_tokens preserves words. This proves the pak gets THAT, and not
    something else — I checked, and without it the whole feature could be rewired to write the
    server's words back and every unit test would still pass."""

    def test_the_written_lyrics_still_contain_the_dropped_word(self):
        tokens = [{"t": 0, "d": 1, "w": "hold"},
                  {"t": 1, "d": 1, "w": "me"},
                  {"t": 2, "d": 1, "w": "closer"}]
        manifest = {"lyrics": "lyrics.json",
                    "stems": [{"id": "vocals", "file": "stems/vocals.ogg"}]}
        written = {}

        def capture(pak, *, add_files=None, **kw):
            path = (add_files or {})["lyrics.json"]
            written["lyrics"] = json.loads(Path(path).read_text(encoding="utf-8"))

        with mock.patch.object(realign.pak_io, "read_manifest", return_value=manifest),              mock.patch.object(realign.pak_io, "read_member_bytes",
                               side_effect=[json.dumps(tokens).encode(), b"audio"]),              mock.patch.object(realign, "align_vocals_remote",
                               return_value=[{"t": 10.0, "d": 0.4, "w": "hold"},
                                             {"t": 11.0, "d": 0.6, "w": "closer"}]),              mock.patch.object(realign.pak_io, "repack", side_effect=capture):
            self.assertTrue(realign.realign_pak("song.sloppak", server_url="http://s"))

        got = [tok["w"] for tok in written["lyrics"]]
        self.assertEqual(got, ["hold", "me", "closer"],
                         "the aligner dropped 'me'; the pak must NOT lose it")
        self.assertEqual(written["lyrics"][0]["t"], 10.0, "and the timings must be the new ones")


class TheUploadDescribesTheFileItActuallyIs(unittest.TestCase):
    """A pak from another tool can carry a .wav or .flac vocals stem. Telling the server (or a
    proxy in front of it) that a wav is an ogg is how an upload gets rejected or mis-decoded."""

    def test_content_type_follows_the_suffix(self):
        from pathlib import Path as P
        self.assertEqual(realign._content_type(P("v.wav")), "audio/wav")
        self.assertEqual(realign._content_type(P("v.flac")), "audio/flac")
        self.assertEqual(realign._content_type(P("v.ogg")), "audio/ogg")
        self.assertEqual(realign._content_type(P("v.weird")), "application/octet-stream")


class TheRoundTripPreservesTheWords(unittest.TestCase):
    def test_the_song_comes_back_word_for_word(self):
        """The point of the feature, end to end: same words, same lines, new timings."""
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

        # What the server gives back for that text — with "tiny" left unplaced, as happens.
        aligned = realign.segments_to_words([
            {"start": 10.0, "end": 10.3, "text": "hold", "new_line": True},
            {"start": 10.4, "end": 10.6, "text": "me"},
            {"start": 10.7, "end": 11.2, "text": "closer"},
            {"start": 12.5, "end": 13.1, "text": "dancer", "new_line": True},
        ])
        realign._verify_words_survived(text, aligned)
        out = realign.retime_tokens(original, aligned)

        self.assertEqual(realign.lyrics_to_text(out), text,
                         "a re-align must not change a single word — that is the entire promise")
        self.assertEqual([x["w"] for x in out], [x["w"] for x in original],
                         "including the syllable splits and the line breaks the user authored")
        self.assertEqual(out[0]["t"], 10.0, "and the timings must actually be the new ones")
        self.assertNotEqual(out[4]["t"], original[4]["t"],
                            "even the word the aligner dropped gets a new, plausible time")


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

        def close(self):
            pass

    def test_it_posts_the_text_and_asks_for_word_granularity(self):
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


class TheResponseIsHandledLikeAServerCanMisbehave(unittest.TestCase):
    class _Resp:
        def __init__(self, status=200, payload=None, text="", bad_json=False):
            self.status_code, self._payload, self.text = status, payload, text
            self._bad_json, self.closed = bad_json, False

        def json(self):
            if self._bad_json:
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
            return self._payload

        def close(self):
            self.closed = True

    def _call(self, resp):
        import tempfile
        from pathlib import Path as P
        with tempfile.TemporaryDirectory() as td:
            v = P(td) / "vocals.ogg"
            v.write_bytes(b"a")
            with mock.patch("requests.post", return_value=resp):
                return realign.align_vocals_remote(v, "hi", "http://s")

    def test_the_response_is_closed_even_when_it_fails(self):
        # Raising with the response open holds a pooled connection until GC — once per song in a
        # batch re-align.
        resp = self._Resp(status=500, text="boom")
        with self.assertRaises(RuntimeError):
            self._call(resp)
        self.assertTrue(resp.closed, "the connection must go back to the pool")

    def test_the_response_is_closed_on_success(self):
        resp = self._Resp(payload={"segments": [{"start": 0, "end": 1, "text": "hi"}]})
        self._call(resp)
        self.assertTrue(resp.closed)

    def test_a_200_that_is_not_json_says_what_the_server_actually_sent(self):
        """A proxy's HTML error page with a 200 on it. resp.json() alone raises a decode error
        naming a byte offset, which tells the user precisely nothing."""
        resp = self._Resp(bad_json=True, text="<html><body>502 Bad Gateway</body></html>")
        with self.assertRaises(RuntimeError) as e:
            self._call(resp)
        self.assertIn("502 Bad Gateway", str(e.exception))
        self.assertTrue(resp.closed)

    def test_a_json_body_without_segments_is_an_error(self):
        resp = self._Resp(payload={"error": "no aligner for 'xx'"})
        with self.assertRaises(RuntimeError) as e:
            self._call(resp)
        self.assertIn("no segments", str(e.exception))


if __name__ == "__main__":
    unittest.main()

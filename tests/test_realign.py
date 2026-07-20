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
from contextlib import ExitStack
from pathlib import Path
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
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                realign.pak_io, "read_manifest", return_value=manifest))
            stack.enter_context(mock.patch.object(
                realign.pak_io, "read_member_bytes", return_value=lyrics))
            stack.enter_context(mock.patch.object(
                realign, "align_vocals_remote",
                return_value=[{"t": 0.0, "d": 1.0, "w": "goodbye"}]))
            repack = stack.enter_context(mock.patch.object(realign.pak_io, "repack"))

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

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                realign.pak_io, "read_manifest", return_value=manifest))
            stack.enter_context(mock.patch.object(
                realign.pak_io, "read_member_bytes",
                side_effect=[json.dumps(tokens).encode(), b"audio"]))
            stack.enter_context(mock.patch.object(
                realign, "align_vocals_remote",
                return_value=[{"t": 10.0, "d": 0.4, "w": "hold"},
                              {"t": 11.0, "d": 0.6, "w": "closer"}]))
            stack.enter_context(mock.patch.object(
                realign.pak_io, "repack", side_effect=capture))

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


class TheTimingsAreVerifiedNotAssumed(unittest.TestCase):
    """#27: the words guard passes trivially under forced alignment (the server returns OUR
    text), so the only thing realign changes — the timings — went entirely unguarded. A
    chant intro / backing-vocal bleed pinned words onto the wrong vocal content, the result
    scattered −72s..+3.9s, and it was written out as success. These defend the numbers the
    way the tests above defend the words."""

    def _tokens(self, n=10, start=10.0, step=1.0):
        return [{"t": start + i * step, "d": 0.4, "w": f"word{i}"} for i in range(n)]

    def _aligned(self, times):
        return [{"t": float(t), "d": 0.4, "w": f"word{i}"} for i, t in enumerate(times)]

    def test_a_scattered_alignment_is_refused(self):
        # In-order words (subsequence guard passes) whose deltas scatter by tens of seconds —
        # the #27 shape. A real fix is one shift plus sub-second drift; this is neither.
        tokens = self._tokens()                            # 10..19s
        aligned = self._aligned([i * 8.0 for i in range(10)])  # 0..72s, deltas -10..+53
        with self.assertRaises(RuntimeError) as e:
            realign._verify_timings_sane(tokens, aligned)
        self.assertIn("implausible", str(e.exception))
        self.assertIn("left unchanged", str(e.exception),
                      "the message must say the pak was not touched")

    def test_a_uniform_shift_is_allowed_at_any_size(self):
        # A chart that is legitimately 10s off must be fixable in one click: gate the SPREAD,
        # never the shift.
        tokens = self._tokens()
        aligned = self._aligned([10.0 + i * 1.0 + 10.0 for i in range(10)])  # exactly +10s
        stats = realign._verify_timings_sane(tokens, aligned)
        self.assertAlmostEqual(stats["median_shift_s"], 10.0, places=2)
        self.assertLessEqual(stats["spread_s"], 0.01)
        self.assertEqual(stats["anchored"], 10)

    def test_sub_second_drift_around_a_shift_is_allowed(self):
        tokens = self._tokens()
        aligned = self._aligned([10.0 + i * 1.0 + 2.0 + (0.05 * (i % 3)) for i in range(10)])
        stats = realign._verify_timings_sane(tokens, aligned)
        self.assertLess(stats["spread_s"], 0.5)

    def test_words_running_backwards_are_refused(self):
        # Two words pinned to different occurrences of similar audio: starts go backwards even
        # though every delta is individually modest.
        tokens = [{"t": 10.0, "d": 0.4, "w": "hold"},
                  {"t": 11.0, "d": 0.4, "w": "me"},
                  {"t": 12.0, "d": 0.4, "w": "closer"}]
        aligned = [{"t": 15.0, "d": 0.4, "w": "hold"},
                   {"t": 16.0, "d": 0.4, "w": "me"},
                   {"t": 14.9, "d": 0.4, "w": "closer"}]
        with self.assertRaises(RuntimeError) as e:
            realign._verify_timings_sane(tokens, aligned)
        self.assertIn("out-of-order", str(e.exception))

    def test_the_threshold_is_tunable(self):
        tokens = self._tokens()
        aligned = self._aligned([10.0 + i * 1.0 + (0.5 * i) for i in range(10)])  # 4.5s spread
        with self.assertRaises(RuntimeError):
            realign._verify_timings_sane(tokens, aligned)                # default 3.0
        realign._verify_timings_sane(tokens, aligned, max_spread_sec=6.0)  # loosened

    def test_a_broken_threshold_cannot_disable_the_guard(self):
        # NaN makes every `>` comparison false — the gate would silently accept anything.
        # This is the last line of defense before the write, so a corrupted setting must
        # fall back to the shipped default, not switch the guard off (or, for a negative
        # value, reject every legitimate result).
        tokens = self._tokens()
        scattered = self._aligned([i * 8.0 for i in range(10)])
        for bad in (float("nan"), float("inf"), -1.0, "junk", None):
            with self.assertRaises(RuntimeError, msg=f"max_spread_sec={bad!r}"):
                realign._verify_timings_sane(tokens, scattered, max_spread_sec=bad)
        # ...and the default must still ACCEPT a sane result under the same bad inputs.
        sane = self._aligned([10.0 + i * 1.0 + 5.0 for i in range(10)])
        for bad in (float("nan"), -1.0):
            stats = realign._verify_timings_sane(tokens, sane, max_spread_sec=bad)
            self.assertEqual(stats["anchored"], 10)

    def test_the_timing_guard_runs_before_the_repack(self):
        """Same rule as the words guard: a guard that fires after the write is not a guard."""
        tokens = [{"t": 10.0 + i, "d": 0.4, "w": f"word{i}"} for i in range(10)]
        manifest = {"lyrics": "lyrics.json",
                    "stems": [{"id": "vocals", "file": "stems/vocals.ogg"}]}
        scattered = [{"t": float(i * 8), "d": 0.4, "w": f"word{i}"} for i in range(10)]
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                realign.pak_io, "read_manifest", return_value=manifest))
            stack.enter_context(mock.patch.object(
                realign.pak_io, "read_member_bytes",
                side_effect=[json.dumps(tokens).encode(), b"audio"]))
            stack.enter_context(mock.patch.object(
                realign, "align_vocals_remote", return_value=scattered))
            repack = stack.enter_context(mock.patch.object(realign.pak_io, "repack"))

            with self.assertRaises(RuntimeError) as e:
                realign.realign_pak("song.sloppak", server_url="http://s")
        self.assertIn("implausible", str(e.exception))
        repack.assert_not_called()

    def test_a_sane_realign_reports_its_stats(self):
        tokens = [{"t": 0.0, "d": 0.4, "w": "hold"},
                  {"t": 1.0, "d": 0.4, "w": "me"},
                  {"t": 2.0, "d": 0.4, "w": "closer"}]
        manifest = {"lyrics": "lyrics.json",
                    "stems": [{"id": "vocals", "file": "stems/vocals.ogg"}]}
        aligned = [{"t": 10.0, "d": 0.4, "w": "hold"},
                   {"t": 12.0, "d": 0.4, "w": "closer"}]   # "me" interpolated
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                realign.pak_io, "read_manifest", return_value=manifest))
            stack.enter_context(mock.patch.object(
                realign.pak_io, "read_member_bytes",
                side_effect=[json.dumps(tokens).encode(), b"audio"]))
            stack.enter_context(mock.patch.object(
                realign, "align_vocals_remote", return_value=aligned))
            stack.enter_context(mock.patch.object(realign.pak_io, "repack"))

            stats = realign.realign_pak("song.sloppak", server_url="http://s")
        self.assertEqual(stats["words"], 3)
        self.assertEqual(stats["anchored"], 2)
        self.assertEqual(stats["interpolated"], 1)
        self.assertAlmostEqual(stats["median_shift_s"], 10.0, places=2)
        self.assertIn("backup", stats)


class LowConfidencePlacementsAreNotTrusted(unittest.TestCase):
    """WhisperX scores every placed word. A low score IS the poisoned anchor from #27 — the
    word pinned to a chant instead of the sung line. The transcribe path already drops those
    (lib/lyrics_transcribe.py); realign now does the same, and the dropped word gets
    interpolated between anchors that were actually trusted."""

    def test_low_score_words_are_dropped_when_scores_are_present(self):
        words = realign.segments_to_words([
            {"start": 10.0, "end": 10.4, "text": "hold", "score": 0.9},
            {"start": 55.0, "end": 55.4, "text": "me", "score": 0.05},
            {"start": 11.0, "end": 11.4, "text": "closer", "score": 0.9},
        ], min_score=0.35)
        self.assertEqual([w["w"] for w in words], ["hold", "closer"])

    def test_a_scoreless_response_is_accepted_whole(self):
        # An older server that doesn't send scores must keep working.
        words = realign.segments_to_words([
            {"start": 10.0, "end": 10.4, "text": "hold"},
            {"start": 11.0, "end": 11.4, "text": "me"},
        ], min_score=0.35)
        self.assertEqual(len(words), 2)

    def test_a_missing_score_in_a_scored_response_is_untrusted(self):
        # Mirrors the transcribe path: when the server DOES score, an unscored word is the
        # aligner declining to vouch for it.
        words = realign.segments_to_words([
            {"start": 10.0, "end": 10.4, "text": "hold", "score": 0.9},
            {"start": 55.0, "end": 55.4, "text": "me"},
        ], min_score=0.35)
        self.assertEqual([w["w"] for w in words], ["hold"])

    def test_no_threshold_means_no_filtering(self):
        words = realign.segments_to_words([
            {"start": 10.0, "end": 10.4, "text": "hold", "score": 0.01},
        ])
        self.assertEqual(len(words), 1)

    def test_the_poisoned_anchor_lands_by_interpolation_not_by_its_bad_time(self):
        tokens = [{"t": 0.0, "d": 0.4, "w": "hold"},
                  {"t": 1.0, "d": 0.4, "w": "me"},
                  {"t": 2.0, "d": 0.4, "w": "closer"}]
        aligned = realign.segments_to_words([
            {"start": 10.0, "end": 10.4, "text": "hold", "score": 0.9},
            {"start": 55.0, "end": 55.4, "text": "me", "score": 0.05},   # pinned to a chant
            {"start": 11.0, "end": 11.4, "text": "closer", "score": 0.9},
        ], min_score=0.35)
        out = realign.retime_tokens(tokens, aligned)
        self.assertEqual([x["w"] for x in out], ["hold", "me", "closer"])
        self.assertGreaterEqual(out[1]["t"], 10.0, "'me' must land between its neighbours")
        self.assertLessEqual(out[1]["t"] + out[1]["d"], 11.0 + 0.01,
                             "…not at the chant's 55s")


class TheBackupSurvivesARealign(unittest.TestCase):
    """#27's second failure: the pak was rewritten in place and the pre-realign copy deleted
    on success, so a plausible-but-wrong result destroyed the only copy of the authored
    timings. keep_backup preserves the pre-rewrite state past a successful repack."""

    def _zip_pak(self, td):
        import zipfile
        pak = Path(td) / "song.feedpak"
        with zipfile.ZipFile(pak, "w") as z:
            z.writestr("manifest.yaml", "lyrics: lyrics.json\n")
            z.writestr("lyrics.json", '[{"t":0,"d":1,"w":"orig"}]')
        return pak

    def test_zip_form_keeps_the_bak_when_asked(self):
        import tempfile
        import pak_io
        with tempfile.TemporaryDirectory() as td:
            pak = self._zip_pak(td)
            original = pak.read_bytes()
            new = Path(td) / "new_lyrics.json"
            new.write_text('[{"t":9,"d":1,"w":"new"}]', encoding="utf-8")
            pak_io.repack(pak, add_files={"lyrics.json": new}, keep_backup=True)
            bak = pak.with_name(pak.name + ".bak")
            self.assertTrue(bak.exists(), "the pre-realign pak is the undo — it must survive")
            self.assertEqual(bak.read_bytes(), original)

    def test_zip_form_still_cleans_up_by_default(self):
        # Split/transcribe batches must not litter one .bak per processed song.
        import tempfile
        import pak_io
        with tempfile.TemporaryDirectory() as td:
            pak = self._zip_pak(td)
            new = Path(td) / "new_lyrics.json"
            new.write_text("[]", encoding="utf-8")
            pak_io.repack(pak, add_files={"lyrics.json": new})
            self.assertFalse(pak.with_name(pak.name + ".bak").exists())

    def test_a_later_default_repack_does_not_delete_a_kept_backup(self):
        # The undo a re-align deliberately left behind must survive a later
        # split/transcribe on the same pak, whose repack cleans up ITS OWN backup only.
        import tempfile
        import pak_io
        with tempfile.TemporaryDirectory() as td:
            pak = self._zip_pak(td)
            original = pak.read_bytes()
            new = Path(td) / "new_lyrics.json"
            new.write_text('[{"t":9,"d":1,"w":"new"}]', encoding="utf-8")
            pak_io.repack(pak, add_files={"lyrics.json": new}, keep_backup=True)
            bak = pak.with_name(pak.name + ".bak")
            self.assertTrue(bak.exists())

            newer = Path(td) / "newer.json"
            newer.write_text('[{"t":5,"d":1,"w":"newer"}]', encoding="utf-8")
            pak_io.repack(pak, add_files={"lyrics.json": newer})  # e.g. a split job
            self.assertTrue(bak.exists(), "the realign undo must survive later repacks")
            self.assertEqual(bak.read_bytes(), original,
                             "…and still hold the pre-realign pak")

    def test_dir_form_backs_up_the_replaced_member(self):
        import tempfile
        import pak_io
        with tempfile.TemporaryDirectory() as td:
            pak = Path(td) / "song.feedpak"
            pak.mkdir()
            (pak / "manifest.yaml").write_text("lyrics: lyrics.json\n", encoding="utf-8")
            (pak / "lyrics.json").write_text('[{"t":0,"d":1,"w":"orig"}]', encoding="utf-8")
            new = Path(td) / "new_lyrics.json"
            new.write_text('[{"t":9,"d":1,"w":"new"}]', encoding="utf-8")
            pak_io.repack(pak, add_files={"lyrics.json": new}, keep_backup=True)
            bak = pak / "lyrics.json.bak"
            self.assertTrue(bak.exists())
            self.assertIn("orig", bak.read_text(encoding="utf-8"))
            self.assertIn("new", (pak / "lyrics.json").read_text(encoding="utf-8"))
            # A second rewrite must not clobber the original backup.
            newer = Path(td) / "newer.json"
            newer.write_text('[{"t":5,"d":1,"w":"newer"}]', encoding="utf-8")
            pak_io.repack(pak, add_files={"lyrics.json": newer}, keep_backup=True)
            self.assertIn("orig", bak.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

"""A failed job must say WHY it failed. (#16)

The reported truncation was the **UI**: `.msg` sat inside a `nowrap` / `overflow:hidden` /
`text-overflow:ellipsis` span, so a failed job's error was clipped to one line and the part
saying *why* was exactly the part that got cut. That is fixed in screen.js + plugin.css — the
error now gets its own wrapped, selectable, copyable block.

These tests cover the *second* truncation, the one behind it: the plugin capped a server error
body at 300 chars before the UI ever saw it. To be straight about it — a bare FastAPI 422 body is
~139 chars and DID fit, so the cap is not what bit the user this time. But the bodies that carry
the most diagnosis are the ones that don't fit: a multi-field validation error, a 500 with a
traceback, an HTML error page from a reverse proxy. Those are precisely the cases where you need
the text and precisely the cases the old cap ate.

So: keep the body whole up to a sane bound, and when it genuinely has to be cut, SAY so and say
how much there was — a silently cut error is how "the error is truncated" becomes the bug report
instead of the actual bug (#17).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import split_stems  # noqa: E402


class _Resp:
    def __init__(self, text):
        self.text = text


# What FastAPI actually answers when the client omits a required Form field — i.e. the real #17.
_422 = (
    '{"detail":[{"type":"missing","loc":["body","text"],"msg":"Field required",'
    '"input":null,"url":"https://errors.pydantic.dev/2.5/v/missing"}]}'
)


class TheDiagnosisSurvives(unittest.TestCase):
    def test_a_422_validation_body_is_kept_whole(self):
        body = split_stems._err_body(_Resp(_422))
        self.assertIn('"loc":["body","text"]', body,
                      "the field the server rejected is the entire diagnosis — losing it turns "
                      "a fixable bug into 'the error is truncated'")
        self.assertIn("Field required", body)
        self.assertNotIn("…", body, "a body this size must not be truncated at all")

    def test_a_traceback_survives_the_old_cap(self):
        """The case the old 300-char cap really did eat.

        A 500 from the split server carries a traceback, and the LAST line — the exception and
        its message — is the diagnosis. 300 chars keeps the header and throws away the answer."""
        frame = '  File "/app/server.py", line {}, in _do_split\n    run_model(x)\n'
        tb = ("Traceback (most recent call last):\n"
              + "".join(frame.format(i) for i in range(12))
              + "RuntimeError: CUDA out of memory. Tried to allocate 2.20 GiB")
        self.assertGreater(len(tb), 300)
        self.assertNotIn("CUDA out of memory", tb[:300],
                         "the old cap cut the traceback off above the exception")

        body = split_stems._err_body(_Resp(tb))
        self.assertIn("CUDA out of memory", body,
                      "the exception at the bottom is the whole point of a traceback")

    def test_a_short_body_is_passed_through_untouched(self):
        self.assertEqual(split_stems._err_body(_Resp("  Internal Server Error\n")),
                         "Internal Server Error")

    def test_an_empty_body_does_not_crash(self):
        self.assertEqual(split_stems._err_body(_Resp("")), "")
        self.assertEqual(split_stems._err_body(_Resp(None)), "")

    def test_a_giant_html_error_page_is_capped_and_says_so(self):
        # The cap still exists: a server answering with a 2 MB HTML page must not push a novel
        # into the job record, which is persisted to disk and re-read on every load.
        huge = "<html>" + ("x" * 500_000) + "</html>"
        body = split_stems._err_body(_Resp(huge))
        self.assertLess(len(body), split_stems._MAX_ERR_BODY + 100)
        self.assertIn("truncated", body, "a cut body must admit that it was cut")
        self.assertIn(str(len(huge)), body, "and say how much there was")


if __name__ == "__main__":
    unittest.main()

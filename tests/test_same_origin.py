"""Unit tests for ``split_stems._same_origin``.

This helper is security-sensitive: it decides whether the user's API key is attached
to a URL. The stem download URLs come from the SERVER's response, so a self-hosted or
compromised split server could hand back a URL pointing at a host it controls — and if
we treated that as "ours", we'd send it the key.

Two bugs already shipped here and were caught in review, so the tricky forms are pinned
down explicitly:

* an empty ``netloc`` was originally taken to mean "relative", which made
  ``https:evil.com/steal`` (scheme present, netloc empty) look local;
* only the host was compared at first, so a different scheme or port slipped through.

Same import style as ``test_stem_ids`` — ``split_stems`` defers the heavy
``pak_io``/``sloppak`` import, so these run without the feedBack host:
``python -m unittest -v`` from the repo root, or ``python -m pytest tests``.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import split_stems as ss  # noqa: E402

SERVER = "http://127.0.0.1:7865"


class SameOriginAllows(unittest.TestCase):
    """Cases where the key MAY be sent."""

    def test_absolute_paths(self):
        # What the server actually returns, and the only form the caller rewrites onto
        # server_url (a leading "/").
        for url in ("/download/abc123/vocals.flac", "/jobs/abc123"):
            self.assertTrue(ss._same_origin(url, SERVER), url)

    def test_exact_origin(self):
        self.assertTrue(ss._same_origin("http://127.0.0.1:7865/download/x/vocals.wav", SERVER))

    def test_origin_matches_regardless_of_path_or_query(self):
        self.assertTrue(
            ss._same_origin("http://127.0.0.1:7865/download/x/v.flac?token=1#frag", SERVER))


class SameOriginRefuses(unittest.TestCase):
    """Cases where the key MUST NOT be sent."""

    def test_different_host(self):
        self.assertFalse(ss._same_origin("http://evil.example.com/steal", SERVER))

    def test_scheme_only_url_is_not_relative(self):
        # Regression: scheme present, netloc empty. Was classed as "relative" and
        # would have handed evil.com the API key.
        self.assertFalse(ss._same_origin("https:evil.com/steal", SERVER))
        self.assertFalse(ss._same_origin("http:evil.com/steal", SERVER))

    def test_scheme_relative_url(self):
        # "//evil.com/x" inherits the scheme but NOT the host — not ours.
        self.assertFalse(ss._same_origin("//evil.com/steal", SERVER))

    def test_different_scheme(self):
        self.assertFalse(ss._same_origin("https://127.0.0.1:7865/x", SERVER))

    def test_different_port(self):
        self.assertFalse(ss._same_origin("http://127.0.0.1:9999/x", SERVER))

    def test_userinfo_host_confusion(self):
        # "user@host" style: the real host is evil.com, not 127.0.0.1.
        self.assertFalse(ss._same_origin("http://127.0.0.1:7865@evil.com/steal", SERVER))

    def test_localhost_is_not_the_same_as_127_0_0_1(self):
        # Different netloc string; we compare origins, not resolve DNS.
        self.assertFalse(ss._same_origin("http://localhost:7865/x", SERVER))

    def test_bare_relative_and_garbage_are_not_ours(self):
        # We only ever rewrite a leading-"/" path onto server_url, so anything else
        # scheme-less is not a URL of ours - don't attach the key to it.
        for url in ("", "download/abc/vocals.flac", "::::", "?x=1"):
            self.assertFalse(ss._same_origin(url, SERVER), url)


class SameOriginServerUrlForms(unittest.TestCase):
    def test_https_server(self):
        s = "https://split.example.com"
        self.assertTrue(ss._same_origin("/download/x/v.flac", s))
        self.assertTrue(ss._same_origin("https://split.example.com/download/x/v.flac", s))
        self.assertFalse(ss._same_origin("http://split.example.com/download/x/v.flac", s))


if __name__ == "__main__":
    unittest.main()

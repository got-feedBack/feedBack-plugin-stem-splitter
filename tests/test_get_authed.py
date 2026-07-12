"""Unit tests for ``split_stems._get_authed`` — the redirect-safe authed GET.

``_same_origin`` alone was not enough: it only ever sees the FIRST hop. ``requests``
follows redirects by default and, while it drops the ``Authorization`` header on a
cross-host redirect, it knows nothing about the ``X-API-Key`` header this server
authenticates with — it forwards it. So a malicious or compromised split server could
hand back a perfectly on-origin download URL that 302s to a host it controls and harvest
the user's key.

``_get_authed`` disables redirects and re-evaluates the origin at every hop. These tests
pin that down: the key must appear on on-origin hops and MUST NOT appear on any hop that
has left the server's origin.

``python -m unittest -v`` from the repo root, or ``python -m pytest tests``.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import split_stems as ss  # noqa: E402

SERVER = "http://127.0.0.1:7865"
KEY = {"X-API-Key": "secret"}


class FakeResponse:
    def __init__(self, status_code=200, location=None):
        self.status_code = status_code
        self.headers = {"location": location} if location else {}
        self.content = b"audio"
        self.closed = False

    def close(self):
        self.closed = True


class GetAuthed(unittest.TestCase):
    def _run(self, hops, url=SERVER + "/download/x/vocals.flac", headers=KEY):
        """`hops` is the list of responses requests.get returns, in order.
        Returns (result, calls) where calls is the list of (url, headers) seen."""
        calls = []

        def fake_get(u, headers=None, timeout=None, allow_redirects=None):
            calls.append((u, headers))
            # Redirects must be handled by US, never by requests.
            assert allow_redirects is False, "redirects must not be auto-followed"
            return hops[len(calls) - 1]

        with mock.patch("requests.get", side_effect=fake_get):
            try:
                result = ss._get_authed(url, SERVER, headers, timeout=5)
            except RuntimeError as e:
                result = e
        return result, calls

    def test_direct_response_carries_the_key(self):
        (resp, final), calls = self._run([FakeResponse(200)])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(final, SERVER + "/download/x/vocals.flac")
        self.assertEqual(calls, [(SERVER + "/download/x/vocals.flac", KEY)])

    def test_on_origin_redirect_keeps_the_key(self):
        hops = [FakeResponse(302, location="/download/x/vocals-v2.flac"),
                FakeResponse(200)]
        (resp, final), calls = self._run(hops)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(final, SERVER + "/download/x/vocals-v2.flac")
        self.assertEqual([h for _, h in calls], [KEY, KEY])
        self.assertTrue(hops[0].closed, "the redirect response must be closed")

    def test_off_origin_redirect_drops_the_key(self):
        # THE bug this exists to prevent: on-origin first hop, 302 to attacker.
        hops = [FakeResponse(302, location="http://evil.example.com/steal"),
                FakeResponse(200)]
        (resp, final), calls = self._run(hops)
        self.assertEqual(calls[0][1], KEY)          # first hop is ours
        self.assertIsNone(calls[1][1])              # attacker gets NOTHING
        self.assertEqual(final, "http://evil.example.com/steal")

    def test_scheme_relative_redirect_drops_the_key(self):
        # "//evil.com/x" inherits the scheme but not the host.
        hops = [FakeResponse(302, location="//evil.example.com/steal"),
                FakeResponse(200)]
        _, calls = self._run(hops)
        self.assertIsNone(calls[1][1])

    def test_port_change_redirect_drops_the_key(self):
        hops = [FakeResponse(307, location="http://127.0.0.1:9999/steal"),
                FakeResponse(200)]
        _, calls = self._run(hops)
        self.assertIsNone(calls[1][1])

    def test_redirect_back_on_origin_regains_the_key(self):
        # Origin is re-evaluated per hop, not latched off once.
        hops = [FakeResponse(302, location="http://evil.example.com/bounce"),
                FakeResponse(302, location=SERVER + "/download/x/vocals.flac"),
                FakeResponse(200)]
        _, calls = self._run(hops)
        self.assertEqual([h for _, h in calls], [KEY, None, KEY])

    def test_redirect_loop_raises(self):
        hops = [FakeResponse(302, location=SERVER + "/loop")
                for _ in range(ss._MAX_REDIRECTS + 1)]
        result, calls = self._run(hops)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("redirects", str(result))
        self.assertEqual(len(calls), ss._MAX_REDIRECTS + 1)

    def test_no_key_configured_sends_no_headers(self):
        (_, _), calls = self._run([FakeResponse(200)], headers=None)
        self.assertEqual(calls, [(SERVER + "/download/x/vocals.flac", None)])

    def test_redirect_without_location_is_returned_as_is(self):
        # A 302 with no Location isn't a redirect we can follow — don't spin.
        (resp, _), calls = self._run([FakeResponse(302)])
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(len(calls), 1)


class RedactUrl(unittest.TestCase):
    """The URLs we refuse to authenticate to are attacker- or third-party-chosen, and a
    redirect target is very often pre-signed. Logging one verbatim would write someone's
    credential into the app log - the same leak we just refused to make over the wire."""

    def test_strips_signed_query(self):
        u = ("https://s3.example.com/bucket/vocals.flac"
             "?X-Amz-Signature=deadbeef&X-Amz-Credential=AKIA")
        r = ss._redact_url(u)
        self.assertEqual(r, "https://s3.example.com/bucket/vocals.flac?…")
        self.assertNotIn("deadbeef", r)
        self.assertNotIn("AKIA", r)

    def test_strips_bare_token(self):
        r = ss._redact_url("http://evil.example.com/steal?token=hunter2")
        self.assertNotIn("hunter2", r)

    def test_strips_fragment(self):
        self.assertEqual(ss._redact_url("http://h/x.flac#frag"), "http://h/x.flac")

    def test_keeps_scheme_host_path(self):
        self.assertEqual(ss._redact_url("http://127.0.0.1:7865/download/a/vocals.flac"),
                         "http://127.0.0.1:7865/download/a/vocals.flac")

    def test_relative_url(self):
        self.assertEqual(ss._redact_url("/download/a/v.flac?t=1"), "/download/a/v.flac?…")

    def test_garbage_never_raises(self):
        for u in ("", "::::", "http://[", None):
            ss._redact_url(u)   # must not raise - it's on a logging path


if __name__ == "__main__":
    unittest.main()

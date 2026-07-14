"""`update_server()` — the path that lets a fix reach an install that already exists.

Every bug locked in here was found in review, not in use, and none of them would have shown up
in my own live test of the button: I run on the default port, so a dropped port looks fine, and
a progress bar that jumps backwards for one frame is not something you notice while watching a
server restart. They are exactly the bugs a test catches and a demo doesn't.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import demucs_server as ds  # noqa: E402


class _Stubbed:
    """update_server() with everything that touches the network, the disk, or a process stubbed
    out — so what's under test is the ORCHESTRATION: what it emits, and what it hands to start."""

    def __init__(self, td, was_running=True, before="a" * 40, after="b" * 40):
        self.cfg = Path(td)
        self.progress = []          # every (pct, phase) emitted, in order
        self.started = []           # every start_server(**kwargs)
        self._commits = [before, after]
        self.patches = [
            mock.patch.object(ds, "installed", return_value=True),
            mock.patch.object(ds, "is_running", return_value=(was_running, 123)),
            mock.patch.object(ds, "source_meta", side_effect=self._commit),
            mock.patch.object(ds, "stop_server"),
            mock.patch.object(ds, "download_source"),
            mock.patch.object(ds, "write_launcher"),
            mock.patch.object(ds, "patch_driver_scripts", return_value=["run_demucs.py"]),
            mock.patch.object(ds, "verify_install"),
            mock.patch.object(ds, "models_downloaded", return_value=False),
            mock.patch.object(ds, "start_server", side_effect=self._start),
            mock.patch.object(ds, "server_status", return_value={}),
        ]

    def _commit(self, *_a, **_k):
        return {"commit": self._commits.pop(0) if len(self._commits) > 1 else self._commits[0]}

    def _start(self, config_dir, **kw):
        self.started.append(kw)
        return {}

    def _cb(self, ev):
        self.progress.append((ev["pct"], ev["phase"]))

    def run(self, **kw):
        for p in self.patches:
            p.start()
        try:
            return ds.update_server(self.cfg, progress_cb=self._cb, **kw)
        finally:
            for p in self.patches:
                p.stop()


class TheProgressBarNeverGoesBackwards(unittest.TestCase):
    def test_monotonic(self):
        """It emitted 0.86 for the dependency check and then 0.85 for the result — so the bar
        visibly ran backwards. Harmless, and precisely the kind of thing that creeps back in
        every time someone inserts a step."""
        with tempfile.TemporaryDirectory() as td:
            s = _Stubbed(td)
            s.run()
        pcts = [p for p, _ in s.progress]
        self.assertEqual(pcts, sorted(pcts),
                         f"progress must never decrease, got {pcts}")
        self.assertEqual(pcts[-1], 1.0, "and it must finish at 100%")


class TheServerComesBackWhereItWas(unittest.TestCase):
    """The bug: update_server() took port/device/model and then restarted on the DEFAULTS.

    A user who moved the server off 7865 (or onto CUDA) clicked "Update server" and got it back
    somewhere else — or colliding with whatever was already on 7865. Invisible to me: I run on
    the default port, so my live test of the button could not have failed."""

    def test_restarts_on_the_configured_port_device_and_model(self):
        with tempfile.TemporaryDirectory() as td:
            s = _Stubbed(td)
            s.run(port=9123, device="cuda", model="htdemucs")
        self.assertEqual(len(s.started), 1)
        self.assertEqual(s.started[0]["port"], 9123)
        self.assertEqual(s.started[0]["device"], "cuda")
        self.assertEqual(s.started[0]["model"], "htdemucs")

    def test_a_failed_update_puts_the_server_back_on_its_own_port(self):
        # We stopped a WORKING server. If the update dies, leaving it down is strictly worse
        # than never having clicked — and bringing it back on the wrong port is barely better.
        with tempfile.TemporaryDirectory() as td:
            s = _Stubbed(td)
            s.patches.append(
                mock.patch.object(ds, "download_source", side_effect=RuntimeError("no network")))
            with self.assertRaises(RuntimeError):
                s.run(port=9123, device="cuda")
        self.assertEqual(len(s.started), 1, "the server must be restarted, not left down")
        self.assertEqual(s.started[0]["port"], 9123)
        self.assertEqual(s.started[0]["device"], "cuda")

    def test_a_stopped_server_is_not_started_by_an_update(self):
        with tempfile.TemporaryDirectory() as td:
            s = _Stubbed(td, was_running=False)
            s.run()
        self.assertEqual(s.started, [], "updating must not start a server the user had stopped")

    def test_deps_that_no_longer_import_do_not_get_restarted(self):
        # A server that can't import its own dependencies would only crash-loop. "It keeps
        # restarting" is a far worse message than "the update needs new dependencies".
        with tempfile.TemporaryDirectory() as td:
            s = _Stubbed(td)
            s.patches.append(
                mock.patch.object(ds, "verify_install", side_effect=RuntimeError("no sphn")))
            out = s.run()
        self.assertTrue(out.get("needs_reinstall"))
        self.assertEqual(s.started, [], "a server that can't import its deps must not be started")


class CheckUpdateContract(unittest.TestCase):
    def test_unknown_is_present_even_when_github_is_unreachable(self):
        # A caller reading `unknown` to decide whether to offer an update to an install with no
        # recorded commit must get the same signal on every path, not a missing key on one.
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td)
            with mock.patch.object(ds, "installed", return_value=True), \
                 mock.patch.object(ds, "source_meta", return_value={}), \
                 mock.patch.object(ds, "_resolve_commit", return_value=None):
                out = ds.check_update(cfg)
        self.assertIn("unknown", out)
        self.assertTrue(out["unknown"])
        self.assertFalse(out["update_available"])


class StatusPollStaysCheap(unittest.TestCase):
    """server_status() is polled every few seconds and must stay offline and cheap.

    It called models_downloaded() (which is the AND of the three presence checks) AND then each
    of the three again for models_present — running every check twice per poll, including
    _has_whisper()'s rglob() over the whole faster-whisper snapshot tree."""

    def test_the_cache_is_walked_once_per_poll(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td)
            cache = ds.cache_dir(cfg)
            with mock.patch.object(ds, "installed", return_value=True), \
                 mock.patch.object(ds, "is_running", return_value=(False, None)), \
                 mock.patch.object(ds, "_read_state", return_value={}), \
                 mock.patch.object(ds, "_has_roformer", return_value=True) as rof, \
                 mock.patch.object(ds, "_has_whisper", return_value=True) as whi, \
                 mock.patch.object(ds, "_has_aligner", return_value=True) as ali:
                st = ds.server_status(cfg)

        # Count only the calls for OUR cache. The routes tests leave an autostart/status poller
        # on a daemon thread, and it calls these same patched functions with its own config dir —
        # a bare call_count is a cross-test race that fails depending on file order.
        for name, m in (("_has_roformer", rof), ("_has_whisper", whi), ("_has_aligner", ali)):
            mine = [c for c in m.call_args_list if c.args and c.args[0] == cache]
            self.assertEqual(len(mine), 1,
                             f"{name} ran {len(mine)}× in one poll — this is the hot path")
        # and the flag must still agree with the per-model dict it is now derived from
        self.assertTrue(st["models_downloaded"])
        self.assertEqual(st["models_present"],
                         {"bs_roformer_sw": True, "whisperx": True, "whisperx_aligners": True})

    def test_the_flag_is_false_when_any_single_model_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td)
            with mock.patch.object(ds, "installed", return_value=True), \
                 mock.patch.object(ds, "is_running", return_value=(False, None)), \
                 mock.patch.object(ds, "_read_state", return_value={}), \
                 mock.patch.object(ds, "_has_roformer", return_value=True), \
                 mock.patch.object(ds, "_has_whisper", return_value=True), \
                 mock.patch.object(ds, "_has_aligner", return_value=False):
                st = ds.server_status(cfg)
        self.assertFalse(st["models_downloaded"],
                         "warming up with the aligner missing downloads it AT LAUNCH")


if __name__ == "__main__":
    unittest.main()

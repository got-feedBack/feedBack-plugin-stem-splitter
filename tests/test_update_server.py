"""`update_server()` — the path that lets a fix reach an install that already exists.

Every bug locked in here was found in review, not in use, and none of them would have shown up
in my own live test of the button: I run on the default port, so a dropped port looks fine, and
a progress bar that jumps backwards for one frame is not something you notice while watching a
server restart. They are exactly the bugs a test catches and a demo doesn't.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import demucs_server as ds  # noqa: E402
import demucs_server as demucs_server_mod  # noqa: E402  (the module routes.py patches against)

P = "/api/plugins/stem_splitter"

# The lifecycle routes only read these four.
SETTINGS = {
    "remote_model": "bs_roformer_sw",
    "local_server_port": 7865,
    "local_server_device": "",
    "local_server_ref": "",
    "local_server_cuda_tag": "",
    "local_server_gpu": None,
    "local_server_autostart": False,
}


def _settle(seen, tries=50):
    """run_server_op() runs the op on a daemon thread; give it a moment to land."""
    import time
    for _ in range(tries):
        if seen:
            return
        time.sleep(0.05)


class _Stubbed:
    """update_server() with everything that touches the network, the disk, or a process stubbed
    out — so what's under test is the ORCHESTRATION: what it emits, and what it hands to start."""

    def __init__(self, td, was_running=True, before="a" * 40, after="b" * 40, live_port=None):
        self.cfg = Path(td)
        self.progress = []          # every (pct, phase) emitted, in order
        self.started = []           # every start_server(**kwargs)
        self._commits = [before, after]
        self.patches = [
            mock.patch.object(ds, "installed", return_value=True),
            mock.patch.object(ds, "is_running", return_value=(was_running, live_port)),
            mock.patch.object(ds, "source_meta", side_effect=self._commit),
            mock.patch.object(ds, "stop_server"),
            mock.patch.object(ds, "download_source", side_effect=self._download),
            mock.patch.object(ds, "write_launcher"),
            mock.patch.object(ds, "patch_driver_scripts", return_value=["run_demucs.py"]),
            mock.patch.object(ds, "verify_install"),
            mock.patch.object(ds, "models_downloaded", return_value=False),
            mock.patch.object(ds, "start_server", side_effect=self._start),
            mock.patch.object(ds, "server_status", return_value={}),
        ]

    def _commit(self, *_a, **_k):
        return {"commit": self._commits.pop(0) if len(self._commits) > 1 else self._commits[0]}

    # download_source() and start_server() are whole operations with their OWN 0→1 progress.
    # Stubbing them silent hid a real bug: update_server forwarded its callback into them
    # verbatim, so the bar leapt back to 2% right after we reported 20%. A stub that emits
    # nothing cannot see that — so these emit like the real thing.
    def _sub_progress(self, cb):
        if cb:
            cb({"line": "starting…", "pct": 0.02, "phase": "sub"})
            cb({"line": "halfway…", "pct": 0.5, "phase": "sub"})
            cb({"line": "done", "pct": 1.0, "phase": "sub"})

    def _download(self, config_dir, ref=None, progress_cb=None):
        self._sub_progress(progress_cb)

    def _start(self, config_dir, progress_cb=None, **kw):
        self.started.append(kw)
        self._sub_progress(progress_cb)
        return {}

    def _cb(self, ev):
        self.progress.append((ev["pct"], ev["phase"]))

    def run(self, **kw):
        for p in self.patches:
            p.start()
        try:
            return ds.update_server(self.cfg, progress_cb=self._cb, **kw)
        finally:
            # LIFO. A test that patches the same attribute twice (the failure cases below do)
            # has the second patch capture the FIRST MOCK as its original — so unwinding in
            # start order restores a MagicMock onto the module and leaks it into every test
            # that runs after.
            for p in reversed(self.patches):
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

    def test_a_new_source_that_cannot_import_is_never_started(self):
        """The NEW source must not be started — it would only crash-loop, and "it keeps
        restarting" is a far worse message than "the update needs new dependencies".

        What does get started is the OLD source, put back by the rollback (see
        AFailedUpdateLeavesNothingBroken): the user clicked a button and ends up exactly where
        they were, which is the whole point."""
        with tempfile.TemporaryDirectory() as td:
            s = _Stubbed(td)
            s.patches.append(
                mock.patch.object(ds, "verify_install", side_effect=RuntimeError("no sphn")))
            s.patches.append(mock.patch.object(ds, "_restore_source", return_value=True))
            out = s.run()
        self.assertTrue(out.get("needs_reinstall"))
        self.assertTrue(out.get("rolled_back"))
        self.assertEqual(len(s.started), 1,
                         "the rolled-back server must be running again — leaving it stopped is "
                         "the failure this rollback exists to prevent")


    def test_the_live_port_beats_the_configured_one(self):
        """The settings say 7865, the server is actually on 9001 (the user edited the port and
        never restarted). Putting it back on 7865 moves a running server — or collides with
        whatever is there — which is the exact failure the port argument exists to prevent."""
        with tempfile.TemporaryDirectory() as td:
            s = _Stubbed(td, live_port=9001)
            s.run(port=7865)
        self.assertEqual(s.started[0]["port"], 9001,
                         "the server must come back where it was, not where the settings guess")


class RefNormalization(unittest.TestCase):
    """check_update() trimmed the ref; update_server() passed the raw settings value through.

    Same input, two behaviours: a ref with stray whitespace reports an update available and then
    fails to apply it — which the user reads as "the update button is broken"."""

    def test_check_and_apply_normalize_identically(self):
        with tempfile.TemporaryDirectory() as td:
            s = _Stubbed(td)
            seen = {}
            s.patches.append(mock.patch.object(
                ds, "download_source",
                side_effect=lambda cfg, ref=None, progress_cb=None: seen.setdefault("ref", ref)))
            s.run(ref="  main\n")

            with mock.patch.object(ds, "installed", return_value=True), \
                 mock.patch.object(ds, "source_meta", return_value={"commit": "x" * 40}), \
                 mock.patch.object(ds, "_resolve_commit", return_value="y" * 40) as res:
                ds.check_update(s.cfg, ref="  main\n")

        self.assertEqual(seen["ref"], "main")
        self.assertEqual(res.call_args.args[0], "main")

    def test_an_empty_ref_falls_back_to_the_default(self):
        with tempfile.TemporaryDirectory() as td:
            s = _Stubbed(td)
            seen = {}
            s.patches.append(mock.patch.object(
                ds, "download_source",
                side_effect=lambda cfg, ref=None, progress_cb=None: seen.setdefault("ref", ref)))
            s.run(ref="   ")
        self.assertEqual(seen["ref"], ds.DEFAULT_SOURCE_REF)


class TheStubsDoNotLeak(unittest.TestCase):
    """A patch stopped out of order restores a MagicMock onto the module — and every test that
    runs afterwards silently exercises the mock instead of the code."""

    def test_the_module_is_intact_after_a_double_patched_run(self):
        real_download, real_verify = ds.download_source, ds.verify_install
        with tempfile.TemporaryDirectory() as td:
            s = _Stubbed(td)
            s.patches.append(
                mock.patch.object(ds, "verify_install", side_effect=RuntimeError("no sphn")))
            s.run()
        self.assertIs(ds.download_source, real_download)
        self.assertIs(ds.verify_install, real_verify)


class EveryLifecyclePathRunsTheSameServer(unittest.TestCase):
    """The model was threaded through Update and nowhere else.

    So a user who changed the split model got that model after an Update and DEFAULT_MODEL after
    a plain Start — the same server warming a different model depending on which button they
    pressed, and a split whose behaviour depends on how the server happened to come up. Start,
    Update, Install, Prepare models and autostart now all take their (port, device, model) from
    one place."""

    def _routes(self, td, settings):
        """A real JobManager over a real settings file — the routes read it per request, so a
        mock that only spans setup() would be gone by the time the button is pressed."""
        import routes
        (Path(td) / "stem_splitter.json").write_text(json.dumps(settings), encoding="utf-8")
        app = FastAPI()
        # can_manage False keeps the autostart probe from doing anything on its own.
        with mock.patch.object(demucs_server_mod, "can_manage", return_value=(False, "test")):
            routes.setup(app, {"config_dir": td})
        return TestClient(app)

    def test_start_uses_the_configured_model_not_the_default(self):
        seen = {}
        settings = dict(SETTINGS, remote_model="htdemucs", local_server_port=9001,
                        local_server_device="cuda")
        with tempfile.TemporaryDirectory() as td:
            client = self._routes(td, settings)
            with mock.patch.object(demucs_server_mod, "start_server",
                                   side_effect=lambda cfg, **kw: seen.update(kw) or {}):
                client.post(f"{P}/server/start")
                _settle(seen)

        self.assertEqual(seen.get("model"), "htdemucs",
                         "Start ignored the configured model, so the same server warmed a "
                         "different model depending on which button started it")
        self.assertEqual(seen.get("port"), 9001)
        self.assertEqual(seen.get("device"), "cuda")

    def test_prepare_models_downloads_the_configured_model(self):
        # Otherwise the explicit ~2 GB fetch grabs the weights for a model the user isn't using.
        seen = {}
        settings = dict(SETTINGS, remote_model="htdemucs")
        with tempfile.TemporaryDirectory() as td:
            client = self._routes(td, settings)
            with mock.patch.object(demucs_server_mod, "prepare_models",
                                   side_effect=lambda cfg, **kw: seen.update(kw) or {}):
                client.post(f"{P}/server/prepare_models")
                _settle(seen)
        self.assertEqual(seen.get("model"), "htdemucs")


class AFailedUpdateLeavesNothingBroken(unittest.TestCase):
    """download_source() overwrites the source tree IN PLACE.

    So by the time verify_install() discovers the new revision needs a dependency this install
    doesn't have, the old — working — source is already gone. The user is left with a stopped
    server and the only source on disk being one that cannot run: strictly worse than never
    having clicked, and not recoverable from the button they clicked. Snapshot, then roll back.
    """

    def _install(self, td, body="OLD", commit="a" * 40):
        src = ds.src_dir(Path(td))
        src.mkdir(parents=True, exist_ok=True)
        (src / "server.py").write_text(body, encoding="utf-8")
        ds._source_meta_file(Path(td)).write_text(
            json.dumps({"repo": "x", "ref": "main", "commit": commit}), encoding="utf-8")
        return src

    def _run(self, td, download, verify, was_running=True):
        started = []
        # source_meta() is NOT stubbed: it reads source.json, which is exactly what has to roll
        # back with the tree, so the test must see the real file.
        with mock.patch.object(ds, "installed", return_value=True), \
             mock.patch.object(ds, "is_running", return_value=(was_running, 7865)), \
             mock.patch.object(ds, "stop_server"), \
             mock.patch.object(ds, "download_source", side_effect=download), \
             mock.patch.object(ds, "write_launcher"), \
             mock.patch.object(ds, "patch_driver_scripts", return_value=[]), \
             mock.patch.object(ds, "verify_install", side_effect=verify), \
             mock.patch.object(ds, "models_downloaded", return_value=False), \
             mock.patch.object(ds, "server_status", return_value={}), \
             mock.patch.object(ds, "start_server",
                               side_effect=lambda cfg, **kw: started.append(kw) or {}):
            try:
                out = ds.update_server(Path(td))
            except Exception as e:
                out = e
        return out, started

    def _overwrite(self, td, commit="b" * 40):
        """What download_source() really does: rewrite the tree AND record the new commit."""
        def download(config_dir, ref=None, progress_cb=None):
            (ds.src_dir(Path(td)) / "server.py").write_text("NEW", encoding="utf-8")
            ds._source_meta_file(Path(td)).write_text(
                json.dumps({"repo": "x", "ref": "main", "commit": commit}), encoding="utf-8")
        return download

    def _boom(self, msg="no sphn"):
        def verify(cfg, progress_cb=None):
            raise RuntimeError(msg)
        return verify

    def test_a_source_that_cannot_import_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as td:
            src = self._install(td)
            out, started = self._run(td, self._overwrite(td), verify=self._boom())

            self.assertEqual((src / "server.py").read_text(encoding="utf-8"), "OLD",
                             "the working source must be back on disk — rolling forward strands "
                             "the user with a server that only crash-loops")
            self.assertTrue(out["rolled_back"])
            self.assertTrue(out["needs_reinstall"])
            self.assertFalse(out["updated"], "nothing was updated: we put it back")
            self.assertEqual(len(started), 1, "the server it was running must be running again")

    def test_the_recorded_commit_is_rolled_back_with_the_tree(self):
        """source.json is part of the install, not decoration.

        download_source() rewrites it. Restore the tree but not the commit, and check_update()
        believes the REJECTED revision is installed — so it reports "up to date" and stops
        offering the update. The old source runs under a new name, and the fix the user was
        trying to reach becomes unreachable from the button that exists to reach it."""
        with tempfile.TemporaryDirectory() as td:
            self._install(td, commit="a" * 40)
            self._run(td, self._overwrite(td, commit="b" * 40), verify=self._boom())

            self.assertEqual(ds.source_meta(Path(td)).get("commit"), "a" * 40,
                             "the recorded commit must be the one actually on disk, or the "
                             "update we just rejected looks installed and is never offered again")

    def test_no_snapshot_means_no_update(self):
        # An in-place overwrite we cannot undo is the exact failure the snapshot prevents. Better
        # to refuse: nothing has been touched yet, so refusing costs only the click.
        with tempfile.TemporaryDirectory() as td:
            src = self._install(td)
            with mock.patch.object(ds, "_snapshot_source",
                                   side_effect=RuntimeError("disk full")), \
                 mock.patch.object(ds, "installed", return_value=True), \
                 mock.patch.object(ds, "stop_server") as stop, \
                 mock.patch.object(ds, "download_source") as dl:
                with self.assertRaises(RuntimeError):
                    ds.update_server(Path(td))
            stop.assert_not_called()
            dl.assert_not_called()
            self.assertEqual((src / "server.py").read_text(encoding="utf-8"), "OLD")

    def test_a_failed_restore_keeps_the_backup(self):
        # At that moment the snapshot is the ONLY intact copy of a working install. Deleting it
        # turns a bad day into an unrecoverable one.
        with tempfile.TemporaryDirectory() as td:
            src = self._install(td)
            with mock.patch.object(ds, "_restore_source", return_value=False):
                out, _ = self._run(td, self._overwrite(td), verify=self._boom())
            self.assertFalse(out["rolled_back"])
            self.assertTrue(src.with_name(src.name + ".bak").is_dir(),
                            "the backup must survive a failed restore — it is the only copy of "
                            "the user's working install left")

    def test_a_failed_fetch_is_rolled_back_too(self):
        # A half-extracted tree is the same trap as a source that can't import.
        def half_extract(config_dir, ref=None, progress_cb=None):
            (ds.src_dir(config_dir) / "server.py").write_text("HALF", encoding="utf-8")
            raise RuntimeError("connection reset")

        with tempfile.TemporaryDirectory() as td:
            src = self._install(td)
            out, started = self._run(td, half_extract, verify=lambda cfg, progress_cb=None: None)
            self.assertIsInstance(out, RuntimeError)      # the failure is still reported
            self.assertEqual((src / "server.py").read_text(encoding="utf-8"), "OLD")
            self.assertEqual(len(started), 1)

    def test_the_snapshot_is_cleaned_up_on_success(self):
        with tempfile.TemporaryDirectory() as td:
            src = self._install(td)
            self._run(td, self._overwrite(td), verify=lambda cfg, progress_cb=None: None)
            self.assertEqual((src / "server.py").read_text(encoding="utf-8"), "NEW",
                             "a successful update must actually update")
            self.assertFalse(src.with_name(src.name + ".bak").exists(),
                             "the snapshot must not be left behind")


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
                             f"{name} ran {len(mine)}x in one poll — this is the hot path")
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

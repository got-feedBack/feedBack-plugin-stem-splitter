"""The "needs setup" prompt must name what's actually missing.

"The local demucs server is running, but its models haven't been downloaded yet (~2 GB)" reads
as *nothing is downloaded* to a user who already paid for that 2 GB fetch once — and it hides
the common case this release exists to fix, where everything is present except the wav2vec2
aligner the server's cache sweeper ate overnight.

missing_models() has always known which ones are absent. It just wasn't wired to the one place
the user would ever see it.
"""
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import demucs_server  # noqa: E402
import routes  # noqa: E402


class TheSetupPromptNamesTheMissingWeights(unittest.TestCase):
    def _prompt(self, missing):
        with tempfile.TemporaryDirectory() as td:
            mgr = routes.JobManager.__new__(routes.JobManager)   # no worker thread, no probes
            mgr.config_dir = Path(td)
            with mock.patch.object(mgr, "resolve_split_engine", return_value=("remote", "")), \
                 mock.patch.object(mgr, "local_server_url", return_value="http://127.0.0.1:7865"), \
                 mock.patch.object(demucs_server, "missing_models", return_value=missing):
                return mgr.needs_server_setup()

    def test_nothing_missing_is_not_a_prompt(self):
        self.assertIsNone(self._prompt([]))

    def test_only_the_aligner_says_so(self):
        # The case the sweeper actually produces: 2 GB on disk, 361 MB eaten.
        out = self._prompt(["whisperx aligner"])
        self.assertEqual(out["missing"], ["whisperx aligner"])
        self.assertIn("whisperx aligner", out["message"])

    def test_the_size_is_what_is_actually_being_fetched(self):
        # A flat "~2 GB" overstates the aligner-only case by 5×, and 2 GB is exactly the number
        # that makes someone cancel a 360 MB download.
        out = self._prompt(["whisperx aligner"])
        self.assertIn("MB", out["size"])
        self.assertNotIn("GB", out["message"])

    def test_a_fresh_install_names_them_all_and_says_gigabytes(self):
        out = self._prompt(["bs_roformer_sw", "whisperx", "whisperx aligner"])
        for name in ("bs_roformer_sw", "whisperx", "whisperx aligner"):
            self.assertIn(name, out["message"])
        self.assertIn("GB", out["size"])

    def test_a_forced_local_engine_is_never_blocked(self):
        # A user who forced demucs/audio-separator doesn't need the server's models at all, and
        # must not be blocked just because the server happens to be running without them.
        with tempfile.TemporaryDirectory() as td:
            mgr = routes.JobManager.__new__(routes.JobManager)
            mgr.config_dir = Path(td)
            with mock.patch.object(mgr, "resolve_split_engine", return_value=("demucs", "")), \
                 mock.patch.object(demucs_server, "missing_models", return_value=["whisperx"]):
                self.assertIsNone(mgr.needs_server_setup())

    def test_a_real_remote_server_is_not_ours_to_set_up(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = routes.JobManager.__new__(routes.JobManager)
            mgr.config_dir = Path(td)
            with mock.patch.object(mgr, "resolve_split_engine", return_value=("remote", "")), \
                 mock.patch.object(mgr, "local_server_url", return_value=None), \
                 mock.patch.object(demucs_server, "missing_models", return_value=["whisperx"]):
                self.assertIsNone(mgr.needs_server_setup())


class ReAlignAsksItsOwnQuestion(unittest.TestCase):
    """A re-align never splits. It hands the vocal stem it already has, plus the lyrics, to the
    server's aligner.

    So gating it on the SPLIT engine — which is what the shared check used to do — is wrong twice
    over: it blocks a user who forced local demucs from a job that never touches the split engine,
    and then offers them the 700 MB roformer separator for an alignment that will never load it.
    """

    def _prompt(self, missing, *, lyrics_url="http://127.0.0.1:7865",
                local_url="http://127.0.0.1:7865", split_engine="demucs"):
        with tempfile.TemporaryDirectory() as td:
            mgr = routes.JobManager.__new__(routes.JobManager)
            mgr.config_dir = Path(td)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    mgr, "_lyrics_server_url", return_value=lyrics_url))
                stack.enter_context(mock.patch.object(
                    mgr, "local_server_url", return_value=local_url))
                stack.enter_context(mock.patch.object(
                    mgr, "resolve_split_engine", return_value=(split_engine, "")))
                stack.enter_context(mock.patch.object(
                    demucs_server, "missing_models", return_value=missing))
                stack.enter_context(mock.patch.object(
                    demucs_server, "download_size", return_value="~1.9 GB"))
                return mgr.needs_server_setup("realign")

    def test_it_does_not_ask_for_the_separator(self):
        """The roformer checkpoint is 700 MB and alignment never loads it. Demanding it puts a
        700 MB download in front of a job that doesn't need a byte of it."""
        out = self._prompt(["bs_roformer_sw"])
        self.assertIsNone(out, "a missing separator must not block a re-align")

    def test_it_does_ask_for_the_aligner(self):
        out = self._prompt(["whisperx aligner"])
        self.assertEqual(out["missing"], ["whisperx aligner"])
        self.assertIn("Re-aligning", out["message"], "the prompt must speak about THIS job")

    def test_it_filters_the_separator_out_of_a_mixed_list(self):
        out = self._prompt(["bs_roformer_sw", "whisperx", "whisperx aligner"])
        self.assertEqual(out["missing"], ["whisperx", "whisperx aligner"])
        self.assertNotIn("bs_roformer_sw", out["message"])

    def test_everything_present_is_no_prompt(self):
        self.assertIsNone(self._prompt([]))

    def test_the_split_engine_is_irrelevant(self):
        # A user who forced local demucs must not be blocked from a job that never splits.
        out = self._prompt(["whisperx"], split_engine="demucs")
        self.assertIsNotNone(out, "the split engine has nothing to do with re-aligning")

    def test_someone_elses_server_is_not_ours_to_set_up(self):
        out = self._prompt(["whisperx"], lyrics_url="http://nas.local:9000",
                           local_url="http://127.0.0.1:7865")
        self.assertIsNone(out, "we cannot download models onto a server we do not manage")

    def test_no_lyrics_server_at_all(self):
        self.assertIsNone(self._prompt(["whisperx"], lyrics_url=None))


class TheSplitPathIsUnchanged(unittest.TestCase):
    """The re-align branch must not have moved the ground under the split one."""

    def test_split_still_asks_about_every_model(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = routes.JobManager.__new__(routes.JobManager)
            mgr.config_dir = Path(td)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    mgr, "resolve_split_engine", return_value=("remote", "")))
                stack.enter_context(mock.patch.object(
                    mgr, "local_server_url", return_value="http://127.0.0.1:7865"))
                stack.enter_context(mock.patch.object(
                    demucs_server, "missing_models", return_value=["bs_roformer_sw"]))
                stack.enter_context(mock.patch.object(
                    demucs_server, "download_size", return_value="~700 MB"))
                out = mgr.needs_server_setup()      # default kind
        self.assertEqual(out["missing"], ["bs_roformer_sw"],
                         "a split DOES need the separator — don't filter it out here")


if __name__ == "__main__":
    unittest.main()

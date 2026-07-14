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


if __name__ == "__main__":
    unittest.main()

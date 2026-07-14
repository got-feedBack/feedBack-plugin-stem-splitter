"""Green at shutdown must mean green at startup — with no network.

Two bugs conspired to break that, and the user watched ~1 GB come down every time they
opened the app:

1. The server's 24h cache sweeper deleted the model weights (fixed upstream in
   feedBack-demucs-server: it now deletes only things that look like stem caches).

2. `models_downloaded()` — the gate that decides warmup-vs-skip-warmup at launch — checked
   only the ROFORMER checkpoint, on the reasoning that bs_roformer_sw is the model we split
   with. But warmup doesn't warm only the model we split with: it warms whisperx and its
   wav2vec2 aligner too. So an install with the checkpoint but no aligner reported
   "downloaded", started WITH warmup, and the server quietly fetched 361 MB at launch — the
   exact thing this plugin promises never to do.

The gate is now all-or-nothing: everything warmup would touch, or we start with
--skip-warmup and fetch nothing until the user asks.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import demucs_server as ds  # noqa: E402


class Cache:
    """Builds a cache dir with whichever weights you name."""

    def __init__(self, td, roformer=False, whisper=False, aligner=False, old_aligner=False):
        self.cfg = Path(td)
        c = ds.cache_dir(self.cfg)
        c.mkdir(parents=True, exist_ok=True)
        if roformer:
            (c / "_roformer-models").mkdir(parents=True, exist_ok=True)
            (c / "_roformer-models" / "BS-Roformer-SW.ckpt").write_bytes(b"x")
        if whisper:
            (c / "huggingface" / "hub" /
             "models--Systran--faster-whisper-medium").mkdir(parents=True, exist_ok=True)
        if aligner:
            d = c / "torch" / "hub" / "checkpoints"
            d.mkdir(parents=True, exist_ok=True)
            (d / "wav2vec2_fairseq_base_ls960_asr_ls960.pth").write_bytes(b"x")
        if old_aligner:                       # the pre-move layout
            d = c / "hub" / "checkpoints"
            d.mkdir(parents=True, exist_ok=True)
            (d / "wav2vec2_fairseq_base_ls960_asr_ls960.pth").write_bytes(b"x")


class TheGateIsAllOrNothing(unittest.TestCase):
    def test_everything_present_warms_up(self):
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, roformer=True, whisper=True, aligner=True)
            self.assertTrue(ds.models_downloaded(c.cfg))
            self.assertEqual(ds.missing_models(c.cfg), [])

    def test_roformer_alone_does_NOT_warm_up(self):
        """THE bug. Warmup would have fetched whisperx + the aligner at launch."""
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, roformer=True)
            self.assertFalse(
                ds.models_downloaded(c.cfg),
                "warming up with the aligner missing makes the server download it AT LAUNCH")
            self.assertIn("whisperx aligner", ds.missing_models(c.cfg))

    def test_missing_aligner_alone_does_not_warm_up(self):
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, roformer=True, whisper=True)
            self.assertFalse(ds.models_downloaded(c.cfg))
            self.assertEqual(ds.missing_models(c.cfg), ["whisperx aligner"])

    def test_nothing_present(self):
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td)
            self.assertFalse(ds.models_downloaded(c.cfg))
            self.assertEqual(len(ds.missing_models(c.cfg)), 3)

    def test_missing_models_names_them(self):
        # "not ready" is useless; "whisperx aligner is missing" is actionable.
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, whisper=True)
            self.assertEqual(set(ds.missing_models(c.cfg)),
                             {"bs_roformer_sw", "whisperx aligner"})


class TorchHomeMigration(unittest.TestCase):
    """TORCH_HOME moved from the cache root to cache/torch, so the sweeper stops eating the
    aligner. Pointing torch at the new path WITHOUT moving the file would re-download 361 MB —
    a re-download, in a change whose whole purpose is to stop re-downloading."""

    def test_the_old_layout_is_still_recognized(self):
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, roformer=True, whisper=True, old_aligner=True)
            self.assertTrue(ds.models_downloaded(c.cfg),
                            "an existing install must not be told its weights are missing "
                            "just because we moved the goalposts")

    def test_the_file_is_moved_not_refetched(self):
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, old_aligner=True)
            cache = ds.cache_dir(c.cfg)
            ds._migrate_torch_home(cache)
            self.assertFalse((cache / "hub").exists(), "old location should be gone")
            self.assertTrue(
                (cache / "torch" / "hub" / "checkpoints" /
                 "wav2vec2_fairseq_base_ls960_asr_ls960.pth").exists(),
                "the 361 MB file must have MOVED, not been re-downloaded")

    def test_migration_is_idempotent_and_never_clobbers(self):
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, aligner=True, old_aligner=True)   # both exist
            cache = ds.cache_dir(c.cfg)
            ds._migrate_torch_home(cache)                   # must not overwrite the new one
            self.assertTrue((cache / "torch" / "hub" / "checkpoints").is_dir())
            ds._migrate_torch_home(cache)                   # and must be safe to re-run

    def test_no_old_dir_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, aligner=True)
            ds._migrate_torch_home(ds.cache_dir(c.cfg))     # must not raise


if __name__ == "__main__":
    unittest.main()

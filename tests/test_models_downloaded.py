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

    # Sizes and marker bytes, not payloads. Materializing 60 MB byte strings at import cost
    # ~180 MB resident for tests that only ever assert on size and first byte — enough to slow,
    # or flake, a memory-constrained CI runner. write() writes the marker and truncates to
    # length: same assertions, a sparse file, and nothing held in memory.
    BIG = 60 * 1024 * 1024               # over _MIN_WEIGHT_BYTES
    SMALL = 1024                         # a partial download
    # Distinguishable. The never-clobber test compared SIZES of two files written with identical
    # bytes — so it would have passed whether or not the destination was overwritten. A test that
    # cannot fail on the bug it exists to catch is worse than none.
    NEW, NEW_BYTE = 60 * 1024 * 1024, b"N"   # what is already at the destination
    OLD, OLD_BYTE = 61 * 1024 * 1024, b"O"   # the old layout's copy (different size AND byte)

    @staticmethod
    def write(path: Path, size: int, marker: bytes = b"x") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(marker)
            f.truncate(size)

    def __init__(self, td, roformer=False, whisper=False, aligner=False, old_aligner=False,
                 empty_new_hub=False, whisper_shell=False, whisper_incomplete=False,
                 tiny_roformer=False):
        self.cfg = Path(td)
        c = ds.cache_dir(self.cfg)
        c.mkdir(parents=True, exist_ok=True)
        if roformer or tiny_roformer:
            self.write(c / "_roformer-models" / "BS-Roformer-SW.ckpt",
                       self.SMALL if tiny_roformer else self.BIG)
        if whisper or whisper_shell or whisper_incomplete:
            repo = c / "huggingface" / "hub" / "models--Systran--faster-whisper-medium"
            (repo / "blobs").mkdir(parents=True, exist_ok=True)
            (repo / "refs").mkdir(parents=True, exist_ok=True)
            rev = repo / "snapshots" / "08e178d4"
            rev.mkdir(parents=True, exist_ok=True)
            if whisper:                          # a completed download
                self.write(rev / "model.bin", self.BIG)
            if whisper_incomplete:               # interrupted: payload + a .incomplete marker
                self.write(rev / "model.bin", self.BIG)
                self.write(repo / "blobs" / "abc123.incomplete", self.SMALL)
            # whisper_shell: the directory structure and nothing else — which is exactly what
            # huggingface_hub leaves behind the moment a download STARTS.
        if aligner:
            self.write(c / "torch" / "hub" / "checkpoints" /
                       "wav2vec2_fairseq_base_ls960_asr_ls960.pth", self.NEW, self.NEW_BYTE)
        if old_aligner:                       # the pre-move layout
            self.write(c / "hub" / "checkpoints" /
                       "wav2vec2_fairseq_base_ls960_asr_ls960.pth", self.OLD, self.OLD_BYTE)
        if empty_new_hub:                     # torch/hub exists but is EMPTY
            (c / "torch" / "hub").mkdir(parents=True, exist_ok=True)


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


class PartialDownloadsAreNotWeights(unittest.TestCase):
    '''"The file is there" is not the same claim as "the weights are on disk", and this gate
    promises the second one. A directory that exists because a download STARTED must not be
    mistaken for one that exists because a download FINISHED.'''

    def test_whisper_directory_shell_is_not_enough(self):
        """huggingface_hub creates models--org--repo/{blobs,refs,snapshots} when the download
        BEGINS. A presence check would report the weights as ready, start the server WITH
        warmup, and have it resume the download at launch — the exact silent startup fetch this
        gate exists to prevent, now wearing a convincing disguise."""
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, roformer=True, aligner=True, whisper_shell=True)
            self.assertFalse(ds._has_whisper(ds.cache_dir(c.cfg)))
            self.assertFalse(ds.models_downloaded(c.cfg))
            self.assertIn("whisperx", ds.missing_models(c.cfg))

    def test_whisper_with_an_incomplete_marker_is_not_ready(self):
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, roformer=True, aligner=True, whisper_incomplete=True)
            self.assertFalse(ds._has_whisper(ds.cache_dir(c.cfg)))

    def test_a_truncated_roformer_checkpoint_is_not_ready(self):
        # A 1 KB .ckpt is a failed download, not a 700 MB model.
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, tiny_roformer=True, whisper=True, aligner=True)
            self.assertFalse(ds._has_roformer(ds.cache_dir(c.cfg)))
            self.assertFalse(ds.models_downloaded(c.cfg))


class MigrationWhenTheDestinationAlreadyExists(unittest.TestCase):
    """The bug: _migrate_torch_home() bailed out whenever cache/torch/hub existed.

    It can exist and be EMPTY (an earlier run created it) while the 361 MB aligner is still only
    in the old location. _has_aligner() accepts the old layout, so we would report the weights
    as present, start WITH warmup, and torch — looking under the NEW TORCH_HOME — would find
    nothing and re-download at launch. The one case the fast path skips is the one that breaks.
    """

    def test_empty_destination_does_not_strand_the_aligner(self):
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, old_aligner=True, empty_new_hub=True)
            cache = ds.cache_dir(c.cfg)
            ds._migrate_torch_home(cache)
            moved = (cache / "torch" / "hub" / "checkpoints" /
                     "wav2vec2_fairseq_base_ls960_asr_ls960.pth")
            self.assertTrue(moved.is_file(),
                            "the aligner must be MOVED into the new layout, not left behind "
                            "for torch to re-download")
            self.assertTrue(ds._has_aligner(cache))

    def test_a_file_already_at_the_destination_is_never_clobbered(self):
        """The two files must be DISTINGUISHABLE, or this test cannot fail.

        It originally wrote identical bytes to both locations and compared sizes — so it would
        have passed whether or not the destination was overwritten. Caught in review; a test
        that cannot fail on the bug it exists to catch is worse than no test, because it
        converts absent coverage into confidence.
        """
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, aligner=True, old_aligner=True)      # both layouts hold a file
            cache = ds.cache_dir(c.cfg)
            dest = (cache / "torch" / "hub" / "checkpoints" /
                    "wav2vec2_fairseq_base_ls960_asr_ls960.pth")
            ds._migrate_torch_home(cache)

            with open(dest, "rb") as f:
                first = f.read(1)
            self.assertEqual(dest.stat().st_size, Cache.NEW,
                             "the destination file must be the one that was already there")
            self.assertEqual(first, Cache.NEW_BYTE,
                             "the old-layout file must NOT have overwritten it — this is the "
                             "file torch will actually use")

    def test_other_files_still_merge_across(self):
        with tempfile.TemporaryDirectory() as td:
            c = Cache(td, aligner=True)
            cache = ds.cache_dir(c.cfg)
            old = cache / "hub" / "checkpoints"
            Cache.write(old / "some_other_model.pth", Cache.BIG)
            ds._migrate_torch_home(cache)
            self.assertTrue((cache / "torch" / "hub" / "checkpoints" /
                             "some_other_model.pth").is_file())


if __name__ == "__main__":
    unittest.main()

"""Selective re-split merge rules (issue #11).

People replace stems on purpose (a re-recorded guitar) and add stems the
engines know nothing about (a click track). A re-split must overwrite only
what the user selected; everything else keeps its manifest entry verbatim.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from split_stems import _merge_stem_entries  # noqa: E402


def e(sid, file=None, default=True, **extra):
    d = {"id": sid, "file": file or f"stems/{sid}.ogg", "default": default}
    d.update(extra)
    return d


class TestMergeStemEntries(unittest.TestCase):
    def test_replace_all_none_replaces_engine_outputs(self):
        existing = [e("guitar", "stems/guitar.wav"), e("drums")]
        produced = [e("guitar"), e("drums"), e("bass")]
        merged = _merge_stem_entries(existing, produced, None)
        by_id = {s["id"]: s for s in merged}
        self.assertEqual(by_id["guitar"]["file"], "stems/guitar.ogg")  # replaced
        self.assertIn("bass", by_id)

    def test_replace_all_preserves_custom_stems(self):
        # The old wholesale replacement silently delisted these.
        existing = [e("guitar"), e("click", "stems/click.ogg")]
        produced = [e("guitar"), e("drums")]
        merged = _merge_stem_entries(existing, produced, None)
        self.assertIn("click", [s["id"] for s in merged])

    def test_unselected_stem_keeps_entry_verbatim(self):
        mine = e("guitar", "stems/my_take.flac", default=False, language="en")
        existing = [mine, e("drums")]
        produced = [e("guitar"), e("drums")]
        merged = _merge_stem_entries(existing, produced, {"drums"})
        by_id = {s["id"]: s for s in merged}
        self.assertEqual(by_id["guitar"], mine)                        # untouched
        self.assertEqual(by_id["drums"]["file"], "stems/drums.ogg")    # replaced

    def test_new_id_lands_even_when_unselected(self):
        # Selection protects EXISTING stems; a stem the pak never had has
        # nothing to protect and always lands.
        existing = [e("guitar")]
        produced = [e("guitar"), e("piano")]
        merged = _merge_stem_entries(existing, produced, {"guitar"})
        self.assertIn("piano", [s["id"] for s in merged])

    def test_full_fallback_survives_merge(self):
        existing = [e("guitar"), e("full", "stems/full.wav", default=False)]
        produced = [e("guitar")]
        merged = _merge_stem_entries(existing, produced, {"guitar"})
        by_id = {s["id"]: s for s in merged}
        self.assertEqual(by_id["full"]["file"], "stems/full.wav")

    def test_canonical_order(self):
        existing = [e("other"), e("full", default=False)]
        produced = [e("drums"), e("guitar")]
        merged = _merge_stem_entries(existing, produced, None)
        ids = [s["id"] for s in merged]
        self.assertEqual(ids, ["guitar", "drums", "other", "full"])


if __name__ == "__main__":
    unittest.main()

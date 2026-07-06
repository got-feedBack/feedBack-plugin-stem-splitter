"""Unit tests for the pure stem-id helpers in ``split_stems``.

These import ``split_stems`` directly (it defers the heavy ``pak_io``/``sloppak``
import), so they run without the feedBack host:  ``python -m unittest -v`` from
the repo root, or ``python -m pytest tests``.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import split_stems as ss  # noqa: E402


class NormalizeStemId(unittest.TestCase):
    def test_demucs_bare_names(self):
        for name in ("vocals", "drums", "bass", "guitar", "piano", "other"):
            self.assertEqual(ss._normalize_stem_id(name), name)

    def test_audio_separator_paren_labels(self):
        v = "mix_(Vocals)_model_bs_roformer_ep_317_sdr_12.9755"
        i = "mix_(Instrumental)_model_bs_roformer_ep_317_sdr_12.9755"
        self.assertEqual(ss._normalize_stem_id(v), "vocals")
        self.assertEqual(ss._normalize_stem_id(i), "other")

    def test_bs_roformer_sw_6_stem_labels(self):
        # Real server naming: "<base>_(<Label>)_BS-Roformer-SW.flac"
        for label, expect in [("Vocals", "vocals"), ("Drums", "drums"),
                              ("Bass", "bass"), ("Guitar", "guitar"),
                              ("Piano", "piano"), ("Other", "other")]:
            name = f"mix_({label})_BS-Roformer-SW"
            self.assertEqual(ss._normalize_stem_id(name), expect,
                             f"{name!r} -> {expect}")

    def test_model_token_does_not_shadow_paren_label(self):
        # "BS-Roformer-SW" contains no stem word, but ensure the paren label wins.
        self.assertEqual(
            ss._normalize_stem_id("mix_(Bass)_BS-Roformer-SW"), "bass")

    def test_no_vocals_maps_to_other_not_vocals(self):
        # "no_vocals" (the instrumental companion) must not match bare "vocals".
        self.assertEqual(ss._normalize_stem_id("mix_no_vocals_htdemucs"), "other")
        self.assertEqual(ss._normalize_stem_id("mix_vocals_htdemucs"), "vocals")

    def test_word_boundary_avoids_false_match(self):
        # "brother" contains "other" but not as a token — must not map.
        self.assertIsNone(ss._normalize_stem_id("brother"))

    def test_unknown_returns_none(self):
        self.assertIsNone(ss._normalize_stem_id("mix_12345_checkpoint"))

    def test_keys_alias(self):
        self.assertEqual(ss._normalize_stem_id("mix_(Keys)_model"), "piano")


class Sanitize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(ss._sanitize("(Vocals) 2!"), "vocals_2")

    def test_empty_falls_back(self):
        self.assertEqual(ss._sanitize("!!!"), "stem")


if __name__ == "__main__":
    unittest.main()

"""The first-run flow must not require an app restart (user report).

Sequence that broke: the user tries to split BEFORE any engine is installed
(the availability probe puts the not-yet-existing engine dir on sys.path and
fails an import), then installs the engine through the plugin UI, then tries
again — and the import still fails until the app restarts, because the first
failure left a negative finder for the engine dir in sys.path_importer_cache.

_prepend_engine_path() must invalidate import caches so the post-install
attempt sees the freshly created pip --target tree.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import split_stems  # noqa: E402


class TestEngineImportCache(unittest.TestCase):
    def test_engine_installed_after_failed_probe_is_importable(self):
        modname = "fake_engine_pkg_for_cache_test"
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "engine"     # does not exist yet — pre-install

            # 1. First-run probe: dir on sys.path while absent, import fails.
            split_stems._prepend_engine_path(str(engine))
            try:
                with self.assertRaises(ImportError):
                    __import__(modname)

                # 2. "Install" the engine (what pip --target does).
                engine.mkdir()
                (engine / f"{modname}.py").write_text("x = 1", encoding="utf-8")

                # 3. Second attempt goes through the same helper — must now
                #    import. Without invalidate_caches() this raises
                #    ImportError until the process restarts (stale
                #    sys.path_importer_cache entry).
                split_stems._prepend_engine_path(str(engine))
                mod = __import__(modname)
                self.assertEqual(mod.x, 1)
            finally:
                sys.path.remove(str(engine))
                sys.modules.pop(modname, None)


if __name__ == "__main__":
    unittest.main()

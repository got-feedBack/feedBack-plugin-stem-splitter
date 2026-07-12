"""Unit tests for ``demucs_server._bootstrap_insert_line``.

The bootstrap is injected into the demucs server's driver scripts (``run_demucs.py``,
``run_roformer.py``) so the subprocesses they run can see the plugin's ``pip --target``
tree. Getting the insertion point wrong is not cosmetic:

* above a ``from __future__ import ...`` it's a hard SyntaxError — every split would
  fail with a traceback pointing at a file the user never wrote;
* above the module docstring it silently demotes the docstring to a bare expression.

Both drivers open with a docstring today and neither has a ``__future__`` import, but
this runs against whatever upstream ships, so pin the behaviour down.

``demucs_server`` imports cleanly without the feedBack host, so:
``python -m unittest -v`` from the repo root, or ``python -m pytest tests``.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import demucs_server as ds  # noqa: E402


class BootstrapInsertLine(unittest.TestCase):
    def test_bare_module(self):
        self.assertEqual(ds._bootstrap_insert_line("import os\n"), 0)

    def test_shebang_and_coding_cookie(self):
        src = "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\nimport os\n"
        self.assertEqual(ds._bootstrap_insert_line(src), 2)

    def test_module_docstring_is_kept_first(self):
        src = '"""Driver."""\nimport os\n'
        self.assertEqual(ds._bootstrap_insert_line(src), 1)

    def test_multiline_docstring(self):
        src = '"""Driver.\n\nLonger blurb.\n"""\nimport os\n'
        self.assertEqual(ds._bootstrap_insert_line(src), 4)

    def test_future_import_must_stay_at_top(self):
        # Inserting above this is a SyntaxError, not a style nit.
        src = '"""Driver."""\nfrom __future__ import annotations\nimport os\n'
        self.assertEqual(ds._bootstrap_insert_line(src), 2)

    def test_shebang_docstring_and_future_together(self):
        src = ('#!/usr/bin/env python\n'
               '"""Driver."""\n'
               'from __future__ import annotations\n'
               'import os\n')
        self.assertEqual(ds._bootstrap_insert_line(src), 3)

    def test_unparseable_source_falls_back_to_shebang_scan(self):
        # Not ours to validate — a broken driver should fail on its own terms.
        src = "#!/usr/bin/env python\ndef (:\n"
        self.assertEqual(ds._bootstrap_insert_line(src), 1)

    def test_patched_source_still_compiles(self):
        # The real end-to-end invariant: whatever we splice in must parse.
        src = ('#!/usr/bin/env python\n'
               '"""Driver."""\n'
               'from __future__ import annotations\n'
               'import os\n'
               'print(os.getcwd(), __doc__)\n')
        boot = ds._BOOTSTRAP_TEMPLATE.format(pylibs="/tmp/pylibs")
        lines = src.splitlines(keepends=True)
        head = ds._bootstrap_insert_line(src)
        patched = "".join(lines[:head]) + boot + "".join(lines[head:])
        compile(patched, "run_demucs.py", "exec")   # raises on a misplaced __future__
        self.assertIn(ds._BOOTSTRAP_MARKER, patched)

    def test_docstring_survives_the_patch(self):
        src = '"""Driver."""\nimport os\n'
        boot = ds._BOOTSTRAP_TEMPLATE.format(pylibs="/tmp/pylibs")
        lines = src.splitlines(keepends=True)
        head = ds._bootstrap_insert_line(src)
        patched = "".join(lines[:head]) + boot + "".join(lines[head:])
        ns: dict = {}
        exec(compile(patched, "run_demucs.py", "exec"), ns)
        self.assertEqual(ns["__doc__"], "Driver.")


if __name__ == "__main__":
    unittest.main()

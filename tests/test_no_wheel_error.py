"""Unit tests for ``demucs_server._is_no_wheel_error``.

``_install_diffq`` tolerates exactly one failure: "no compatible wheel exists for this
interpreter" (true on macOS 3.11+ and Linux 3.13, where neither diffq nor diffq-fixed
publishes one). It carries on without diffq, which is safe — diffq is only needed for
quantized demucs checkpoints, not for bs_roformer_sw.

Everything else — a network blip, a dead index, a proxy, a permissions error — is a REAL
failure. Swallowing it would silently degrade the install: pip would be skipped, the
install would report success, and the user would only discover it much later as a
ModuleNotFoundError raised from a subprocess. So this predicate is what stands between
"expected, carry on" and "fail loudly", and it must not over-match.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import demucs_server as ds  # noqa: E402


def _err(pip_output: str, msg: str = "pip install failed (exit 1) for diffq-fixed"):
    e = RuntimeError(msg)
    e.pip_output = pip_output
    return e


class NoWheelIsTolerated(unittest.TestCase):
    """The two lines pip actually prints when no wheel exists. Both must match."""

    def test_no_matching_distribution(self):
        self.assertTrue(_is := ds._is_no_wheel_error(_err(
            "ERROR: No matching distribution found for diffq-fixed>=0.2")))

    def test_could_not_find_a_version(self):
        self.assertTrue(ds._is_no_wheel_error(_err(
            "ERROR: Could not find a version that satisfies the requirement "
            "diffq-fixed>=0.2 (from versions: none)")))

    def test_real_pip_output_verbatim(self):
        # Exactly what `pip download --only-binary=:all:` prints for linux-cp313.
        self.assertTrue(ds._is_no_wheel_error(_err(
            "ERROR: Could not find a version that satisfies the requirement "
            "diffq-fixed>=0.2 (from versions: none)\n"
            "ERROR: No matching distribution found for diffq-fixed>=0.2")))

    def test_case_insensitive(self):
        self.assertTrue(ds._is_no_wheel_error(_err("no matching distribution found")))


class EverythingElseMustFail(unittest.TestCase):
    """These must NOT be mistaken for "no wheel" — each is a real failure that would
    otherwise silently produce an install missing diffq."""

    def test_network_failure(self):
        self.assertFalse(ds._is_no_wheel_error(_err(
            "WARNING: Retrying (Retry(total=4)) after connection broken by "
            "'NewConnectionError(...): Failed to establish a new connection'\n"
            "ERROR: Could not install packages due to an OSError")))

    def test_dead_index(self):
        self.assertFalse(ds._is_no_wheel_error(_err(
            "ERROR: Could not install packages due to an OSError: "
            "HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.")))

    def test_permission_error(self):
        self.assertFalse(ds._is_no_wheel_error(_err(
            "ERROR: Could not install packages due to an OSError: "
            "[Errno 13] Permission denied: '/target/diffq'")))

    def test_resolution_conflict(self):
        self.assertFalse(ds._is_no_wheel_error(_err(
            "ERROR: Cannot install diffq-fixed because these package versions have "
            "conflicting dependencies.")))

    def test_disk_full(self):
        self.assertFalse(ds._is_no_wheel_error(_err(
            "ERROR: Could not install packages due to an OSError: "
            "[Errno 28] No space left on device")))

    def test_empty_output(self):
        self.assertFalse(ds._is_no_wheel_error(_err("")))

    def test_exception_without_pip_output_attribute(self):
        # A RuntimeError from somewhere else entirely must not be swallowed.
        self.assertFalse(ds._is_no_wheel_error(RuntimeError("boom")))


if __name__ == "__main__":
    unittest.main()

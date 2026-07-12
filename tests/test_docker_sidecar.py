"""Unit tests for ``docker_sidecar``'s pure helpers — no Docker daemon required.

The interesting parts of this module (pull, create, start) need a live daemon and are
exercised by hand. But three things are pure, load-bearing, and easy to get wrong in ways
that are invisible until they bite someone:

* **Port binding.** The published port must be bound to **127.0.0.1**, not 0.0.0.0.
  Docker's default is 0.0.0.0, which would put an unauthenticated inference server on every
  interface of the host — the whole LAN. Nothing needs that: Electron reaches it on
  loopback, and a containerized feedBack reaches it by container name over the shared
  network. This is a security property, so it gets a test.

* **The published port is NOT the managed local server's port.** They collide at 7865, and
  on Windows the collision does not fail — both bind, and requests silently go to the wrong
  server. A regression here is invisible on Linux and catastrophic on Windows.

* **DOCKER_HOST parsing.** Deciding we can speak to a daemon we can't (npipe://, ssh://)
  turns a clean "one-click unavailable, use compose" into a confusing crash.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import docker_sidecar as ds  # noqa: E402


class PortsDoNotCollide(unittest.TestCase):
    def test_published_port_is_not_the_local_servers(self):
        import demucs_server
        self.assertNotEqual(
            ds.DEFAULT_PORT, demucs_server.DEFAULT_PORT,
            "the sidecar must not publish onto the managed local server's port — on "
            "Windows that collision does not fail, it silently serves the wrong server")

    def test_in_container_port_is_the_servers_own(self):
        # Inside the container the server always listens on 7865; only the HOST-side
        # published port moves. Conflating the two is what caused the collision.
        import demucs_server
        self.assertEqual(ds.SERVER_PORT, demucs_server.DEFAULT_PORT)


class PortBindingIsLoopbackOnly(unittest.TestCase):
    """Docker's default HostIp is 0.0.0.0. Ours must not be."""

    def _binding(self, port=7866):
        spec = ds._container_spec(port=port, gpu=False, image="img", networks=[])
        return spec["HostConfig"]["PortBindings"][f"{ds.SERVER_PORT}/tcp"][0]

    def test_binds_to_loopback(self):
        self.assertEqual(self._binding()["HostIp"], "127.0.0.1")

    def test_binds_the_requested_port(self):
        self.assertEqual(self._binding(9999)["HostPort"], "9999")

    def test_compose_snippet_is_loopback_only(self):
        snippet = ds.compose_snippet(port=7866)
        self.assertIn('"127.0.0.1:7866:7865"', snippet)
        # A bare "7866:7865" is 0.0.0.0 — the LAN. It must not appear.
        self.assertNotIn('- "7866:7865"', snippet)


class ComposeSnippet(unittest.TestCase):
    def test_pins_the_model_feedback_actually_uses(self):
        # Warmup only prefetches the DEFAULT model, so getting this wrong means the first
        # split pays a cold multi-GB download at the worst possible moment.
        self.assertIn(f"SLOPSMITH_DEMUCS_MODEL={ds.DEFAULT_MODEL}", ds.compose_snippet())

    def test_persists_the_model_cache(self):
        s = ds.compose_snippet()
        self.assertIn(f"{ds.CACHE_VOLUME}:/app/cache", s)
        self.assertIn("volumes:", s)

    def test_gpu_off_by_default_but_documented(self):
        s = ds.compose_snippet(gpu=False)
        self.assertIn("# gpus: all", s)          # commented, with the requirements noted
        self.assertNotIn("\n    gpus: all", s)

    def test_gpu_on_when_asked(self):
        self.assertIn("    gpus: all", ds.compose_snippet(gpu=True))


class DockerHostParsing(unittest.TestCase):
    def _with_env(self, value):
        env = dict(os.environ)
        if value is None:
            env.pop("DOCKER_HOST", None)
        else:
            env["DOCKER_HOST"] = value
        return mock.patch.dict(os.environ, env, clear=True)

    def test_env_wins(self):
        with self._with_env("tcp://1.2.3.4:2375"):
            self.assertEqual(ds.docker_host(), "tcp://1.2.3.4:2375")

    def test_unix_scheme_accepted(self):
        with self._with_env("unix:///var/run/docker.sock"):
            self.assertEqual(ds.docker_host(), "unix:///var/run/docker.sock")

    def test_npipe_is_refused(self):
        # Windows named pipe: the stdlib cannot speak it. Claiming we can turns a clean
        # "use compose instead" into a crash.
        with self._with_env("npipe:////./pipe/docker_engine"):
            self.assertIsNone(ds.docker_host())

    def test_ssh_is_refused(self):
        with self._with_env("ssh://user@host"):
            self.assertIsNone(ds.docker_host())

    def test_no_env_and_no_socket_is_none(self):
        with self._with_env(None), mock.patch.object(ds.Path, "exists", return_value=False):
            self.assertIsNone(ds.docker_host())


class GpuRuntimeDetection(unittest.TestCase):
    """`" ".join(a_string)` spaces out every character, so the DefaultRuntime check used to
    be dead code — it could never match, silently hiding the GPU option."""

    def _info(self, body):
        return mock.patch.object(ds, "_request", return_value=(200, body))

    def test_nvidia_in_runtimes(self):
        with self._info({"Runtimes": {"runc": {}, "nvidia": {}}, "DefaultRuntime": "runc"}):
            self.assertTrue(ds.gpu_available())

    def test_nvidia_as_default_runtime_only(self):
        with self._info({"Runtimes": {"runc": {}}, "DefaultRuntime": "nvidia"}):
            self.assertTrue(ds.gpu_available())

    def test_no_nvidia_anywhere(self):
        with self._info({"Runtimes": {"runc": {}}, "DefaultRuntime": "runc"}):
            self.assertFalse(ds.gpu_available())

    def test_daemon_unreachable_is_false_not_an_exception(self):
        with mock.patch.object(ds, "_request", side_effect=RuntimeError("no daemon")):
            self.assertFalse(ds.gpu_available())


if __name__ == "__main__":
    unittest.main()

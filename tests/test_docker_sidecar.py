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


class ImageRefParsing(unittest.TestCase):
    """A naive rpartition(':') gets two perfectly valid forms wrong, and both are forms a
    real deployment would use: a private registry (host:port) and a digest pin."""

    def test_normal_tagged_ref(self):
        self.assertEqual(ds.split_ref("ghcr.io/got-feedback/feedback-demucs-server:latest"),
                         ("ghcr.io/got-feedback/feedback-demucs-server", "latest"))

    def test_no_tag_defaults_to_latest(self):
        self.assertEqual(ds.split_ref("busybox"), ("busybox", "latest"))

    def test_registry_with_a_port_and_no_tag(self):
        # The colon belongs to the PORT. rpartition would have given
        # ("registry.local", "5000/repo") and the pull would fail bizarrely.
        self.assertEqual(ds.split_ref("registry.local:5000/repo"),
                         ("registry.local:5000/repo", "latest"))

    def test_registry_with_a_port_and_a_tag(self):
        self.assertEqual(ds.split_ref("registry.local:5000/repo:v2"),
                         ("registry.local:5000/repo", "v2"))

    def test_digest_pin(self):
        # What a security-conscious deployment sets STEM_SPLITTER_SIDECAR_IMAGE to.
        self.assertEqual(ds.split_ref("repo@sha256:abc123"), ("repo", "sha256:abc123"))

    def test_digest_pin_with_registry(self):
        self.assertEqual(ds.split_ref("ghcr.io/o/r@sha256:deadbeef"),
                         ("ghcr.io/o/r", "sha256:deadbeef"))

    def test_local_tag(self):
        self.assertEqual(ds.split_ref("demucs-permfix:test"), ("demucs-permfix", "test"))


class ReachabilityFromInsideAContainer(unittest.TestCase):
    """Publishing to 127.0.0.1 on the HOST is useless to a containerized feedBack: that
    loopback is *this container*. Starting a server nobody can reach is worse than not
    starting one — it looks like success."""

    def _ctx(self, containerized, host_net=False, networks=()):
        return (
            mock.patch("demucs_server.in_container", return_value=containerized),
            mock.patch.object(ds, "_self_uses_host_networking", return_value=host_net),
            mock.patch.object(ds, "_self_networks", return_value=list(networks)),
        )

    def _run(self, fn, containerized, host_net=False, networks=()):
        a, b, c = self._ctx(containerized, host_net, networks)
        with a, b, c:
            return fn()

    def test_on_the_host_loopback_is_fine(self):
        self.assertEqual(self._run(ds._reachability_problem, containerized=False), "")
        self.assertEqual(self._run(lambda: ds.url_for(7866, False), containerized=False),
                         "http://127.0.0.1:7866")

    def test_host_networking_makes_loopback_valid(self):
        self.assertEqual(
            self._run(ds._reachability_problem, containerized=True, host_net=True), "")
        self.assertEqual(
            self._run(lambda: ds.url_for(7866, False), containerized=True, host_net=True),
            "http://127.0.0.1:7866")

    def test_shared_user_defined_network_is_reachable_by_name(self):
        self.assertEqual(
            self._run(ds._reachability_problem, containerized=True, networks=["appnet"]), "")
        self.assertEqual(
            self._run(lambda: ds.url_for(7866, True), containerized=True, networks=["appnet"]),
            f"http://{ds.CONTAINER_NAME}:{ds.SERVER_PORT}")

    def test_bridge_only_container_is_refused_not_silently_broken(self):
        # The trap: default bridge has no name DNS, and the published port is on the HOST's
        # loopback. A one-click here would "succeed" and produce an unusable server.
        problem = self._run(ds._reachability_problem, containerized=True, networks=[])
        self.assertTrue(problem)
        self.assertIn("bridge", problem)
        self.assertIn("compose", problem)      # tells them what to do instead

    def test_bridge_only_container_gets_no_url_rather_than_a_wrong_one(self):
        url = self._run(lambda: ds.url_for(7866, False), containerized=True, networks=[])
        self.assertIsNone(
            url, "returning http://127.0.0.1:7866 from a bridge-only container points at "
                 "the app container itself — autodetection and the health check would fail "
                 "with a confusing timeout instead of an honest error")


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

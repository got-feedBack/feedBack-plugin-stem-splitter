"""Route-level tests for the Docker sidecar endpoints.

These exist because of a pattern, not a hunch. Two of the worst bugs in this feature were
in the ROUTE while the underlying helper was perfectly correct, so testing the helper — or
even driving ``docker_sidecar.up()`` by hand against a live daemon, which I did — proved
nothing about what the button actually does:

* ``/sidecar/up`` defaulted the published port via ``_as_port(body.get("port"))``, whose
  fallback is the MANAGED LOCAL SERVER's port (7865). The UI never sends a port, so every
  one-click published onto exactly the port this feature goes out of its way to avoid — and
  on Windows that collision doesn't fail, it silently serves from the wrong server.

* ``/sidecar_status`` generated the compose snippet with ``gpu=gpu_available``, turning the
  *recommended* copy-paste path GPU-on by default. A daemon can advertise an nvidia runtime
  on a host with no usable GPU, and then ``docker compose up`` fails on the very command we
  told the user to run.

Both are pure wiring. Neither is visible from the library, and both are caught here.

No Docker daemon required: ``docker_sidecar`` is stubbed.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import demucs_server  # noqa: E402
import docker_sidecar  # noqa: E402
import routes  # noqa: E402

P = "/api/plugins/stem_splitter"


class SidecarRoutes(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        app = FastAPI()
        # setup() spawns a worker thread and an autostart probe; neither touches Docker.
        routes.setup(app, {"config_dir": self._tmp.name})
        self.client = TestClient(app)

    def tearDown(self):
        self._tmp.cleanup()

    # ── the compose snippet must be the one that reliably comes up ────────────
    def test_compose_snippet_is_cpu_by_default_even_when_a_gpu_is_available(self):
        fake = {"docker": True, "in_container": False, "running": False, "url": None,
                "gpu_available": True, "container": None, "reason": "", "port": 7866}
        with mock.patch.object(docker_sidecar, "status", return_value=dict(fake)):
            body = self.client.get(f"{P}/sidecar_status").json()

        compose = body["compose"]
        self.assertIn("# gpus: all", compose,
                      "the GPU line must be present but COMMENTED, with its requirements")
        self.assertNotIn("\n    gpus: all", compose,
                         "a daemon can advertise an nvidia runtime on a host with no usable "
                         "GPU; enabling it by default breaks `docker compose up` on the very "
                         "command we told the user to run")

    def test_compose_snippet_is_offered_even_with_no_docker_daemon(self):
        # The compose path needs no daemon and no socket. It is the one we RECOMMEND, so it
        # must be visible precisely when the one-click is not.
        fake = {"docker": False, "in_container": True, "running": False, "url": None,
                "gpu_available": False, "container": None, "reason": "no socket"}
        with mock.patch.object(docker_sidecar, "status", return_value=dict(fake)):
            body = self.client.get(f"{P}/sidecar_status").json()
        self.assertIn("image:", body["compose"])
        self.assertTrue(body["in_container"])

    def test_compose_snippet_binds_loopback_only(self):
        # Security: a bare "7866:7865" is 0.0.0.0 — an unauthenticated inference server on
        # the whole LAN.
        with mock.patch.object(docker_sidecar, "status",
                               return_value={"docker": True, "in_container": False,
                                             "running": False, "gpu_available": False,
                                             "container": None, "reason": "", "url": None}):
            compose = self.client.get(f"{P}/sidecar_status").json()["compose"]
        self.assertIn("127.0.0.1:", compose)

    # ── the port the button actually publishes on ─────────────────────────────
    def test_up_publishes_on_the_sidecar_port_not_the_local_servers(self):
        """THE regression test. The UI posts {"gpu": ...} with no port."""
        seen = {}

        def fake_up(port, gpu, progress_cb=None, **kw):
            seen["port"] = port
            return {"running": True}

        with mock.patch.object(docker_sidecar, "up", side_effect=fake_up):
            r = self.client.post(f"{P}/sidecar/up", json={"gpu": False})
        self.assertEqual(r.status_code, 200)

        # run_server_op runs the op on a daemon thread; give it a moment to land.
        for _ in range(50):
            if "port" in seen:
                break
            import time
            time.sleep(0.05)

        self.assertEqual(seen.get("port"), docker_sidecar.DEFAULT_PORT)
        self.assertNotEqual(
            seen.get("port"), demucs_server.DEFAULT_PORT,
            "publishing onto the managed local server's port is the exact collision this "
            "feature exists to avoid, and on Windows it does not even fail — both bind, and "
            "requests silently go to the wrong server")

    def test_up_honours_an_explicit_port(self):
        seen = {}
        with mock.patch.object(docker_sidecar, "up",
                               side_effect=lambda port, gpu, progress_cb=None, **kw:
                               seen.setdefault("port", port) or {"running": True}):
            self.client.post(f"{P}/sidecar/up", json={"gpu": False, "port": 9001})
        for _ in range(50):
            if "port" in seen:
                break
            import time
            time.sleep(0.05)
        self.assertEqual(seen.get("port"), 9001)


if __name__ == "__main__":
    unittest.main()

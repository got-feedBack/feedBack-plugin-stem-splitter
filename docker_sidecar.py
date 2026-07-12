"""Run the demucs server as a SIBLING CONTAINER, driven through the Docker socket.

Why this exists
---------------
A containerized feedBack cannot install the demucs server on its host: ``Popen`` forks a
child into the *caller's* namespaces, and there is no syscall for "run this over there".
The host's filesystem isn't visible, and even if it were, executing it would still land
inside our own container. Docker users therefore had two options, both manual: hand-write
a compose service, or run the server themselves and paste a URL.

The Docker socket is the one escape hatch that actually exists. If the user has mounted
``/var/run/docker.sock`` (Portainer users routinely do), we can ask the HOST's daemon to
run the published image as a sibling container — a real container on the real host, with
real GPU access, and no Python on the host at all. That is as close to the Electron
one-click experience as is physically available to a container.

What this is NOT
----------------
This does not "install the server outside Docker". Nothing can. It asks the host daemon to
run a container next to us.

Security posture
----------------
**The Docker socket is root-equivalent on the host.** We do not ask for it, we do not
enable it, and we never mount it ourselves — we only *use* it if the user has already
chosen to expose it. Every call here is behind an explicit button press, and the only
container we will ever create is the pinned image below with a fixed spec. There is no
route that takes an arbitrary image, command, or bind mount from the caller.

Transports
----------
* ``unix:///var/run/docker.sock`` — the real target (Linux/macOS hosts, and any container
  with the socket mounted).
* ``tcp://host:port`` via ``DOCKER_HOST`` — a remote/exposed daemon. Also the only way to
  exercise this on a Windows dev box.
* ``npipe://`` (Docker Desktop on Windows) — NOT supported: it's a named pipe and the
  stdlib cannot speak it. Callers fall back to showing the compose snippet.

No third-party deps (no `docker` SDK): a plugin cannot add packages to the host app, and
a one-click feature that first needs a pip install isn't one click.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import urllib.parse
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("feedBack.plugin.stem_splitter")

ProgressCB = Optional[Callable[[dict], None]]

# The image published by got-feedBack/feedBack-demucs-server (public on GHCR).
DEFAULT_IMAGE = os.environ.get(
    "STEM_SPLITTER_SIDECAR_IMAGE",
    "ghcr.io/got-feedback/feedback-demucs-server:latest",
)
CONTAINER_NAME = "feedback-demucs-server"
CACHE_VOLUME = "feedback-demucs-cache"      # model weights (~1.5 GB) survive recreation
DEFAULT_PORT = 7865
DEFAULT_MODEL = "bs_roformer_sw"            # warmup only prefetches the DEFAULT model

_API = "v1.41"                              # widely supported; avoids negotiating


# ── transport ────────────────────────────────────────────────────────────────

class _UnixHTTPConnection(http.client.HTTPConnection):
    """http.client over an AF_UNIX socket — how you talk to /var/run/docker.sock."""

    def __init__(self, path: str, timeout: float = 30.0):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self._path)
        self.sock = s


def docker_host() -> str | None:
    """The daemon we'd talk to, or None if there's nothing we can speak to.

    Honours DOCKER_HOST first (that's the convention, and it's how a Windows dev box or a
    remote daemon is reachable), then the well-known socket path.
    """
    env = (os.environ.get("DOCKER_HOST") or "").strip()
    if env:
        if env.startswith(("unix://", "tcp://", "http://")):
            return env
        return None          # npipe:// (Windows) and ssh:// are not speakable here
    sock = Path("/var/run/docker.sock")
    try:
        if sock.exists():
            return "unix:///var/run/docker.sock"
    except OSError:
        pass
    return None


def _connect(timeout: float = 30.0) -> http.client.HTTPConnection:
    host = docker_host()
    if not host:
        raise RuntimeError("no reachable Docker daemon")
    if host.startswith("unix://"):
        return _UnixHTTPConnection(host[len("unix://"):], timeout=timeout)
    u = urllib.parse.urlsplit(host.replace("tcp://", "http://", 1))
    return http.client.HTTPConnection(u.hostname or "localhost", u.port or 2375,
                                      timeout=timeout)


def _request(method: str, path: str, body: dict | None = None,
             timeout: float = 30.0) -> tuple[int, object]:
    """One Docker API call. Returns (status, parsed-json-or-text)."""
    conn = _connect(timeout)
    try:
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        conn.request(method, f"/{_API}{path}", body=payload, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", "replace")
        try:
            return resp.status, json.loads(raw) if raw.strip() else None
        except ValueError:
            return resp.status, raw
    finally:
        conn.close()


def _stream(method: str, path: str, progress_cb: ProgressCB = None,
            timeout: float = 1800.0) -> None:
    """A streaming call (image pull). Docker emits one JSON object per line."""
    conn = _connect(timeout)
    try:
        conn.request(method, f"/{_API}{path}")
        resp = conn.getresponse()
        if resp.status >= 400:
            raise RuntimeError(f"docker: HTTP {resp.status}: "
                               f"{resp.read().decode('utf-8', 'replace')[:300]}")
        # Docker does not send a trailing newline on the last chunk, so read by line
        # and tolerate a partial tail.
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("error"):
                raise RuntimeError(f"docker: {ev['error']}")
            if progress_cb:
                progress_cb(ev)
    finally:
        conn.close()


def ping() -> tuple[bool, str]:
    """Can we talk to a daemon? (ok, reason)."""
    host = docker_host()
    if not host:
        if os.name == "nt":
            return (False, "Docker Desktop on Windows exposes a named pipe, which this "
                           "cannot speak. Enable 'Expose daemon on tcp://localhost:2375' "
                           "in Docker Desktop settings, or use the compose snippet.")
        return (False, "No Docker socket found. Mount /var/run/docker.sock into this "
                       "container (or set DOCKER_HOST) to enable one-click setup.")
    try:
        status, body = _request("GET", "/_ping", timeout=5.0)
    except Exception as e:
        return (False, f"Could not reach the Docker daemon at {host}: {e}")
    if status != 200:
        return (False, f"Docker daemon at {host} answered HTTP {status}")
    return (True, "")


# ── our own container (so the sidecar is reachable by name) ──────────────────

def _self_container_id() -> str | None:
    """Our own container id, if we're in one.

    The container's hostname IS its short id by default, and Docker's API accepts a short
    id wherever it accepts a name — so this is enough to look ourselves up.
    """
    try:
        from demucs_server import in_container
        if not in_container():
            return None
    except Exception:
        return None
    hid = socket.gethostname().strip()
    return hid or None


def _self_networks() -> list[str]:
    """The networks OUR container is on.

    Attaching the sidecar to one of them is what makes `http://feedback-demucs-server:7865`
    resolve from inside the app container. Relying on a published host port instead would
    mean guessing at host.docker.internal (Desktop-only) or the gateway IP, both of which
    are exactly the kind of thing that works on the maintainer's machine and nowhere else.
    """
    cid = _self_container_id()
    if not cid:
        return []
    try:
        status, body = _request("GET", f"/containers/{cid}/json", timeout=5.0)
    except Exception:
        return []
    if status != 200 or not isinstance(body, dict):
        return []
    nets = ((body.get("NetworkSettings") or {}).get("Networks") or {})
    # "bridge" is the default network: joining it does NOT give name-based DNS
    # (only user-defined networks do), so it's useless for our purpose.
    return [n for n in nets if n != "bridge"]


# ── lifecycle ────────────────────────────────────────────────────────────────

def _find_container() -> dict | None:
    """Our sidecar, running or not."""
    f = urllib.parse.quote(json.dumps({"name": [CONTAINER_NAME]}))
    try:
        status, body = _request("GET", f"/containers/json?all=1&filters={f}", timeout=10.0)
    except Exception:
        return None
    if status != 200 or not isinstance(body, list):
        return None
    # The name filter is a substring match, so confirm an exact hit.
    for c in body:
        if any(n.lstrip("/") == CONTAINER_NAME for n in (c.get("Names") or [])):
            return c
    return None


def gpu_available() -> bool:
    """Does the daemon advertise the nvidia runtime?

    Checked against the DAEMON, not our own container: the toolkit lives on the host, and
    a GPU request against a daemon without it fails at container-create with a confusing
    error rather than falling back to CPU.
    """
    try:
        status, body = _request("GET", "/info", timeout=10.0)
    except Exception:
        return False
    if status != 200 or not isinstance(body, dict):
        return False
    if "nvidia" in (body.get("Runtimes") or {}):
        return True
    return "nvidia" in " ".join(body.get("DefaultRuntime", "") or "")


def _container_spec(port: int, gpu: bool, image: str, networks: list[str]) -> dict:
    env = [
        f"PORT={DEFAULT_PORT}",
        "HOST=0.0.0.0",
        f"SLOPSMITH_DEMUCS_MODEL={DEFAULT_MODEL}",
        "SKIP_WARMUP=false",
        "SLOPSMITH_DEMUCS_CACHE=/app/cache",
        "TORCH_HOME=/app/cache/torch",
        "HF_HOME=/app/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE=/app/cache/huggingface/hub",
    ]
    host_config: dict = {
        # Publish to the host as well as attaching to our network: an Electron user (not
        # in a container) reaches it on 127.0.0.1, and a human can curl it either way.
        "PortBindings": {f"{DEFAULT_PORT}/tcp": [{"HostPort": str(port)}]},
        "Binds": [f"{CACHE_VOLUME}:/app/cache"],
        "RestartPolicy": {"Name": "unless-stopped"},
    }
    if gpu:
        # The modern equivalent of `--gpus all`. Needs nvidia-container-toolkit on the
        # HOST; we only set it when the daemon says it has the runtime.
        host_config["DeviceRequests"] = [
            {"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}
        ]
    spec: dict = {
        "Image": image,
        "Env": env,
        "ExposedPorts": {f"{DEFAULT_PORT}/tcp": {}},
        "HostConfig": host_config,
        "Labels": {
            "org.feedback.managed-by": "stem_splitter",
            "org.feedback.role": "demucs-server",
        },
    }
    if networks:
        spec["NetworkingConfig"] = {
            "EndpointsConfig": {networks[0]: {"Aliases": [CONTAINER_NAME]}}
        }
    return spec


def pull_image(image: str = DEFAULT_IMAGE, progress_cb: ProgressCB = None) -> None:
    """Pull the image. ~4.8 GB compressed — an explicit-click operation, never implicit."""
    ref, _, tag = image.rpartition(":")
    if not ref:                       # no tag given
        ref, tag = image, "latest"
    q = urllib.parse.urlencode({"fromImage": ref, "tag": tag})

    layers: dict[str, tuple[int, int]] = {}

    def on_event(ev: dict) -> None:
        pid = ev.get("id")
        det = ev.get("progressDetail") or {}
        if pid and det.get("total"):
            layers[pid] = (int(det.get("current") or 0), int(det["total"]))
        done = sum(c for c, _ in layers.values())
        total = sum(t for _, t in layers.values())
        if progress_cb:
            progress_cb({
                "line": f"{ev.get('status', '')} {pid or ''}".strip(),
                # Cap at 0.9: layers are still being extracted after the bytes land, and a
                # bar that sits at 100% while the user waits is worse than one that doesn't.
                "pct": min(0.9, done / total) if total else None,
                "phase": "Pulling image",
            })

    _stream("POST", f"/images/create?{q}", progress_cb=on_event)


def up(port: int = DEFAULT_PORT, gpu: bool = False, image: str = DEFAULT_IMAGE,
       progress_cb: ProgressCB = None) -> dict:
    """Pull (if needed), create and start the sidecar. Idempotent."""
    ok, reason = ping()
    if not ok:
        raise RuntimeError(reason)

    existing = _find_container()
    if existing and existing.get("State") == "running":
        return status()

    if existing:
        # A stopped container may have been created with a stale spec (different port,
        # no GPU, an older image). Recreate rather than start it and hope.
        _request("DELETE", f"/containers/{existing['Id']}?force=1&v=0", timeout=60.0)

    if progress_cb:
        progress_cb({"line": f"Pulling {image} (~4.8 GB)…", "pct": 0.02,
                     "phase": "Pulling image"})
    pull_image(image, progress_cb)

    want_gpu = bool(gpu) and gpu_available()
    if gpu and not want_gpu:
        if progress_cb:
            progress_cb({"line": "The Docker daemon has no nvidia runtime — starting on "
                                 "CPU. (Install nvidia-container-toolkit on the host for "
                                 "GPU.)", "pct": 0.92, "phase": "Starting"})

    nets = _self_networks()
    spec = _container_spec(port, want_gpu, image, nets)

    if progress_cb:
        progress_cb({"line": "Creating the container…", "pct": 0.93, "phase": "Starting"})
    st, body = _request("POST", f"/containers/create?name={CONTAINER_NAME}", spec,
                        timeout=60.0)
    if st not in (200, 201) or not isinstance(body, dict) or not body.get("Id"):
        raise RuntimeError(f"could not create the container: {body}")
    cid = body["Id"]

    st, body = _request("POST", f"/containers/{cid}/start", timeout=60.0)
    if st not in (204, 304):
        raise RuntimeError(f"could not start the container: {body}")

    if progress_cb:
        progress_cb({"line": "Started.", "pct": 1.0, "phase": "Done"})
    return status()


def down(remove: bool = False) -> dict:
    """Stop the sidecar. `remove` also deletes the container (the model cache VOLUME is
    kept either way — re-pulling 1.5 GB of weights because someone clicked Stop would be
    a cruel default)."""
    c = _find_container()
    if not c:
        return status()
    _request("POST", f"/containers/{c['Id']}/stop?t=10", timeout=60.0)
    if remove:
        _request("DELETE", f"/containers/{c['Id']}?v=0", timeout=60.0)
    return status()


def url_for(port: int, networks_shared: bool) -> str:
    """How WE reach the sidecar.

    From inside a container that shares a user-defined network with it, by name (Docker's
    embedded DNS). Otherwise via the published host port on loopback.
    """
    if networks_shared:
        return f"http://{CONTAINER_NAME}:{DEFAULT_PORT}"
    return f"http://127.0.0.1:{port}"


def status() -> dict:
    """Everything the settings UI needs to render the sidecar card."""
    ok, reason = ping()
    out: dict = {
        "docker": ok,
        "reason": reason,
        "host": docker_host(),
        "in_container": False,
        "image": DEFAULT_IMAGE,
        "container": None,
        "running": False,
        "url": None,
        "gpu_available": False,
    }
    try:
        from demucs_server import in_container
        out["in_container"] = in_container()
    except Exception:
        pass
    if not ok:
        return out

    out["gpu_available"] = gpu_available()
    c = _find_container()
    if not c:
        return out

    running = c.get("State") == "running"
    port = DEFAULT_PORT
    for p in c.get("Ports") or []:
        if p.get("PrivatePort") == DEFAULT_PORT and p.get("PublicPort"):
            port = int(p["PublicPort"])
            break
    shared = bool(set(_self_networks()) &
                  set((c.get("NetworkSettings") or {}).get("Networks") or {}))
    out.update({
        "container": {"id": c.get("Id", "")[:12], "state": c.get("State"),
                      "status": c.get("Status"), "image": c.get("Image")},
        "running": running,
        "port": port,
        "url": url_for(port, shared) if running else None,
    })
    return out


def compose_snippet(port: int = DEFAULT_PORT, gpu: bool = False) -> str:
    """The manual path — always shown, because it needs no socket and no trust.

    This is the option we RECOMMEND for Docker users: adding a service to the compose file
    they already have costs nothing and grants nobody root on their host.
    """
    gpu_lines = "    gpus: all\n" if gpu else "    # gpus: all   # needs nvidia-container-toolkit (Linux/WSL2 only)\n"
    return (
        "  demucs:\n"
        f"    image: {DEFAULT_IMAGE}\n"
        f"    container_name: {CONTAINER_NAME}\n"
        "    restart: unless-stopped\n"
        "    ports:\n"
        f"      - \"{port}:{DEFAULT_PORT}\"\n"
        "    volumes:\n"
        f"      - {CACHE_VOLUME}:/app/cache\n"
        "    environment:\n"
        f"      - SLOPSMITH_DEMUCS_MODEL={DEFAULT_MODEL}\n"
        "      - SLOPSMITH_DEMUCS_CACHE=/app/cache\n"
        "      - TORCH_HOME=/app/cache/torch\n"
        "      - HF_HOME=/app/cache/huggingface\n"
        f"{gpu_lines}"
        "\n"
        "volumes:\n"
        f"  {CACHE_VOLUME}:\n"
    )

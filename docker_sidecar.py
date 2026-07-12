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
import time
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

# The port the server listens on INSIDE the container. Fixed; nothing else is in there.
SERVER_PORT = 7865

# The port we publish it on. Deliberately NOT 7865: that is the managed local server's
# port (demucs_server.DEFAULT_PORT), and a user can plausibly have both — a local server
# they installed earlier, and a sidecar they start now.
#
# The collision is nastier than "one of them fails to bind":
#   * Linux  — `docker run` fails with "address already in use". Loud, at least.
#   * Windows — BOTH bind successfully, because Docker publishes on 0.0.0.0 while the
#     local server listens on 127.0.0.1, and a loopback connection goes to the MORE
#     SPECIFIC bind. So the container starts, reports healthy, and every request silently
#     goes to the other server. That is how this was found: /health came back describing a
#     Windows cache_dir from inside a Linux container.
#
# Publishing on a different port sidesteps it entirely, and _find_container() reads the
# real published port back out of Docker rather than assuming.
DEFAULT_PORT = 7866

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
            # Do NOT tell people to switch on the TCP daemon. It is an UNAUTHENTICATED,
            # root-equivalent endpoint — anything that can reach localhost can then create
            # containers and mount the filesystem — and switching it on alone wouldn't even
            # work here, since docker_host() reads DOCKER_HOST, not the Docker Desktop
            # setting. Recommending a security downgrade that doesn't function is the worst
            # of both. The compose path needs none of this.
            return (False, "One-click setup isn't available on Windows: Docker Desktop "
                           "exposes a named pipe, which this can't speak. Use the compose "
                           "service above — it does the same thing. (Advanced: setting "
                           "DOCKER_HOST to a reachable daemon also works, but exposing the "
                           "Docker API is root-equivalent on that machine.)")
        # Deliberately NOT "mount the docker socket to enable this". The socket is
        # root-equivalent on the host; suggesting it to get a convenience button is bad
        # advice, and the compose snippet reaches the same end state without it.
        return (False, "One-click setup isn't available (no Docker socket is mounted). "
                       "Use the compose service above — it does the same thing and needs "
                       "no daemon access.")
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


def _self_uses_host_networking() -> bool:
    """Are we on the host's network stack (`--network host`)?

    The one case where loopback genuinely works from inside a container: there is no network
    namespace of our own, so 127.0.0.1 IS the host's 127.0.0.1 and the published port is
    directly reachable.
    """
    cid = _self_container_id()
    if not cid:
        return False
    try:
        status, body = _request("GET", f"/containers/{cid}/json", timeout=5.0)
    except Exception:
        return False
    if status != 200 or not isinstance(body, dict):
        return False
    mode = str(((body.get("HostConfig") or {}).get("NetworkMode") or "")).lower()
    return mode == "host"


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
    # NOT " ".join(...): DefaultRuntime is a STRING, and joining it spaces out every
    # character ("n v i d i a"), so the substring check could never match. That silently
    # hid the GPU option on any daemon that sets DefaultRuntime=nvidia without listing it
    # under Runtimes.
    return "nvidia" in str(body.get("DefaultRuntime") or "").lower()


def _container_spec(port: int, gpu: bool, image: str, networks: list[str]) -> dict:
    env = [
        f"PORT={SERVER_PORT}",
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
        #
        # HostIp=127.0.0.1 is LOAD-BEARING. Docker's default is 0.0.0.0, which would put an
        # unauthenticated inference server on EVERY interface of the host — visible to the
        # whole LAN — when nothing here needs more than loopback: Electron reaches it on
        # localhost, and a containerized feedBack reaches it by container NAME over the
        # shared network, not through the published port at all.
        "PortBindings": {f"{SERVER_PORT}/tcp": [{"HostIp": "127.0.0.1",
                                                 "HostPort": str(port)}]},
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
        "ExposedPorts": {f"{SERVER_PORT}/tcp": {}},
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


def image_exists(image: str) -> bool:
    """Is the image already on the daemon?"""
    try:
        status, _ = _request("GET", f"/images/{urllib.parse.quote(image, safe='')}/json",
                             timeout=10.0)
    except Exception:
        return False
    return status == 200


def split_ref(image: str) -> tuple[str, str]:
    """Split an image reference into (name, tag-or-digest) the way Docker does.

    A naive ``rpartition(":")`` gets two perfectly valid forms wrong:

      * ``registry.local:5000/repo``  — the colon belongs to the REGISTRY PORT, not a tag.
        Splitting on it yields name=``registry.local`` tag=``5000/repo``, and the pull fails
        with something baffling. Anyone running a private registry hits this immediately.
      * ``repo@sha256:abc…``          — a digest pin, which is exactly what a
        security-conscious deployment would set STEM_SPLITTER_SIDECAR_IMAGE to. The ``:``
        inside the digest gets treated as a tag separator.

    The rule Docker uses: a colon is a tag separator only if it appears AFTER the last
    slash (i.e. it's in the final path component, not in a ``host:port``).
    """
    if "@" in image:                      # digest pin wins; it can contain ':'
        name, _, digest = image.partition("@")
        return name, digest
    colon = image.rfind(":")
    if colon > image.rfind("/"):          # the colon is in the last path component -> a tag
        return image[:colon], image[colon + 1:]
    return image, "latest"


def pull_image(image: str = DEFAULT_IMAGE, progress_cb: ProgressCB = None) -> None:
    """Pull the image. ~4.8 GB compressed — an explicit-click operation, never implicit."""
    ref, tag = split_ref(image)
    q = urllib.parse.urlencode({"fromImage": ref, "tag": tag})

    # Docker reports progress PER LAYER, and a layer is downloaded and then extracted.
    # Track both: this is a ~4.8 GB pull, and extraction of the CUDA-torch layer alone
    # takes minutes. A bar that shows only downloads sits at "100%" (or, worse, at a
    # capped 90%) through all of it, which reads exactly like a hang — the single most
    # common way a long-running install gets killed by an impatient user.
    dl: dict[str, tuple[int, int]] = {}
    ex: dict[str, tuple[int, int]] = {}

    def _gb(n: float) -> str:
        return f"{n / 1e9:.1f} GB"

    def on_event(ev: dict) -> None:
        pid = ev.get("id") or ""
        status = (ev.get("status") or "").strip()
        det = ev.get("progressDetail") or {}
        cur, tot = int(det.get("current") or 0), int(det.get("total") or 0)

        if status.startswith("Downloading") and pid and tot:
            dl[pid] = (cur, tot)
        elif status.startswith("Download complete") and pid and pid in dl:
            dl[pid] = (dl[pid][1], dl[pid][1])
        elif status.startswith("Extracting") and pid and tot:
            ex[pid] = (cur, tot)
        elif status.startswith("Pull complete") and pid and pid in ex:
            ex[pid] = (ex[pid][1], ex[pid][1])

        d_done, d_tot = sum(c for c, _ in dl.values()), sum(t for _, t in dl.values())
        e_done, e_tot = sum(c for c, _ in ex.values()), sum(t for _, t in ex.values())

        # Download is the bulk of the wall-clock; give it 80% of the bar and extraction
        # the rest, so BOTH phases visibly move.
        pct = None
        phase = "Pulling image"
        line = f"{status} {pid}".strip()
        if d_tot:
            pct = 0.8 * (d_done / d_tot)
            line = f"downloading… {_gb(d_done)} / {_gb(d_tot)}"
        if e_tot and d_tot and d_done >= d_tot:
            phase = "Extracting"
            pct = 0.8 + 0.18 * (e_done / e_tot)
            line = f"extracting… {_gb(e_done)} / {_gb(e_tot)}"
        if progress_cb:
            progress_cb({"line": line, "pct": pct, "phase": phase})

    _stream("POST", f"/images/create?{q}", progress_cb=on_event)


def demucs_server_default_port() -> int:
    """The managed LOCAL server's port — the thing we most plausibly collide with."""
    try:
        import demucs_server
        return int(demucs_server.DEFAULT_PORT)
    except Exception:
        return 7865


# Our container always reports this, because we set SLOPSMITH_DEMUCS_CACHE=/app/cache in
# its env. A server running on the HOST reports a host path (a Windows path, a config dir),
# never this. It is the cheapest reliable "is this actually my container?" discriminator
# available without adding an endpoint to the server.
_OUR_CACHE_DIR = "/app/cache"


def _assert_port_is_ours(port: int, progress_cb: ProgressCB = None) -> None:
    """Refuse to hand back a URL that belongs to somebody else's server.

    See the note at DEFAULT_PORT: on Windows a port collision does NOT fail the bind, and
    the result is a container that looks healthy while every request goes to a different
    server. Silently splitting against the wrong server is far worse than failing to start.

    Only meaningful when we are on the HOST. Inside a container, 127.0.0.1 is *this
    container*, not the Docker host — the probe could never reach the published port, so it
    would add ~9s of pointless retries to every start and protect nothing. (There is also
    nothing to collide with there: a containerized app reaches the sidecar by container
    name, and cannot run a managed local server in the first place.)
    """
    import json as _json
    import urllib.request

    try:
        from demucs_server import in_container
        if in_container():
            return
    except Exception:
        pass

    url = f"http://127.0.0.1:{port}/health"
    for attempt in range(10):          # the server binds fast, but not instantly
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                health = _json.loads(r.read().decode("utf-8", "replace"))
            break
        except Exception:
            if attempt == 9:
                # Not reachable yet is not proof of a collision — warmup can be slow and
                # the caller polls status() anyway. Don't turn a slow start into an error.
                if progress_cb:
                    progress_cb({"line": f"Started, but nothing is answering on port {port} "
                                         f"yet — it may still be coming up.",
                                 "pct": 0.98, "phase": "Starting"})
                return
            time.sleep(1.0)
    else:
        return

    cache = str(health.get("cache_dir") or "")
    if cache and cache != _OUR_CACHE_DIR:
        raise RuntimeError(
            f"port {port} is already served by a DIFFERENT demucs server (it reports "
            f"cache_dir={cache!r}, ours is {_OUR_CACHE_DIR!r}). The container is running, "
            f"but every request to that port would go to the other server instead — so "
            f"this is being refused rather than silently splitting against the wrong one. "
            f"The usual cause is the plugin's own managed local server on port "
            f"{demucs_server_default_port()}. Stop it, or publish the container on another "
            f"port."
        )


def _reachability_problem() -> str:
    """Would a sidecar we start actually be reachable FROM HERE? '' if yes.

    Checked BEFORE pulling 4.8 GB, because the failure is silent otherwise: the container
    starts, Docker reports it healthy, and nothing can talk to it.

    From the host (Electron) the published port on 127.0.0.1 is always reachable. From
    inside a container it is NOT — that loopback is *this container*, not the Docker host —
    so we need one of:

      * host networking: we share the host's stack, so 127.0.0.1 really is the host's; or
      * a user-defined network we can attach the sidecar to, giving us Docker's embedded DNS
        and `http://feedback-demucs-server:7865`.

    The default `bridge` network gives NEITHER: it has no name-based DNS, and the published
    port is on the host's loopback, not ours. A one-click there would succeed and produce an
    unusable server — the worst outcome, because it looks like it worked.
    """
    try:
        from demucs_server import in_container
        if not in_container():
            return ""                      # host: loopback is genuinely ours
    except Exception:
        return ""

    if _self_uses_host_networking():
        return ""
    if _self_networks():
        return ""                          # a shared user-defined network: reachable by name

    return (
        "feedBack is in a container that is only on Docker's default 'bridge' network, so a "
        "server started here would be unreachable: the published port lands on the HOST's "
        "loopback (not this container's), and the default bridge has no name-based DNS. "
        "Use the compose service instead — compose puts both containers on a shared network, "
        "which is exactly what makes this work. (Or run feedBack on a user-defined network, "
        "or with host networking.)"
    )


def up(port: int = DEFAULT_PORT, gpu: bool = False, image: str = DEFAULT_IMAGE,
       progress_cb: ProgressCB = None) -> dict:
    """Pull (if needed), create and start the sidecar. Idempotent."""
    ok, reason = ping()
    if not ok:
        raise RuntimeError(reason)

    # Refuse BEFORE the 4.8 GB pull. Starting a server we cannot reach is worse than not
    # starting one: it looks like success.
    problem = _reachability_problem()
    if problem:
        raise RuntimeError(problem)

    existing = _find_container()
    if existing and existing.get("State") == "running":
        # Idempotent ONLY if the running container actually matches what was asked for.
        # Returning it regardless meant changing the port (or the image) in the UI and
        # pressing Start appeared to work and silently did nothing — the user is left
        # looking at a container that is running with the OLD settings.
        cur_port = None
        for p in existing.get("Ports") or []:
            if p.get("PrivatePort") == SERVER_PORT and p.get("PublicPort"):
                cur_port = int(p["PublicPort"])
                break
        same = (cur_port == int(port)) and (existing.get("Image") == image)
        if same:
            return status()
        if progress_cb:
            progress_cb({"line": f"Recreating: it is running with different settings "
                                 f"(port {cur_port}, image {existing.get('Image')}).",
                         "pct": 0.02, "phase": "Starting"})

    if existing:
        # A stale container may carry an old spec (different port, no GPU, an older
        # image). Recreate rather than start it and hope.
        _request("DELETE", f"/containers/{existing['Id']}?force=1&v=0", timeout=60.0)

    # Pull only when we have to, and never let a pull failure kill a usable local image.
    #
    # Three cases this gets right that an unconditional pull gets wrong:
    #   * the image is a LOCAL tag (built by hand, or loaded from a tar) — there is no
    #     registry to pull it from, and pulling 404s;
    #   * the host is air-gapped, or the registry is down / rate-limited — but the image
    #     is already cached, so there is nothing actually wrong;
    #   * the image is present and current — re-pulling 4.8 GB to discover that is rude.
    have = image_exists(image)
    if not have:
        if progress_cb:
            progress_cb({"line": f"Pulling {image} (~4.8 GB)…", "pct": 0.02,
                         "phase": "Pulling image"})
        pull_image(image, progress_cb)          # no local copy: a failure here IS fatal
    else:
        try:
            if progress_cb:
                progress_cb({"line": f"Checking {image} for updates…", "pct": 0.02,
                             "phase": "Pulling image"})
            pull_image(image, progress_cb)
        except Exception as e:
            # We already have it. Refusing to start because the registry is unreachable
            # would be a self-inflicted outage.
            log.warning("stem_splitter: could not refresh %s (%s) - using the local copy",
                        image, e)
            if progress_cb:
                progress_cb({"line": f"Could not reach the registry ({e}) — using the "
                                     f"local copy of {image}.",
                             "pct": 0.9, "phase": "Starting"})

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
        msg = str(body)
        if "port is already allocated" in msg or "address already in use" in msg:
            raise RuntimeError(
                f"host port {port} is already in use, so the container has nowhere to "
                f"publish. Something else is on it — often the plugin's own managed local "
                f"server (it defaults to {demucs_server_default_port()}). Stop that, or "
                f"pick a different port."
            )
        raise RuntimeError(f"could not start the container: {body}")

    # Windows will NOT fail the bind above. Docker publishes on 0.0.0.0 while a local
    # server listens on 127.0.0.1, both succeed, and loopback traffic then goes to the
    # MORE SPECIFIC bind — i.e. to the other server. The container looks healthy and every
    # request silently goes somewhere else. So don't trust the bind: ask the thing on the
    # published port to identify itself, and refuse to hand back a URL that isn't ours.
    _assert_port_is_ours(port, progress_cb)

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


def url_for(port: int, networks_shared: bool) -> str | None:
    """How WE reach the sidecar, or None if we can't.

    From inside a container that shares a user-defined network with it: by NAME (Docker's
    embedded DNS). Otherwise via the published host port on loopback — which is only *ours*
    when we're on the host, or sharing the host's network stack.

    Returns None rather than a plausible-but-wrong URL. Handing back
    `http://127.0.0.1:7866` from inside a bridge-only container points at the app container
    itself; autodetection and the health check would both fail with a confusing timeout
    instead of an honest "this can't be reached from here".
    """
    if networks_shared:
        # By container NAME we reach the port inside the container, not the published one.
        return f"http://{CONTAINER_NAME}:{SERVER_PORT}"

    try:
        from demucs_server import in_container
        containerized = in_container()
    except Exception:
        containerized = False

    if containerized and not _self_uses_host_networking():
        return None                 # loopback here is US, not the host. Don't pretend.
    return f"http://127.0.0.1:{port}"


def container_logs(cid: str, tail: int = 40) -> str:
    """Recent container output, de-multiplexed.

    Docker frames stdout/stderr with an 8-byte header per chunk, so the raw body is not
    text. Without this the UI can only ever say "not running", which is the least useful
    thing it could say to someone whose container is crash-looping.
    """
    try:
        conn = _connect(15.0)
        try:
            conn.request("GET", f"/{_API}/containers/{cid}/logs"
                                f"?stdout=1&stderr=1&tail={int(tail)}")
            resp = conn.getresponse()
            if resp.status >= 400:
                return ""
            raw = resp.read()
        finally:
            conn.close()
    except Exception:
        return ""

    out, i = [], 0
    while i + 8 <= len(raw):
        n = int.from_bytes(raw[i + 4:i + 8], "big")
        out.append(raw[i + 8:i + 8 + n].decode("utf-8", "replace"))
        i += 8 + n
    return "".join(out) if out else raw.decode("utf-8", "replace")


def _diagnose(cid: str, state: str | None) -> str:
    """Turn a dead/looping container into a sentence the user can act on."""
    if state == "running":
        return ""
    logs = container_logs(cid, tail=60)
    if "Permission denied" in logs and "/app/cache" in logs:
        # The one we actually hit: a fresh named volume is created root-owned, but the
        # image runs as uid 10001. Fixed in the image — but an ALREADY-created volume
        # keeps its root ownership forever, so telling the user to pull a new image
        # without also telling them to drop the volume would be useless advice.
        return ("The container can't write to its model cache: the Docker volume "
                f"'{CACHE_VOLUME}' is owned by root, but the server runs as an "
                "unprivileged user. This happens with a volume created by an older, "
                "broken image — Docker only sets a volume's ownership when it first "
                f"creates it. Remove it and start again: `docker volume rm {CACHE_VOLUME}`.")
    if state == "restarting":
        tail = " ".join(logs.strip().splitlines()[-3:])[:300]
        return f"The container keeps restarting. Last output: {tail}"
    if logs.strip():
        return " ".join(logs.strip().splitlines()[-3:])[:300]
    return ""


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
        if p.get("PrivatePort") == SERVER_PORT and p.get("PublicPort"):
            port = int(p["PublicPort"])
            break
    shared = bool(set(_self_networks()) &
                  set((c.get("NetworkSettings") or {}).get("Networks") or {}))
    url = url_for(port, shared) if running else None
    problem = _diagnose(c.get("Id", ""), c.get("State"))
    if running and not url:
        # It IS running — we simply cannot reach it from in here. Say so, rather than
        # advertising a loopback URL that points at this container.
        problem = problem or (
            f"The container is running, but feedBack can't reach it from inside its own "
            f"container: it isn't on a shared Docker network with us, and the published "
            f"port {port} is on the HOST's loopback, not ours. Put both on the same network "
            f"(the compose service does this), or use host networking."
        )
    out.update({
        "container": {"id": c.get("Id", "")[:12], "state": c.get("State"),
                      "status": c.get("Status"), "image": c.get("Image")},
        "running": running,
        "port": port,
        "url": url,
        # A container that exists but isn't running — or is running but unreachable from
        # here — has a REASON, and the user cannot run `docker logs` from inside the app.
        # Surfacing it is the difference between "not running" (useless) and "your volume is
        # root-owned, remove it" (actionable).
        "problem": problem,
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
        # 127.0.0.1: — loopback only. A bare "7866:7865" is 0.0.0.0 and would put an
        # unauthenticated inference server on the whole LAN. feedBack reaches this by
        # container name over the compose network, so it needs no host exposure at all;
        # the binding is here only so a human can curl it.
        f"      - \"127.0.0.1:{port}:{SERVER_PORT}\"\n"
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

# Changelog

## 0.3.0

### Run the demucs server in Docker

- **Sibling-container support.** A containerized feedBack cannot install the server on its
  host — `Popen` forks into the *caller's* namespaces, and there is no syscall for "run
  this over there". The settings page now offers Docker users the two things that are
  actually possible: a **compose snippet** (always shown, recommended — it needs no daemon
  access and grants nobody root on the host), and, *only when the Docker socket is already
  mounted*, a **one-click** that pulls the image and starts the server as a sibling
  container. We never suggest mounting the socket; it is root-equivalent on the host.
- **GPU** is requested only when the daemon advertises the nvidia runtime; otherwise the
  option is hidden with an explanation rather than silently downgrading to CPU.
- **A crash-looping container explains itself** — the plugin reads its logs and recognizes
  the root-owned-volume failure, since you can't run `docker logs` from inside the app.

### Fixes

- **Never publish onto the managed local server's port.** Both defaulted to 7865. On Linux
  the collision fails loudly; on **Windows both bind** (Docker on `0.0.0.0`, the local
  server on `127.0.0.1`, and loopback goes to the more specific bind) — so the container
  looked healthy while **every request went to the other server**. The sidecar now publishes
  on 7866 and refuses to hand back a URL that doesn't identify as its own container.
- **A registry outage no longer blocks a cached image.** Starting the container always
  pulled, so an air-gapped host — or a rate-limited/unreachable registry — would refuse to
  start an image already sitting on the machine. It still checks for an update when the
  image is present (that's cheap: only changed layers transfer), but a failed *refresh* is
  now a warning, not an error. Only a genuinely absent image makes a pull failure fatal.

## 0.2.0

*Released without a version bump — recorded here retroactively.*

### Managed local demucs server

- **Install / start / stop / status for a real demucs server**, managed by the plugin: one
  click installs the dependencies and model weights and starts it. Nothing heavy is ever
  downloaded implicitly — startup never pulls weights, and a split that needs them asks
  first.
- **GPU/CUDA support** for the local server, with the torch build pinned into a single pip
  resolve.
- **Cross-platform install without a venv** (`pip --target`), because the packaged Windows
  app bundles the python.org *embeddable* distribution, which has no `venv`/`ensurepip` and
  ignores `PYTHONPATH`.
- **The install could never have worked on Linux or macOS** until late in review:
  `audio-separator` pulls `diffq`, whose newest wheels stop at cp310, so pip fell back to a
  source build and needed a compiler. Windows escaped it by accident (it resolves
  `diffq-fixed`, which does ship modern wheels). Now `--no-deps` + binary-only.

## 0.1.3

Fixes from code review:

- **Local audio-separator now matches the server (6-stem).** `bs_roformer_sw` mapped
  to the stock 2-stem `model_bs_roformer_ep_317_sdr_12.9755.ckpt`, so the local
  engine produced different results than the remote server, which loads the custom
  6-stem `BS-Roformer-SW.ckpt`. The local mapping now uses the same checkpoint.
- **Stem-id normalization.** Output stems are mapped to canonical feedpak ids by
  label (preferring audio-separator's `_(<Label>)_` token, matching the server's
  extraction) rather than by raw filename. Fixes outputs like
  `mix_(Guitar)_BS-Roformer-SW.flac` becoming unrecognized garbage ids — which also
  broke local lyrics transcription (couldn't find the `vocals` stem).
- **Model weights stay in the managed dir.** The demucs subprocess (`TORCH_HOME`)
  and local WhisperX (`HF_HOME`/`TORCH_HOME`) now cache weights under
  `{config_dir}/models` instead of `~/.cache`, so `engine_status` counts them and
  **Uninstall** actually reclaims them.
- **Real cancellation.** Deleting a running job now interrupts it: progress ticks
  and the remote poll loop are cancel checkpoints, and the demucs subprocess is
  terminated. Canceled jobs report `canceled`, not `failed`.
- **Clear error instead of a crash** when a transcribe needs to split first but no
  split engine is available (previously raised a confusing missing-`engine`
  `TypeError`).
- **Dedicated lyrics server.** `whisperx.server_url` is now honored for lyric
  transcription (was dead code); splitting still uses `demucs_server_url`.
- **Batch is no longer capped at 1000.** "Split/Transcribe all missing" and the
  missing-counts now paginate the whole library.
- **No more `.feedpak.bak` accumulation.** The per-repack backup is removed after
  the atomic replace succeeds (it only guards a crash mid-repack).
- Use `asyncio.get_running_loop()` for the WS push loop (no deprecation warning).
- Added a unit test for stem-id normalization; `split_stems`' pure helpers are now
  importable without the host.

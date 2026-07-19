# Changelog

## 0.4.2

### No more restart after installing an engine

- **Installing a local engine now works on the first try.** ([#23](https://github.com/got-feedBack/feedBack-plugin-stem-splitter/pull/23))
  The natural first-run flow — try to split, get told to install the engine, install it,
  try again — failed until the whole app was restarted: the pre-install attempt left a
  stale negative entry in Python's import cache for the not-yet-existing engine directory,
  and nothing refreshed it. The engine path helper now invalidates import caches, as
  required when packages are installed at runtime.

## 0.4.1

### Getting to the queue

- **The "queued" toast is now a door.** ([#18](https://github.com/got-feedBack/feedBack-plugin-stem-splitter/issues/18))
  Click it and you land in the queue. The moment you start a job is exactly the moment you want
  to watch it, and the toast was the one thing on screen that knew about it — and did nothing.

  Only the *queued* toasts navigate. The rest stay inert on purpose: a toast that jumps you
  somewhere when you only meant to dismiss it is worse than one that does nothing.

- **A queue button in Settings**, with a live count of running and queued jobs. The queue screen
  already had a Settings button; this is that door from the other side. A user who has just
  finished configuring a server wants to watch it work, and previously had to go hunting for the
  way back.

## 0.4.0

### Re-align lyrics to vocals

- **New song-card action: "Re-align lyrics to vocals".** ([#20](https://github.com/got-feedBack/feedBack-plugin-stem-splitter/issues/20))
  The words are already right; the timings aren't. That happens constantly — lyrics pasted from
  the web, imported from a Guitar Pro file, or transcribed against a different mix: correct text,
  drifting timestamps. Until now the only repair was **Transcribe lyrics**, which throws your
  words away and lets Whisper guess them again. You fixed the timing by corrupting the text, and
  a mis-heard line is a worse outcome than a late one.

  Re-align sends the lyrics you already have to the server's `/align` endpoint — *"here are the
  words, when is each one sung?"* — and rewrites **only** the timings. It never invents a word.

  Enabled only when the song has both lyrics to re-time and a vocal stem to time them against —
  greyed out otherwise, since there is nothing a click could do about a missing stem.

  A missing *server* is different, and follows what Split and Transcribe already do: the item
  stays clickable and tells you what's wrong ("re-aligning needs a demucs/WhisperX server"),
  because a greyed-out item with no explanation teaches you nothing about how to fix it. Re-align
  is server-only on purpose — the local engine can transcribe but has no alignment entry point,
  and silently falling back to transcription would replace your words with Whisper's guesses,
  which is exactly what you clicked re-align to avoid.

  The manifest is not touched. `lyrics_source` is a closed vocabulary in the feedpak spec
  (authored | transcribed | user), and re-aligning doesn't change where the words came from —
  a pak whose timings were repaired is not a pak whose provenance changed.

## 0.3.4

### Fixes

- **A failed job now says why it failed.** ([#16](https://github.com/got-feedBack/feedBack-plugin-stem-splitter/issues/16))
  The error line lived inside the job title's span, which is `nowrap` + `overflow:hidden` +
  `text-overflow:ellipsis` — so the message was clipped to one line and the part explaining
  *why* was exactly the part that got cut. The user could see that something had failed and
  nothing about what.

  A failed job now gets its own full-width block: wrapped, monospaced, scrollable if it's a
  traceback, selectable, and with a **Copy** button — because where these end up is a bug report.
  (The copy path falls back to `execCommand` on a plain-http origin, where `navigator.clipboard`
  doesn't exist — which is exactly the NAS install whose users most need to paste an error.)

- **Server error bodies are no longer cut off at 300 characters.** The bodies carrying the most
  diagnosis are the ones that don't fit: a multi-field validation error, a 500 with a traceback
  (where the *last* line is the answer), an HTML error page from a reverse proxy. They're kept
  whole up to 4000 chars, and when a body genuinely has to be cut, it now says so and says how
  much there was. A silently cut error is how "the error is truncated" becomes the bug report
  instead of the actual bug.

## 0.3.3

### Update server

- **New: "Check for update" in the managed-server card.** The server's `server.py` is
  downloaded at *install* time and never touched again — so a bug fixed upstream could not
  reach anyone who had already installed. The only route was to uninstall and re-download
  several GB of wheels for a one-line change, which nobody does, so in practice the fix
  simply never landed.

  It re-fetches the source only (a few hundred KB), re-applies the launcher and driver
  bootstrap, and restarts the server if it was running. Dependencies and the model cache are
  untouched, so nothing is re-downloaded.

  Checking hits GitHub, so it happens only on a click — the status poll stays offline.
  The update runs only if there is genuinely something newer, so the button never restarts a
  healthy server for nothing.

### Fixes

- **The status chips said "downloading" when nothing was being downloaded.** The server marks
  a model as `downloading` while it *warms up* — even when it is only loading a cached file
  from disk into VRAM. So a few-second RAM load was displayed as a download, and users
  reasonably concluded their weights had been thrown away and re-fetched. (Reported exactly
  that way, and the user was right to believe the UI.)

  `server_status()` now reports which weights are on disk, per model, and a warm-up of a file
  we can already see reads **loading** (blue) rather than **downloading** (amber). A genuine
  fetch still says downloading.

  Done plugin-side on purpose: a server-side fix would have to reach people through a source
  refresh, and the plugin already knows what is on disk.

## 0.3.2

### Fixes — no more model re-downloads at launch

Two of the three causes of the nightly ~1 GB re-download are fixed here. The third — the demucs
server's 24h cache sweeper deleting the roformer checkpoint — is fixed in the *server*, and
server.py is downloaded at install time, so that fix only reaches an existing install once it
refreshes its server source (see the "Check for update" button, 0.3.3).

- **The server no longer re-downloads model weights at launch.** `models_downloaded()` — the
  gate that decides warmup-vs-skip-warmup — checked only the *roformer* checkpoint, on the
  reasoning that `bs_roformer_sw` is the model we split with. But warmup doesn't warm only
  that model: it warms whisperx and its wav2vec2 aligner too. So an install with the
  checkpoint but no aligner reported "downloaded", started **with** warmup, and the server
  quietly fetched 361 MB while the user watched the app open — precisely the thing this
  plugin promises never to do. The gate is now all-or-nothing.

- **`TORCH_HOME` moved to `cache/torch`.** `torch.hub` writes to `$TORCH_HOME/hub`, so
  pointing it at the cache root put the aligner in `cache/hub/` — a name the demucs server's
  cache sweeper does not protect, so it deleted the 361 MB file every 24 hours. `torch/` is
  preserved by *every* server version, so this fixes existing installs too, which a
  server-side fix cannot reach (server.py is downloaded at install time). It also matches
  what the container's compose file already does.

  The existing file is **moved, not re-fetched** — a rename on the same volume, so no bytes
  cross the wire. If that move fails (a locked file, a permissions problem), the server runs on
  the *old* `TORCH_HOME` for that run rather than pointing torch somewhere the file isn't — a
  failed migration must not turn into the 361 MB download it exists to prevent. It retries on
  the next start.

- **The "needs setup" prompt now names what is missing, and sizes it** — "it still needs whisperx aligner (~360 MB)"
  rather than "its models haven't been downloaded", which reads as *nothing* is downloaded to
  someone who already paid for the 2 GB fetch once, and hides the common case where everything
  is present except the aligner the sweeper ate.

## 0.3.1

### Fixes

- **The managed demucs server no longer outlives the app.** ([#12](https://github.com/got-feedBack/feedBack-plugin-stem-splitter/issues/12))
  It was found still listening **36 hours** after the app had been closed, holding its port,
  ~1 GB of RAM once warm, and the GPU.

  The server is spawned *detached* on purpose — so a crash or a Ctrl-C in the app can't kill
  it mid-separation, and Stop can kill its process tree without signalling the app itself.
  The cost of that was never paid for: nothing stopped it when the app exited normally
  either. Electron kills the feedBack backend; the server is a *grandchild* in its own
  session, so it simply stayed.

  Now the server watches the app and exits when the app goes away — on a graceful quit, a
  crash, a `taskkill /F`, an OOM kill, anything. A handler *inside* the app can't survive a
  hard kill; a watcher on the other side can. It takes its worker grandchildren
  (`run_demucs.py` / `run_roformer.py`) with it, so a split in flight doesn't just relocate
  the leak. A shutdown hook handles the graceful path so a clean quit stops it at once.

  The Docker sidecar is deliberately unaffected: it's a container with its own lifecycle,
  and killing it on app exit would be wrong.

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

# feedBack — Stem Splitter

Splits `.feedpak` songs that do not yet have per-instrument stems, and transcribes
lyrics when they're absent. Single-song or batch, with automatic "what's missing"
detection driven by the library index.

## Engines

Three ways to run, picked in the plugin's **Settings**:

- **Managed local server (easiest)** — the plugin can install and run
  [`got-feedBack/feedBack-demucs-server`](https://github.com/got-feedBack/feedBack-demucs-server)
  for you. One click: **Install server + models (~5 GB)** downloads its dependencies
  and the model weights, then starts it. While it's running the plugin uses it
  automatically. See [Local demucs server](#local-demucs-server) below.
- **Remote** — posts audio to a demucs/whisperx server you already run
  (`demucs_server_url` in the app settings). No local dependencies. Split model
  defaults to **`bs_roformer_sw`**.
- **Local engine (in-process, opt-in)** — runs the models inside the app instead of
  a server. The heavy libraries (`torch`, `demucs` / `audio-separator`, `whisperx`)
  are **not** installed until you click **Download local engine + models**.

Precedence in "Auto": the managed local server if it's running, else a configured
remote server, else the in-process local engine if installed. If none is available,
the action tells you so rather than failing silently.

> **Nothing heavy is ever downloaded implicitly.** No dependency and no model weight
> is fetched on plugin install, on app launch, or as a side effect of anything else —
> only when you explicitly click one of the download buttons. If you ask for a split
> before the models exist, the plugin **asks first** instead of stalling on a hidden
> multi-GB fetch.

> **Engines all yield 6 stems.** `bs_roformer_sw` is a **6-stem** BS-Roformer-SW
> model (`vocals`/`drums`/`bass`/`guitar`/`piano`/`other`), used by both the remote
> server and the local `audio-separator` engine (which loads the same
> `BS-Roformer-SW.ckpt` as the server, for parity). The local **demucs** engine
> (`htdemucs_6s`) is an equivalent 6-stem alternative. Output labels are normalized
> to canonical stem ids regardless of engine.

Lyrics can use a dedicated WhisperX host: set `whisperx.server_url` in the app config
and lyric transcription posts there (falling back to `demucs_server_url`); splitting
always uses `demucs_server_url`.

## What it does

- **Split stems** — extracts the full mix, runs source separation, writes
  `stems/<id>.ogg` back into the pak, rewrites the manifest (`stems:` +
  `stem_separation:`), and reindexes the song. The full mix is **kept** as a
  `default: false` fallback (its original file — e.g. `full.wav` — is preserved
  verbatim), so there's always a guaranteed-playable baseline.
- **Transcribe lyrics** — isolates a vocal stem (splitting first if needed), runs
  WhisperX, writes `lyrics.json` + manifest `lyrics` / `lyrics_source`, reindexes.

## Local demucs server

Settings → **Local demucs server** manages a real
[feedBack-demucs-server](https://github.com/got-feedBack/feedBack-demucs-server) on this
machine, so you don't have to stand one up yourself.

| Control | What it does |
|---|---|
| **Install server + models (~5 GB)** | Downloads the server source, installs its Python dependencies, downloads the model weights, and starts it. The **only** thing that downloads anything. |
| **Start** / **Stop** | Runs it / kills it (and its worker processes). |
| **Test status** | Probes `/health` — device, GPU, per-model warmup state. |
| **Uninstall server** | Removes the source, its dependencies **and its downloaded weights**. |

Status is shown as colored chips (running, models downloaded, per-model warmup), which
update live while the weights are downloading.

**Start with the app** (on by default, no-op until the server is installed) starts it
in the background on launch. This never slows startup and never downloads:

- weights already on disk → start **with warmup** (a RAM load — the server comes up
  warm, so the first split is fast)
- weights absent → start with `--skip-warmup`, so launching can't trigger the ~5 GB fetch

**Use for the whole app** additionally writes the local URL into the app's
`demucs_server_url`, so other parts of the app use it too. Without it, only this plugin
does (and your own `demucs_server_url` is left untouched).

### GPU (CUDA)

**Plain `pip install torch` gives the CPU-only wheel** — so a naive install leaves an
NVIDIA card completely idle and every split runs at CPU speed (minutes instead of
seconds). The installer therefore:

- **detects an NVIDIA GPU** (via `nvidia-smi`) and ticks **Use GPU (CUDA)** by default
  when one is present;
- installs the **CUDA torch build** (`torch==2.8.0+cu128` from PyTorch's index) — pinned
  *inside the same single pip resolve* as everything else, so it can't reintroduce a
  conflicting dependency tree;
- **verifies after installing** that `torch.version.cuda` is actually set and a GPU is
  visible, rather than trusting the pin.

**No CUDA Toolkit is needed** — the wheels bundle the CUDA runtime; you just need a
recent NVIDIA driver. The GPU build is a bigger download (~5.5 GB vs ~3 GB).

If you already installed the CPU build on a GPU machine, the status shows
**"CPU-only build — GPU idle"** and the button offers **Reinstall with GPU**. Override the
CUDA build with `STEM_SPLITTER_CUDA_TAG` (e.g. `cu126`) if `cu128` doesn't suit your driver.

### Requirements

Needs `pip` and a writable config dir — no `venv` (the packaged Windows app bundles the
embeddable Python, which has none). If the server can't be managed on your setup, the
section disables itself and explains why. In Docker the plugin and server share the
container, so it works, but it's usually CPU-only — running the demucs-server container
separately and pointing `demucs_server_url` at it is often the better option.

## Surfaces

- A **Stem Splitter** nav screen: job queue/dashboard, batch actions, missing lists.
- Per-song **Split stems** / **Transcribe lyrics** actions on the v3 song cards
  (registered via `libraryCardActions`).
- A **Settings** panel: engine selection, the local engine installer, and the managed
  demucs server.

Target Host: feedBack desktop with the v3 UI (`window.feedBack.uiVersion === 'v3'`).

## License

**AGPL-3.0-only** — the same license as the feedBack app. See [LICENSE](LICENSE).

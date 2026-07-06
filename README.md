# feedBack — Stem Splitter

Splits `.feedpak` songs that do not yet have per-instrument stems, and transcribes
lyrics when they're absent. Single-song or batch, with automatic "what's missing"
detection driven by the library index.

## Engines

Two ways to run, picked in the plugin's **Settings**:

- **Remote (default, lightweight)** — posts audio to a got-feedBack demucs/whisperx
  server (`demucs_server_url` in the app settings). No local dependencies. Split model
  defaults to **`bs_roformer_sw`**.
- **Local (opt-in)** — runs the models on this machine. The heavy libraries
  (~2 GB+: `torch`, `demucs` / `audio-separator`, `whisperx`) are **not** installed
  until you click **Download local engine + models** in Settings. Nothing is
  downloaded on install or app launch — only on that explicit action.

Precedence in "Auto": use the remote server if one is configured, else the local
engine if installed, else the action is offered with a prompt to download the engine.

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
  `stem_separation:`), removes the combined `stems/full.ogg`, and reindexes the song.
- **Transcribe lyrics** — isolates a vocal stem (splitting first if needed), runs
  WhisperX, writes `lyrics.json` + manifest `lyrics` / `lyrics_source`, reindexes.

## Surfaces

- A **Stem Splitter** nav screen: job queue/dashboard, batch actions, missing lists,
  engine settings + installer.
- Per-song **Split stems** / **Transcribe lyrics** actions on the v3 song cards
  (registered via `libraryCardActions`).

Target Host: feedBack desktop with the v3 UI (`window.feedBack.uiVersion === 'v3'`).

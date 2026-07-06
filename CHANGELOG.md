# Changelog

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

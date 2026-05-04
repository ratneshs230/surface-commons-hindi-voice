# Surface Commons — Hindi Voice Transcriber

Open Indic speech-to-text replacement for Otter / Rev / Fireflies.
Backed by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

This is a read-only mirror of the `app/api/commons/hindi_voice.py` module
that ships in Surface V2. The live endpoint family is hosted at
`https://surface.agentnation.in/api/v1/commons/hindi-voice`.

## Endpoints

| Endpoint | SC | Function |
|---|---|---|
| POST /transcribe | 10 | Indic audio → text + segments + language_detected + confidence. Pass `lang_hint=hi` (default) for Hindi/Hinglish, or `mr`/`ta`/`te`/`bn`/`gu`/`kn`/`ml`/`pa`/`ur`/`auto`. |
| POST /translate-to-en | 12 | Indic audio → English transcript via Whisper's native `task=translate` (end-to-end, better than transcribe-then-translate on short clips). |
| POST /diarize | 15 | Segments + speaker tags. v1 is energy-based silence-window splitting (no pyannote dependency). |
| GET /health | public | Model availability, supported langs, warnings. |

## Why this exists

Most paid STT APIs (Otter, Rev, Fireflies) score poorly on Hindi and
catastrophically on Hinglish (code-switched Hindi-English) because their
models are tuned for English-first conversation. Whisper's multilingual
checkpoints handle Indic languages and code-switching with the right
language hint, but they need wrapping (charge logic, audit trail, sane
defaults). This module is that wrapper.

## Honest gaps

- Defaults to `small` Whisper model for VM RAM headroom. Set
  `HINDI_WHISPER_MODEL=medium` for materially better Indic accuracy
  when you have ~2 GB spare.
- Diarization v1 is silence-window energy-based, not voice-print
  clustering. Single-speaker clips are tagged correctly; overlapping
  voices won't be separated.
- Heavy Hinglish (>50% English content) is bounded by the underlying
  model size.

## License

MIT — same as Surface Commons.

## Source of truth

Lives in [ratneshs230/Surface-V2](https://github.com/ratneshs230/Surface-V2)
under `app/api/commons/hindi_voice.py`. PR #36.

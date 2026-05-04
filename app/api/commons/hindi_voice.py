"""Hindi Voice Transcriber — Commons guild's INVENT-tier tool.

India-first STT replacement for Otter/Rev/Fireflies whose Hindi accuracy
is poor. Backed by faster-whisper (CTranslate2 backend) with multilingual
model + language-hint forcing for Hindi/Hinglish/regional accents.

Endpoints:

  POST /commons/hindi-voice/transcribe        — audio -> text + segments
  POST /commons/hindi-voice/translate-to-en   — Hindi/Indic audio -> English
  POST /commons/hindi-voice/diarize           — segments + speaker tags
                                                (energy-based v1, no
                                                pyannote heavy install)
  GET  /commons/hindi-voice/health             — public probe

Design notes
------------
* faster-whisper is a CPU-friendly Whisper port. We default to ``small``
  (~466 MB) — Indic accents need more than ``base`` but ``medium`` (1.5
  GB) does not fit comfortably alongside the rest of surface-api on a
  16 GB VM. Operators can override via env (``HINDI_WHISPER_MODEL``).
* ``language`` arg of WhisperModel.transcribe locks the decoder to the
  hinted language token. Forcing ``hi`` (or ``mr``/``ta``) gives
  significantly better accuracy than auto-detect on short clips of
  Indic speech where the language ID head can latch onto English.
* For Hinglish (code-switched), the recommended posture is
  ``lang_hint="hi"`` — Whisper handles intra-utterance switches but
  benefits from knowing the dominant locale.
* Diarization v1 is purely energy-based (silence-window split + alternating
  speaker tag). It is intentionally crude — pyannote.audio adds ~1 GB of
  PyTorch deps and a HuggingFace auth wall. v1 ships honest crude;
  heavier diarization is a future capability.

Auth: service-key + ``X-Agent-Id`` (same pattern as the rest of
``/api/v1/commons/...``). SC charged before processing; refunded on
failure.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.service_auth import require_service_key_and_agent
from app.models.agent import Agent
from app.services.ledger import (
    InsufficientFundsError,
    record_transaction,
)


_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/commons/hindi-voice", tags=["commons-hindi-voice"])


# ── SC pricing (per call, charge-then-process, refund-on-failure) ─────────
SC_COST = {
    "transcribe":      float(os.environ.get("HINDI_VOICE_SC_TRANSCRIBE", "10")),
    "translate_to_en": float(os.environ.get("HINDI_VOICE_SC_TRANSLATE",  "12")),
    "diarize":         float(os.environ.get("HINDI_VOICE_SC_DIARIZE",    "15")),
}

MAX_UPLOAD_BYTES = int(os.environ.get("HINDI_VOICE_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
WHISPER_MODEL = os.environ.get("HINDI_WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("HINDI_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("HINDI_WHISPER_COMPUTE", "int8")
DEFAULT_BEAM = int(os.environ.get("HINDI_WHISPER_BEAM", "1"))

ALLOWED_AUDIO_SUFFIX = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".webm", ".aac", ".opus"}

# Indic language codes Whisper knows about. ``auto`` = let Whisper LID decide.
LangHint = Literal["hi", "hi-IN", "mr", "ta", "te", "bn", "gu", "kn", "ml", "pa", "ur", "auto"]
INDIC_LANGS = {"hi", "mr", "ta", "te", "bn", "gu", "kn", "ml", "pa", "ur"}


_WHISPER_MODEL_CACHE: Any = None


def _whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _normalize_lang_hint(hint: str | None) -> str | None:
    """Map UI hints to Whisper-recognised ISO codes. ``None`` = auto-detect."""
    if hint is None:
        return None
    h = hint.strip().lower()
    if not h or h == "auto":
        return None
    if h in {"hi-in", "hi_in"}:
        return "hi"
    # Whisper codes are 2-letter ISO; pass through if recognised.
    if h in INDIC_LANGS or len(h) == 2:
        return h
    return None  # unrecognised -> fall back to auto


def _load_whisper():
    """Lazy-load + cache the WhisperModel. First call may take 10–30 s."""
    global _WHISPER_MODEL_CACHE
    if _WHISPER_MODEL_CACHE is not None:
        return _WHISPER_MODEL_CACHE
    from faster_whisper import WhisperModel
    _WHISPER_MODEL_CACHE = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )
    return _WHISPER_MODEL_CACHE


# ── Audit + billing helpers (mirrors commons.legal / commons.pdf) ─────────

async def _read_upload(f: UploadFile, *, label: str) -> bytes:
    data = await f.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{label}_empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "upload_too_large",
                "limit_bytes": MAX_UPLOAD_BYTES,
                "got_bytes": len(data),
                "field": label,
            },
        )
    return data


def _json_dump(d: dict) -> str:
    return json.dumps(d, default=str)


async def _log_tool_call(
    db: AsyncSession,
    *,
    agent_id: str,
    tool_name: str,
    input_summary: dict,
    output_len: int,
    sc_cost: float,
    latency_ms: int,
    error: str | None = None,
) -> None:
    await db.execute(
        text(
            """
            INSERT INTO tool_calls
                (id, agent_id, tool_name, input_json, output_len,
                 sc_cost, latency_ms, error, created_at)
            VALUES
                (:id, :agent_id, :tool_name, CAST(:input_json AS JSONB),
                 :output_len, :sc_cost, :latency_ms, :error, NOW())
            """
        ),
        {
            "id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "tool_name": tool_name,
            "input_json": _json_dump(input_summary),
            "output_len": output_len,
            "sc_cost": sc_cost,
            "latency_ms": latency_ms,
            "error": error,
        },
    )


async def _charge(
    db: AsyncSession,
    agent: Agent,
    *,
    tool: str,
    ref: str,
) -> float:
    cost = SC_COST[tool]
    if cost <= 0:
        return 0.0
    try:
        await record_transaction(
            db,
            agent_id=agent.id,
            delta=-cost,
            reason=f"commons.hindi_voice.{tool}",
            ref_id=ref,
            note=f"Hindi Voice: {tool}",
        )
    except InsufficientFundsError as e:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "insufficient_sc",
                "balance": e.balance,
                "needed": cost,
                "tool": f"commons.hindi_voice.{tool}",
            },
        )
    return cost


async def _refund(
    db: AsyncSession,
    agent: Agent,
    *,
    tool: str,
    ref: str,
    amount: float,
    reason: str,
) -> None:
    """Best-effort refund when the work failed after charging.

    The original ``_charge`` records a negative delta. We post a positive
    delta with a paired ref so the ledger nets to zero. Failures here are
    swallowed (logged) so the user always gets a clean error response.
    """
    if amount <= 0:
        return
    try:
        await record_transaction(
            db,
            agent_id=agent.id,
            delta=+amount,
            reason=f"commons.hindi_voice.{tool}.refund",
            ref_id=ref,
            note=f"Refund ({reason})",
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("hindi_voice refund failed agent=%s tool=%s ref=%s err=%s",
                     agent.id, tool, ref, exc)


# ── /health (public) ──────────────────────────────────────────────────────

@router.get("/health")
async def health() -> dict:
    available = _whisper_available()
    warnings: list[str] = []
    if not available:
        warnings.append("faster-whisper not installed — transcribe/translate/diarize will 501")
    if WHISPER_MODEL not in {"medium", "large-v2", "large-v3"}:
        warnings.append(
            f"Running on '{WHISPER_MODEL}' for VM RAM headroom. Indic accuracy "
            "improves materially on 'medium'/'large' — set HINDI_WHISPER_MODEL "
            "if you have spare RAM."
        )
    return {
        "service": "commons.hindi_voice",
        "version": "1.0.0",
        "available": {
            "transcribe": available,
            "translate_to_en": available,
            "diarize": available,
        },
        "whisper": {
            "model": WHISPER_MODEL,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE,
            "supported_indic_langs": sorted(INDIC_LANGS),
        },
        "sc_cost": SC_COST,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "allowed_audio_suffix": sorted(ALLOWED_AUDIO_SUFFIX),
        "warnings": warnings,
    }


# ── Shared upload-prep helper ─────────────────────────────────────────────

def _check_suffix(filename: str | None) -> str:
    name = (filename or "").lower()
    suffix = Path(name).suffix or ".bin"
    if suffix not in ALLOWED_AUDIO_SUFFIX:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_audio_format",
                "suffix": suffix,
                "allowed": sorted(ALLOWED_AUDIO_SUFFIX),
            },
        )
    return suffix


def _whisper_required_or_501() -> None:
    if not _whisper_available():
        raise HTTPException(
            status_code=501,
            detail={
                "error": "whisper_unavailable",
                "reason": "faster-whisper not installed on this VM",
                "remediation": (
                    "pip install faster-whisper soundfile "
                    "&& apt install ffmpeg"
                ),
            },
        )


# ── /transcribe ───────────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe(
    audio_file: Annotated[UploadFile, File(description="Audio (mp3/wav/m4a/ogg/flac/webm/opus) <= 25MB")],
    lang_hint: Annotated[str, Form(description="Language hint: 'hi','hi-IN','mr','ta','te','bn','gu','kn','ml','pa','ur','auto'")] = "hi",
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> dict:
    """Transcribe Hindi/Indic audio. Forces decoder language by default
    to ``hi`` because auto-detect frequently mis-IDs short Indic clips
    as English.
    """
    _whisper_required_or_501()
    suffix = _check_suffix(audio_file.filename)

    t0 = time.monotonic()
    ref = str(uuid.uuid4())
    charged = await _charge(db, agent, tool="transcribe", ref=ref)
    await db.commit()  # release the agent row before the long whisper call

    data = await _read_upload(audio_file, label="audio_file")
    forced_lang = _normalize_lang_hint(lang_hint)

    tmpdir = Path(tempfile.mkdtemp(prefix=f"hindi-voice-{ref}-"))
    in_path = tmpdir / f"in{suffix}"
    try:
        in_path.write_bytes(data)
        try:
            model = _load_whisper()
            segments_iter, info = model.transcribe(
                str(in_path),
                language=forced_lang,
                vad_filter=True,
                beam_size=DEFAULT_BEAM,
                task="transcribe",
            )
            segments: list[dict] = []
            full_text_parts: list[str] = []
            for s in segments_iter:
                seg_text = (s.text or "").strip()
                segments.append({
                    "start": round(float(s.start), 2),
                    "end":   round(float(s.end), 2),
                    "text":  seg_text,
                    "lang":  info.language,
                })
                if seg_text:
                    full_text_parts.append(seg_text)
        except Exception as e:  # noqa: BLE001
            await _refund(db, agent, tool="transcribe", ref=ref,
                          amount=charged, reason="transcribe_failed")
            await db.commit()
            raise HTTPException(
                status_code=500,
                detail={"error": "transcribe_failed", "msg": str(e)[:300]},
            )

        full_text = " ".join(full_text_parts)
        out = {
            "text": full_text,
            "segments": segments,
            "language_detected": info.language,
            "confidence": round(float(info.language_probability), 3),
            "duration_s": round(float(info.duration), 2),
            "lang_hint_used": forced_lang or "auto",
            "model": WHISPER_MODEL,
            "sc_charged": charged,
            "ref": ref,
        }

        await _log_tool_call(
            db, agent_id=agent.id, tool_name="commons.hindi_voice.transcribe",
            input_summary={
                "in_bytes": len(data),
                "filename": (audio_file.filename or "")[:128],
                "lang_hint": forced_lang or "auto",
                "duration_s": out["duration_s"],
                "language_detected": info.language,
                "segment_count": len(segments),
            },
            output_len=len(full_text),
            sc_cost=charged,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        await db.commit()
        return out
    except HTTPException:
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── /translate-to-en ──────────────────────────────────────────────────────

@router.post("/translate-to-en")
async def translate_to_en(
    audio_file: Annotated[UploadFile, File(description="Audio in Indic language <= 25MB")],
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> dict:
    """Whisper's native ``task='translate'`` returns English directly.
    Better than transcribe-then-translate for short clips because the
    decoder is conditioned end-to-end.
    """
    _whisper_required_or_501()
    suffix = _check_suffix(audio_file.filename)

    t0 = time.monotonic()
    ref = str(uuid.uuid4())
    charged = await _charge(db, agent, tool="translate_to_en", ref=ref)
    await db.commit()

    data = await _read_upload(audio_file, label="audio_file")

    tmpdir = Path(tempfile.mkdtemp(prefix=f"hindi-voice-tr-{ref}-"))
    in_path = tmpdir / f"in{suffix}"
    try:
        in_path.write_bytes(data)
        try:
            model = _load_whisper()
            segments_iter, info = model.transcribe(
                str(in_path),
                language=None,           # let Whisper detect source language
                vad_filter=True,
                beam_size=DEFAULT_BEAM,
                task="translate",        # outputs English regardless of source
            )
            segments: list[dict] = []
            parts: list[str] = []
            for s in segments_iter:
                seg_text = (s.text or "").strip()
                segments.append({
                    "start": round(float(s.start), 2),
                    "end":   round(float(s.end), 2),
                    "text":  seg_text,
                    "lang":  "en",
                })
                if seg_text:
                    parts.append(seg_text)
        except Exception as e:  # noqa: BLE001
            await _refund(db, agent, tool="translate_to_en", ref=ref,
                          amount=charged, reason="translate_failed")
            await db.commit()
            raise HTTPException(
                status_code=500,
                detail={"error": "translate_failed", "msg": str(e)[:300]},
            )

        english = " ".join(parts)
        out = {
            "text": english,
            "segments": segments,
            "source_language_detected": info.language,
            "confidence": round(float(info.language_probability), 3),
            "duration_s": round(float(info.duration), 2),
            "model": WHISPER_MODEL,
            "sc_charged": charged,
            "ref": ref,
        }

        await _log_tool_call(
            db, agent_id=agent.id, tool_name="commons.hindi_voice.translate_to_en",
            input_summary={
                "in_bytes": len(data),
                "filename": (audio_file.filename or "")[:128],
                "duration_s": out["duration_s"],
                "source_language_detected": info.language,
                "segment_count": len(segments),
            },
            output_len=len(english),
            sc_cost=charged,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        await db.commit()
        return out
    except HTTPException:
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── /diarize — energy-based v1 ────────────────────────────────────────────

def _decode_pcm_mono16k(in_path: Path) -> tuple[list[int], int]:
    """Try to read 16-bit mono PCM samples + sample rate from a wav file.

    For non-wav inputs we attempt a best-effort conversion via faster-whisper's
    bundled ffmpeg-via-pyav decoder if available. Returns (samples, sr).
    """
    suffix = in_path.suffix.lower()
    if suffix == ".wav":
        with wave.open(str(in_path), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            chans = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(n)
            if sw == 2:
                fmt = f"<{n*chans}h"
                samples = list(struct.unpack(fmt, raw))
                if chans > 1:
                    # downmix
                    samples = [
                        sum(samples[i:i+chans]) // chans
                        for i in range(0, len(samples), chans)
                    ]
                return samples, sr
            # 8/24/32-bit fallback: scale to int16
            try:
                import audioop  # py3.13 deprecates, but still present in 3.12
                pcm16 = audioop.lin2lin(raw, sw, 2)
                if chans > 1:
                    pcm16 = audioop.tomono(pcm16, 2, 0.5, 0.5)
                samples = list(struct.unpack(f"<{len(pcm16)//2}h", pcm16))
                return samples, sr
            except Exception:
                raise HTTPException(
                    status_code=415,
                    detail={"error": "wav_format_unsupported",
                            "msg": f"sampwidth={sw} chans={chans}"},
                )
    # non-wav: rely on faster-whisper's decode_audio which uses pyav/ffmpeg
    try:
        from faster_whisper.audio import decode_audio
        audio = decode_audio(str(in_path), sampling_rate=16000)
        # decode_audio returns float32 in [-1, 1]. Convert to int16-ish range
        # for energy comparison only (we never write back).
        samples = [int(x * 32767) for x in audio.tolist()]
        return samples, 16000
    except Exception as e:
        raise HTTPException(
            status_code=415,
            detail={"error": "audio_decode_failed", "msg": str(e)[:200]},
        )


def _energy_diarize(samples: list[int], sr: int,
                    *, win_ms: int = 30, silence_db: float = -35.0,
                    min_speech_ms: int = 250) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) for non-silent windows.

    Pure-Python, no numpy required (numpy is available but kept optional
    here so the function is testable in isolation).
    """
    if not samples or sr <= 0:
        return []
    win = max(1, int(sr * win_ms / 1000))
    # silence_db relative to int16 peak (32767). 32767 * 10^(-35/20) ≈ 583.
    silence_amp = 32767.0 * (10.0 ** (silence_db / 20.0))
    speech_flags: list[bool] = []
    n = len(samples)
    for i in range(0, n, win):
        chunk = samples[i:i+win]
        # peak amplitude is good enough + cheap (no sum-of-squares)
        peak = 0
        for v in chunk:
            av = -v if v < 0 else v
            if av > peak:
                peak = av
        speech_flags.append(peak > silence_amp)
    # collapse runs
    runs: list[tuple[float, float]] = []
    i = 0
    win_s = win / sr
    while i < len(speech_flags):
        if speech_flags[i]:
            j = i
            while j < len(speech_flags) and speech_flags[j]:
                j += 1
            start_s = i * win_s
            end_s = j * win_s
            if (end_s - start_s) * 1000.0 >= min_speech_ms:
                runs.append((start_s, end_s))
            i = j
        else:
            i += 1
    return runs


def _assign_speakers(segments: list[dict],
                     speech_runs: list[tuple[float, float]]) -> list[dict]:
    """Assign a speaker tag (S1/S2) to each Whisper segment.

    Strategy: sort speech runs by start; alternate speaker IDs every time
    a silence gap exceeds a threshold. Whisper segments inherit the tag
    of the speech-run whose interval most overlaps them. Crude — but
    deterministic and zero-deps.
    """
    if not speech_runs:
        for s in segments:
            s["speaker"] = "S1"
        return segments

    GAP_FOR_TURN_S = 0.6
    run_speakers: list[str] = []
    cur = "S1"
    prev_end = -1.0
    for (st, en) in speech_runs:
        if prev_end >= 0 and (st - prev_end) > GAP_FOR_TURN_S:
            cur = "S2" if cur == "S1" else "S1"
        run_speakers.append(cur)
        prev_end = en

    for seg in segments:
        s, e = seg["start"], seg["end"]
        best_overlap = 0.0
        best_idx = 0
        for i, (rs, re) in enumerate(speech_runs):
            ov = max(0.0, min(e, re) - max(s, rs))
            if ov > best_overlap:
                best_overlap = ov
                best_idx = i
        seg["speaker"] = run_speakers[best_idx]
    return segments


@router.post("/diarize")
async def diarize(
    audio_file: Annotated[UploadFile, File(description="Audio file <= 25MB")],
    lang_hint: Annotated[str, Form(description="Language hint (default 'hi')")] = "hi",
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> dict:
    """Energy-based diarization v1: Whisper for content + silence-window
    speaker turn detection. Honest crude — for production diarization we
    will eventually swap in pyannote.audio (gated by available RAM /
    HuggingFace auth).
    """
    _whisper_required_or_501()
    suffix = _check_suffix(audio_file.filename)

    t0 = time.monotonic()
    ref = str(uuid.uuid4())
    charged = await _charge(db, agent, tool="diarize", ref=ref)
    await db.commit()

    data = await _read_upload(audio_file, label="audio_file")
    forced_lang = _normalize_lang_hint(lang_hint)

    tmpdir = Path(tempfile.mkdtemp(prefix=f"hindi-voice-di-{ref}-"))
    in_path = tmpdir / f"in{suffix}"
    try:
        in_path.write_bytes(data)
        try:
            model = _load_whisper()
            segments_iter, info = model.transcribe(
                str(in_path),
                language=forced_lang,
                vad_filter=True,
                beam_size=DEFAULT_BEAM,
                task="transcribe",
            )
            segments: list[dict] = []
            for s in segments_iter:
                segments.append({
                    "start": round(float(s.start), 2),
                    "end":   round(float(s.end), 2),
                    "text":  (s.text or "").strip(),
                })

            samples, sr = _decode_pcm_mono16k(in_path)
            runs = _energy_diarize(samples, sr)
            segments = _assign_speakers(segments, runs)
        except HTTPException:
            await _refund(db, agent, tool="diarize", ref=ref,
                          amount=charged, reason="diarize_decode_failed")
            await db.commit()
            raise
        except Exception as e:  # noqa: BLE001
            await _refund(db, agent, tool="diarize", ref=ref,
                          amount=charged, reason="diarize_failed")
            await db.commit()
            raise HTTPException(
                status_code=500,
                detail={"error": "diarize_failed", "msg": str(e)[:300]},
            )

        speakers = sorted({s.get("speaker", "S1") for s in segments})
        out = {
            "segments": segments,
            "speakers": speakers,
            "speaker_count": len(speakers),
            "language_detected": info.language,
            "confidence": round(float(info.language_probability), 3),
            "duration_s": round(float(info.duration), 2),
            "method": "energy-window v1 (no pyannote)",
            "model": WHISPER_MODEL,
            "sc_charged": charged,
            "ref": ref,
        }

        await _log_tool_call(
            db, agent_id=agent.id, tool_name="commons.hindi_voice.diarize",
            input_summary={
                "in_bytes": len(data),
                "filename": (audio_file.filename or "")[:128],
                "duration_s": out["duration_s"],
                "language_detected": info.language,
                "segment_count": len(segments),
                "speaker_count": len(speakers),
            },
            output_len=sum(len(s["text"]) for s in segments),
            sc_cost=charged,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        await db.commit()
        return out
    except HTTPException:
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

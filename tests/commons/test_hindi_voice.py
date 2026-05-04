"""Tests for /commons/hindi-voice — Hindi Voice Transcriber.

Two layers, mirroring the legal_cite pattern:

1. Pure-Python tests — run anywhere. They cover language-hint normalization,
   energy-based diarization, and speaker assignment. No DB, no Whisper, no
   network.

2. HTTP integration tests against running surface-api at 127.0.0.1:3000.
   Skipped unless ``SURFACE_HINDI_VOICE_NETWORK=1`` (Whisper transcription
   is heavy and we don't want CI flake on cold-load timeouts). Uses a
   tiny synthesised wav fixture if no real Hindi sample is provided via
   ``HINDI_VOICE_TEST_FIXTURE``.
"""
from __future__ import annotations

import math
import os
import struct
import wave
from pathlib import Path

import httpx
import pytest

from app.api.commons.hindi_voice import (
    SC_COST,
    _assign_speakers,
    _energy_diarize,
    _normalize_lang_hint,
)


BASE = os.environ.get("SURFACE_BASE_URL", "http://127.0.0.1:3000")
SVC_KEY = os.environ.get("SURFACE_SERVICE_KEY") or os.environ.get("SERVICE_API_KEY")
AGENT_ID = os.environ.get("SURFACE_TEST_AGENT_ID")
NETWORK = os.environ.get("SURFACE_HINDI_VOICE_NETWORK") == "1"
FIXTURE_PATH = os.environ.get("HINDI_VOICE_TEST_FIXTURE")


def _client() -> httpx.Client:
    if not SVC_KEY or not AGENT_ID:
        pytest.skip("SURFACE_SERVICE_KEY + SURFACE_TEST_AGENT_ID required")
    return httpx.Client(
        base_url=BASE,
        headers={"X-Service-Key": SVC_KEY, "X-Agent-Id": AGENT_ID},
        timeout=180.0,
    )


# ── Pure: lang-hint normalization ─────────────────────────────────────────

def test_normalize_lang_hint_auto_returns_none():
    assert _normalize_lang_hint(None) is None
    assert _normalize_lang_hint("auto") is None
    assert _normalize_lang_hint("AUTO") is None
    assert _normalize_lang_hint("") is None


def test_normalize_lang_hint_hi_in_to_hi():
    assert _normalize_lang_hint("hi-IN") == "hi"
    assert _normalize_lang_hint("hi_IN") == "hi"
    assert _normalize_lang_hint("HI-IN") == "hi"


def test_normalize_lang_hint_passes_indic():
    for code in ("hi", "mr", "ta", "te", "bn", "gu", "kn", "ml", "pa", "ur"):
        assert _normalize_lang_hint(code) == code


def test_normalize_lang_hint_unknown_falls_back_to_auto():
    # Not in Indic set, not 2-char ISO -> auto.
    assert _normalize_lang_hint("klingon") is None


# ── Pure: energy-based diarizer ───────────────────────────────────────────

def _synth_samples(sr: int = 16000, *,
                   silence_s: float = 0.5,
                   speech_s: float = 1.0,
                   freq: int = 400,
                   amp: int = 12000) -> list[int]:
    """Build a deterministic [silence|speech|silence|speech] signal."""
    n_silence = int(silence_s * sr)
    n_speech = int(speech_s * sr)
    out: list[int] = []
    out.extend([0] * n_silence)
    for i in range(n_speech):
        out.append(int(amp * math.sin(2 * math.pi * freq * i / sr)))
    out.extend([0] * n_silence)
    for i in range(n_speech):
        out.append(int(amp * math.sin(2 * math.pi * freq * i / sr)))
    return out


def test_energy_diarize_finds_two_runs():
    samples = _synth_samples()
    runs = _energy_diarize(samples, 16000)
    assert len(runs) == 2
    # First run starts after ~0.5s silence; second run after gap.
    assert 0.4 <= runs[0][0] <= 0.6
    assert runs[0][1] > runs[0][0]
    assert runs[1][0] > runs[0][1]


def test_energy_diarize_empty():
    assert _energy_diarize([], 16000) == []
    assert _energy_diarize([0] * 1000, 16000) == []


def test_energy_diarize_min_speech_filter():
    # 50ms blip should be filtered out by default min_speech_ms=250.
    sr = 16000
    blip = [12000] * (sr * 50 // 1000)
    samples = [0] * sr + blip + [0] * sr
    runs = _energy_diarize(samples, sr)
    assert runs == []


def test_assign_speakers_alternates_on_long_gap():
    # Two runs separated by >0.6s should get different speakers.
    runs = [(0.0, 1.0), (2.0, 3.0)]
    segs = [
        {"start": 0.1, "end": 0.9, "text": "one"},
        {"start": 2.1, "end": 2.9, "text": "two"},
    ]
    out = _assign_speakers(segs, runs)
    assert out[0]["speaker"] != out[1]["speaker"]


def test_assign_speakers_same_speaker_on_short_gap():
    # Runs separated by <0.6s should share a speaker.
    runs = [(0.0, 1.0), (1.2, 2.0)]
    segs = [
        {"start": 0.1, "end": 0.9, "text": "a"},
        {"start": 1.3, "end": 1.9, "text": "b"},
    ]
    out = _assign_speakers(segs, runs)
    assert out[0]["speaker"] == out[1]["speaker"]


def test_assign_speakers_with_no_runs_falls_back():
    segs = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    out = _assign_speakers(segs, [])
    assert out[0]["speaker"] == "S1"


# ── HTTP: /health is public ───────────────────────────────────────────────

def test_health_public():
    with httpx.Client(base_url=BASE, timeout=10.0) as c:
        r = c.get("/api/v1/commons/hindi-voice/health")
        # service may be up or transcribe may be 501; either is fine here.
        if r.status_code != 200:
            pytest.skip(f"hindi-voice /health unreachable: {r.status_code}")
        d = r.json()
        assert d["service"] == "commons.hindi_voice"
        assert "sc_cost" in d
        assert d["sc_cost"]["transcribe"] == SC_COST["transcribe"]
        assert d["sc_cost"]["translate_to_en"] == SC_COST["translate_to_en"]
        assert d["sc_cost"]["diarize"] == SC_COST["diarize"]
        assert "hi" in d["whisper"]["supported_indic_langs"]


# ── HTTP: /transcribe round-trip with synthesised fixture ─────────────────

def _write_test_wav(path: Path, *, dur_s: float = 1.5,
                    sr: int = 16000, freq: int = 440, amp: int = 8000) -> None:
    """Write a tiny mono 16-bit PCM wav with a sine tone."""
    n = int(dur_s * sr)
    samples = [int(amp * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{n}h", *samples))


@pytest.fixture(scope="module")
def audio_fixture(tmp_path_factory) -> Path:
    if FIXTURE_PATH:
        p = Path(FIXTURE_PATH)
        if not p.exists():
            pytest.skip(f"HINDI_VOICE_TEST_FIXTURE points to missing file: {p}")
        return p
    p = tmp_path_factory.mktemp("hindi-voice") / "tone.wav"
    _write_test_wav(p)
    return p


@pytest.mark.skipif(not NETWORK, reason="set SURFACE_HINDI_VOICE_NETWORK=1 to run live tests")
def test_transcribe_endpoint(audio_fixture: Path):
    with _client() as c:
        with audio_fixture.open("rb") as f:
            r = c.post(
                "/api/v1/commons/hindi-voice/transcribe",
                files={"audio_file": (audio_fixture.name, f, "audio/wav")},
                data={"lang_hint": "hi"},
            )
        if r.status_code == 501:
            pytest.skip("faster-whisper not installed on server")
        assert r.status_code == 200, r.text
        d = r.json()
        assert "text" in d
        assert "segments" in d
        assert "language_detected" in d
        assert "confidence" in d
        assert "sc_charged" in d
        assert d["sc_charged"] == SC_COST["transcribe"]


@pytest.mark.skipif(not NETWORK, reason="set SURFACE_HINDI_VOICE_NETWORK=1 to run live tests")
def test_translate_to_en_endpoint(audio_fixture: Path):
    with _client() as c:
        with audio_fixture.open("rb") as f:
            r = c.post(
                "/api/v1/commons/hindi-voice/translate-to-en",
                files={"audio_file": (audio_fixture.name, f, "audio/wav")},
            )
        if r.status_code == 501:
            pytest.skip("faster-whisper not installed on server")
        assert r.status_code == 200, r.text
        d = r.json()
        assert "text" in d
        assert "source_language_detected" in d
        assert d["sc_charged"] == SC_COST["translate_to_en"]


@pytest.mark.skipif(not NETWORK, reason="set SURFACE_HINDI_VOICE_NETWORK=1 to run live tests")
def test_diarize_endpoint(audio_fixture: Path):
    with _client() as c:
        with audio_fixture.open("rb") as f:
            r = c.post(
                "/api/v1/commons/hindi-voice/diarize",
                files={"audio_file": (audio_fixture.name, f, "audio/wav")},
                data={"lang_hint": "hi"},
            )
        if r.status_code == 501:
            pytest.skip("faster-whisper not installed on server")
        assert r.status_code == 200, r.text
        d = r.json()
        assert "segments" in d
        assert "speakers" in d
        assert "speaker_count" in d
        assert d["sc_charged"] == SC_COST["diarize"]
        # every segment carries a speaker tag
        for seg in d["segments"]:
            assert "speaker" in seg


# ── Bad-input regression ──────────────────────────────────────────────────

@pytest.mark.skipif(not NETWORK, reason="set SURFACE_HINDI_VOICE_NETWORK=1 to run live tests")
def test_unsupported_audio_format_400(tmp_path):
    # Make a fake .xyz file
    bad = tmp_path / "bad.xyz"
    bad.write_bytes(b"not audio")
    with _client() as c:
        with bad.open("rb") as f:
            r = c.post(
                "/api/v1/commons/hindi-voice/transcribe",
                files={"audio_file": ("bad.xyz", f, "application/octet-stream")},
            )
        # 501 first if whisper missing; 400 otherwise.
        assert r.status_code in (400, 501)

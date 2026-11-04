"""Audio/video decoding and degradation via ffmpeg."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime

import numpy as np
from scipy.io import wavfile


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Decode any audio/video file to mono float32 at `sr`."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    pk = float(np.max(np.abs(audio))) or 1.0
    wavfile.write(path, sr, (np.clip(audio / max(pk, 1.0), -1, 1) * 32767).astype(np.int16))


def creation_time(path: str) -> float | None:
    """Best-effort recording start time (unix seconds) from container metadata.

    Prefers Apple's `com.apple.quicktime.creationdate` (local time with zone
    offset), then the generic `creation_time`. Note: some Android phones write
    the *end* of recording, so treat this as approximate (+/- clip length).
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", path],
            check=True, capture_output=True, text=True,
        ).stdout
    except Exception:
        return None
    info = json.loads(out or "{}")
    tags = {}
    for s in info.get("streams", []):
        tags.update({k.lower(): v for k, v in (s.get("tags") or {}).items()})
    tags.update({k.lower(): v for k, v in (info.get("format", {}).get("tags") or {}).items()})
    for key in ("com.apple.quicktime.creationdate", "creation_time", "date"):
        if key in tags:
            try:
                return datetime.fromisoformat(tags[key].replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
    return None


# name -> (pre-filter, encoder args, container ext)
DEGRADATIONS: dict[str, tuple[str | None, list[str], str] | None] = {
    "clean": None,
    "mp3_128k": (None, ["-c:a", "libmp3lame", "-b:a", "128k"], "mp3"),
    "mp3_64k": (None, ["-c:a", "libmp3lame", "-b:a", "64k"], "mp3"),
    "mp3_32k": (None, ["-c:a", "libmp3lame", "-b:a", "32k"], "mp3"),
    # phone-ish chain: mic high-pass, band limit, AGC, AAC
    "aac_phone": ("highpass=f=200,lowpass=f=7000,dynaudnorm=f=250", ["-c:a", "aac", "-b:a", "64k"], "m4a"),
    # spectral noise suppression (stationary noise reduction) then AAC
    "denoise": ("afftdn=nr=24:nf=-45:tn=1", ["-c:a", "aac", "-b:a", "96k"], "m4a"),
    # WhatsApp/Discord style low-bitrate VoIP Opus
    "opus_voip_16k": (None, ["-c:a", "libopus", "-b:a", "16k", "-application", "voip"], "ogg"),
}


def degrade(audio: np.ndarray, sr: int, name: str) -> np.ndarray:
    """Round-trip `audio` through an ffmpeg filter/codec chain."""
    spec = DEGRADATIONS[name]
    if spec is None:
        return audio.astype(np.float32)
    filt, enc, ext = spec
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.wav")
        mid = os.path.join(td, f"mid.{ext}")
        wavfile.write(src, sr, audio.astype(np.float32))
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", src]
        if filt:
            cmd += ["-af", filt]
        if ext == "ogg":
            cmd += ["-ar", "48000"]
        cmd += enc + [mid]
        subprocess.run(cmd, check=True, capture_output=True)
        out = load_audio(mid, sr)
    n = len(audio)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out[:n]

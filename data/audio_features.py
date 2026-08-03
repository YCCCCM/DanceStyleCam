"""AIST-style 35D music features extracted once per aligned source song."""

from __future__ import annotations

from functools import lru_cache
import io
import math
from pathlib import Path

import numpy as np


FPS = 30
HOP_LENGTH = 512
SAMPLE_RATE = FPS * HOP_LENGTH


def audio_frame_count(path: str | Path, fps: int = FPS) -> int:
    import soundfile as sf

    info = sf.info(str(path))
    return int(float(info.frames) / float(info.samplerate) * float(fps))


def extract_music35(path: str | Path, aligned_frame_limit: int, output_frames: int) -> np.ndarray:
    import librosa

    duration = float(aligned_frame_limit) / float(FPS)
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, duration=duration)
    onset = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_chroma=12,
    ).T

    onset_indices = librosa.onset.onset_detect(
        onset_envelope=onset,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    onset_one_hot = np.zeros_like(onset, dtype=np.float32)
    onset_one_hot[onset_indices] = 1.0

    tempo_function = getattr(librosa.feature, "tempo", None) or librosa.beat.tempo
    start_bpm = float(np.asarray(tempo_function(y=audio, sr=SAMPLE_RATE)).reshape(-1)[0])
    _, beat_indices = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_one_hot = np.zeros_like(onset, dtype=np.float32)
    beat_one_hot[np.asarray(beat_indices, dtype=np.int64)] = 1.0

    shared_frames = min(len(onset), len(mfcc), len(chroma))
    features = np.concatenate(
        (
            onset[:shared_frames, None],
            mfcc[:shared_frames],
            chroma[:shared_frames],
            onset_one_hot[:shared_frames, None],
            beat_one_hot[:shared_frames, None],
        ),
        axis=1,
    ).astype(np.float32)
    if len(features) < output_frames:
        features = np.pad(features, ((0, output_frames - len(features)), (0, 0)))
    return features[:output_frames]


@lru_cache(maxsize=2)
def _load_aligned_audio(path: str) -> tuple[np.ndarray, int]:
    import librosa

    audio, sample_rate = librosa.load(path, sr=None)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_pcm16_wav(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer


def _legacy_clip_wav(
    path: str | Path,
    start_frame: int,
    end_frame: int | None,
    aligned_frame_limit: int | None,
) -> io.BytesIO:
    """Reproduce the PCM16 clip written by the released DCM++ builder."""

    import librosa

    audio, sample_rate = _load_aligned_audio(str(Path(path).resolve()))
    if aligned_frame_limit is not None:
        aligned_end = int(float(aligned_frame_limit) / float(FPS) * float(sample_rate)) + 1
        aligned_wav = _write_pcm16_wav(audio[:aligned_end], sample_rate)
        audio, sample_rate = librosa.load(aligned_wav, sr=None)
    if end_frame is None:
        clip = audio
    else:
        start_sample = int(float(start_frame) / float(FPS) * float(sample_rate))
        end_sample = math.ceil(float(end_frame) / float(FPS) * float(sample_rate))
        clip = audio[start_sample:end_sample]
    return _write_pcm16_wav(clip, sample_rate)


@lru_cache(maxsize=512)
def _extract_music35_clip_cached(
    resolved_path: str,
    start_frame: int,
    end_frame: int | None,
    output_frames: int,
    aligned_frame_limit: int | None,
) -> np.ndarray:
    """Extract the exact split-local AIST feature used by legacy DCM++."""

    import librosa

    wav = _legacy_clip_wav(resolved_path, start_frame, end_frame, aligned_frame_limit)
    audio, _ = librosa.load(wav, sr=SAMPLE_RATE)
    onset = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_chroma=12,
    ).T
    onset_indices = librosa.onset.onset_detect(
        onset_envelope=onset.flatten(),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    onset_one_hot = np.zeros_like(onset, dtype=np.float32)
    onset_one_hot[onset_indices] = 1.0

    wav.seek(0)
    tempo_audio, _ = librosa.load(wav)
    tempo_function = getattr(librosa.feature, "tempo", None) or librosa.beat.tempo
    start_bpm = float(np.asarray(tempo_function(y=tempo_audio)).reshape(-1)[0])
    _, beat_indices = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_one_hot = np.zeros_like(onset, dtype=np.float32)
    beat_one_hot[np.asarray(beat_indices, dtype=np.int64)] = 1.0

    shared_frames = min(len(onset), len(mfcc), len(chroma))
    features = np.concatenate(
        (
            onset[:shared_frames, None],
            mfcc[:shared_frames],
            chroma[:shared_frames],
            onset_one_hot[:shared_frames, None],
            beat_one_hot[:shared_frames, None],
        ),
        axis=1,
    ).astype(np.float32)
    if len(features) < output_frames:
        features = np.pad(features, ((0, output_frames - len(features)), (0, 0)))
    return features[:output_frames]


def extract_music35_clip(
    path: str | Path,
    start_frame: int,
    end_frame: int | None,
    output_frames: int,
    aligned_frame_limit: int | None = None,
) -> np.ndarray:
    """Return a legacy-compatible feature for one virtual clip.

    ``end_frame`` is exclusive.  Pass ``None`` for an unsplit full sequence,
    matching the old DCM++ builder's copy-without-slicing branch.
    """

    return _extract_music35_clip_cached(
        str(Path(path).resolve()),
        int(start_frame),
        None if end_frame is None else int(end_frame),
        int(output_frames),
        None if aligned_frame_limit is None else int(aligned_frame_limit),
    )

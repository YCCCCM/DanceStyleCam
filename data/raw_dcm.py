"""Read-only description of the publicly downloadable DCM directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawSequenceFiles:
    sequence_id: str
    audio: Path
    camera: Path
    motion: Path
    camera_centric: Path | None = None
    aligned_motion: Path | None = None
    raw_audio: Path | None = None
    aligned_audio: Path | None = None


@dataclass(frozen=True)
class RawDCM:
    root: Path

    def sequence_files(self, sequence_id: str | int) -> RawSequenceFiles:
        item = str(sequence_id)
        raw_audio = self.root / "amc_raw_data" / f"amc{item}" / f"a{item}.wav"
        # DanceCamAnimator's alignment stage writes a mono/truncated audio
        # stream.  Prefer it when present so music35 matches the released
        # DCM++ features exactly; raw downloads without that derived folder
        # remain supported through the fallback.
        aligned_audio = self.root / "amc_aligned_data" / "Audio" / f"a{item}.wav"
        aligned_camera_centric = self.root / "amc_aligned_data" / "CameraCentric" / f"c{item}.json"
        aligned_motion = self.root / "amc_aligned_data" / "Simplified_MotionGlobalTransform" / f"m{item}_gt.json"
        return RawSequenceFiles(
            sequence_id=item,
            audio=aligned_audio if aligned_audio.is_file() else raw_audio,
            camera=self.root / "amc_camera_json" / f"c{item}.json",
            motion=self.root / "Simplified_MotionGlobalTransform" / f"m{item}_GlobalTransform.json",
            camera_centric=aligned_camera_centric if aligned_camera_centric.is_file() else None,
            aligned_motion=aligned_motion if aligned_motion.is_file() else None,
            raw_audio=raw_audio,
            aligned_audio=aligned_audio if aligned_audio.is_file() else None,
        )

    def missing_files(self, sequence_id: str | int) -> list[Path]:
        files = self.sequence_files(sequence_id)
        return [path for path in (files.audio, files.camera, files.motion) if not path.is_file()]

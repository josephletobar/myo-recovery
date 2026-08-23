"""Core data structures shared by the loader and aligner."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class Stream:
    """A modality's values paired with the time of every row."""

    timestamps: np.ndarray  # (N,), seconds since Unix epoch
    values: np.ndarray       # (N, ...)
    source: str = ""
    headings: tuple[str, ...] = ()

    @property
    def start_time(self) -> float:
        return float(self.timestamps[0])

    @property
    def end_time(self) -> float:
        return float(self.timestamps[-1])

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def effective_rate_hz(self) -> float:
        if len(self.timestamps) < 2 or self.duration <= 0:
            return float("nan")
        return (len(self.timestamps) - 1) / self.duration


@dataclass(slots=True)
class ActivityInterval:
    """One start/stop interval from the experiment activity table."""

    index: int
    activity: str
    start_time: float
    end_time: float
    valid: str
    notes: str

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(slots=True)
class SessionSequence:
    """One activity window, raw or aligned, with the same field structure."""

    subject: str
    session: str
    activity: str
    start_time: float
    end_time: float
    emg_left: Stream
    emg_right: Stream
    hand_pose: Stream
    tactile_left: Stream | None
    tactile_right: Stream | None
    rgb_timestamps: Stream | None
    rgb_frames: list[str] | None = None
    rgb_video_path: str | None = None
    rgb_frame_indices: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def align(self, target_hz: float = 30.0, *, emg_mode: str = "rms") -> "SessionSequence":
        """Return a new sequence whose streams share a regular target timeline."""
        from .aligner import align_sequence

        return align_sequence(self, target_hz=target_hz, emg_mode=emg_mode)

    def combined_emg(self) -> np.ndarray:
        """Return left/right EMG concatenated as (T, 16) when clocks match."""
        if self.emg_left.timestamps.shape != self.emg_right.timestamps.shape or not np.allclose(
            self.emg_left.timestamps, self.emg_right.timestamps
        ):
            raise ValueError("EMG streams have different timestamps; align the sequence first")
        return np.concatenate([self.emg_left.values, self.emg_right.values], axis=1)

    def combined_tactile(self) -> np.ndarray | None:
        """Return tactile data as (T, 2, 32, 32) when clocks match."""
        if self.tactile_left is None or self.tactile_right is None:
            return None
        if self.tactile_left.timestamps.shape != self.tactile_right.timestamps.shape or not np.allclose(
            self.tactile_left.timestamps, self.tactile_right.timestamps
        ):
            raise ValueError("tactile streams have different timestamps; align the sequence first")
        return np.stack([self.tactile_left.values, self.tactile_right.values], axis=1)

    def emg2pose_target(self, side: str):
        """Map this sequence's Xsens pose to EMG2Pose's 20-D target."""
        from .emg2pose import map_hand_pose_to_emg2pose

        return map_hand_pose_to_emg2pose(self.hand_pose, side=side)

    def to_emg2pose(self, side: str):
        """Return one side's EMG and mapped 20-angle target."""
        from .emg2pose import make_emg2pose_sequence

        return make_emg2pose_sequence(self, side=side)

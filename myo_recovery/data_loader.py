"""Windowed loading of the ActionSense/ActionNet HDF5 recording.

The loader keeps each sensor's native timestamp vector. This is intentional:
the recording contains video, pose, EMG, and tactile streams at different rates.
Use an explicit preprocessing step to resample them onto a common model clock.

When ``video_path`` is supplied, RGB is represented lazily as that MP4 path plus
the frame indices covering each activity; video pixels are not copied into RAM.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from .types import ActivityInterval, SessionSequence, Stream


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def _headings_for_data_path(file: h5py.File, data_path: str) -> tuple[str, ...]:
    group = file[data_path.rsplit("/", 1)[0]]
    raw = group.attrs.get("Data headings", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        parsed = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError):
        return ()
    return tuple(str(item) for item in parsed) if isinstance(parsed, list) else ()


class ActionSenseLoader:
    """Read selected activity windows and optional MP4 frame references."""

    ACTIVITY_DATA = "/experiment-activities/activities/data"
    ACTIVITY_TIME = "/experiment-activities/activities/time_s"
    # Unlike most ActionSense streams, this stream stores the absolute frame
    # timestamps directly in ``data``; there is no nested ``time_s`` dataset.
    WORLD_VIDEO_DATA = "/eye-tracking-video-world/frame_timestamp/data"
    MANUS_CALIBRATION_DATA = "/experiment-calibration/third_party/data"
    MANUS_CALIBRATION_TIME = "/experiment-calibration/third_party/time_s"

    def __init__(self, path: str | Path, *, video_path: str | Path | None = None):
        self.path = Path(path)
        self.video_path = Path(video_path) if video_path is not None else None
        self._file: h5py.File | None = None

        stem = self.path.stem
        if "_streamLog" in stem:
            self.session = stem.split("_streamLog", 1)[0]
            suffix = stem.rsplit("_", 1)[-1]
            self.subject = suffix if suffix.startswith("S") else "unknown"
        else:
            self.session = stem
            self.subject = "unknown"

    def __enter__(self) -> "ActionSenseLoader":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def open(self) -> None:
        if self._file is None:
            if self.video_path is not None and not self.video_path.is_file():
                raise FileNotFoundError(f"RGB video does not exist: {self.video_path}")
            self._file = h5py.File(self.path, "r")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("ActionSenseLoader must be used inside 'with' or after open()")
        return self._file

    def _read_timed(
        self,
        data_path: str,
        time_path: str,
        start_time: float,
        end_time: float,
    ) -> Stream:
        """Read only the rows whose timestamps overlap [start_time, end_time]."""
        timestamps = np.asarray(self.file[time_path][:]).reshape(-1)
        first = int(np.searchsorted(timestamps, start_time, side="left"))
        last = int(np.searchsorted(timestamps, end_time, side="right"))
        first = max(0, min(first, len(timestamps)))
        last = max(first, min(last, len(timestamps)))
        return Stream(
            timestamps=timestamps[first:last].astype(np.float64, copy=False),
            values=np.asarray(self.file[data_path][first:last]),
            source=data_path,
            headings=_headings_for_data_path(self.file, data_path),
        )

    def activities(self) -> list[ActivityInterval]:
        """Parse paired Start/Stop rows into activity intervals."""
        data = np.asarray(self.file[self.ACTIVITY_DATA][:])
        times = np.asarray(self.file[self.ACTIVITY_TIME][:]).reshape(-1)
        open_intervals: dict[str, tuple[int, float, str, str]] = {}
        intervals: list[ActivityInterval] = []

        for row_index, (row, timestamp) in enumerate(zip(data, times)):
            activity = _decode(row[0])
            state = _decode(row[1]).strip().lower()
            valid = _decode(row[2])
            notes = _decode(row[3])
            if state == "start":
                open_intervals[activity] = (row_index, float(timestamp), valid, notes)
            elif state == "stop" and activity in open_intervals:
                start_index, start, start_valid, start_notes = open_intervals.pop(activity)
                intervals.append(
                    ActivityInterval(
                        index=start_index,
                        activity=activity,
                        start_time=start,
                        end_time=float(timestamp),
                        valid=valid or start_valid,
                        notes=notes or start_notes,
                    )
                )
        return sorted(intervals, key=lambda item: item.start_time)

    def _select_activity(self, selector: int | str) -> ActivityInterval:
        intervals = self.activities()
        if isinstance(selector, int):
            try:
                return intervals[selector]
            except IndexError as exc:
                raise IndexError(f"activity index {selector} is out of range (0..{len(intervals)-1})") from exc
        matches = [item for item in intervals if item.activity == selector]
        if not matches:
            raise KeyError(f"activity not found: {selector!r}")
        return matches[0]

    def emg2pose_open_hand_neutral_degrees(self, side: str) -> np.ndarray:
        """Estimate a side-specific 20-D Xsens neutral from Manus calibration.

        The raw ActionSense file records ``Manus: Poses Left`` and
        ``Manus: Poses Right`` as timestamped third-party calibration intervals.
        We read only the requested interval and estimate its open-hand offset;
        the raw stream itself is never modified.
        """
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        data = np.asarray(self.file[self.MANUS_CALIBRATION_DATA][:])
        times = np.asarray(self.file[self.MANUS_CALIBRATION_TIME][:]).reshape(-1)
        label = f"Manus: Poses {side.capitalize()}"
        start_time: float | None = None
        interval: tuple[float, float] | None = None
        for row, timestamp in zip(data, times):
            state = _decode(row[0]).strip().lower()
            calibration_type = _decode(row[3]).strip()
            if calibration_type != label:
                continue
            if state == "start":
                start_time = float(timestamp)
            elif state == "stop" and start_time is not None:
                interval = (start_time, float(timestamp))
                break
        if interval is None:
            raise ValueError(f"no valid {label!r} interval in {self.path}")

        from .emg2pose import (
            estimate_open_hand_neutral_degrees,
            map_hand_pose_to_emg2pose,
        )

        pose = self._read_timed(
            "/xsens-joints/rotation_xzy_deg/data",
            "/xsens-joints/rotation_xzy_deg/time_s",
            *interval,
        )
        target = map_hand_pose_to_emg2pose(pose, side=side)
        return estimate_open_hand_neutral_degrees(target)

    def load_activity(
        self,
        selector: int | str,
        *,
        include_tactile: bool = True,
        include_video_timestamps: bool = True,
    ) -> SessionSequence:
        """Load one activity interval without loading the complete recording."""
        interval = self._select_activity(selector)
        start, end = interval.start_time, interval.end_time

        tactile_left = tactile_right = None
        if include_tactile:
            tactile_left = self._read_timed(
                "/tactile-glove-left/tactile_data/data",
                "/tactile-glove-left/tactile_data/time_s",
                start,
                end,
            )
            tactile_right = self._read_timed(
                "/tactile-glove-right/tactile_data/data",
                "/tactile-glove-right/tactile_data/time_s",
                start,
                end,
            )

        rgb_timestamps = None
        rgb_frame_indices = None
        if include_video_timestamps:
            rgb_timestamps = self._read_timed(
                self.WORLD_VIDEO_DATA,
                self.WORLD_VIDEO_DATA,
                start,
                end,
            )
            if self.video_path is not None:
                first = int(np.searchsorted(
                    np.asarray(self.file[self.WORLD_VIDEO_DATA][:]).reshape(-1),
                    start,
                    side="left",
                ))
                rgb_frame_indices = np.arange(
                    first,
                    first + len(rgb_timestamps.timestamps),
                    dtype=np.int64,
                )

        return SessionSequence(
            subject=self.subject,
            session=self.session,
            activity=interval.activity,
            start_time=start,
            end_time=end,
            emg_left=self._read_timed(
                "/myo-left/emg/data", "/myo-left/emg/time_s", start, end
            ),
            emg_right=self._read_timed(
                "/myo-right/emg/data", "/myo-right/emg/time_s", start, end
            ),
            hand_pose=self._read_timed(
                "/xsens-joints/rotation_xzy_deg/data",
                "/xsens-joints/rotation_xzy_deg/time_s",
                start,
                end,
            ),
            tactile_left=tactile_left,
            tactile_right=tactile_right,
            rgb_timestamps=rgb_timestamps,
            rgb_video_path=str(self.video_path) if self.video_path is not None else None,
            rgb_frame_indices=rgb_frame_indices,
        )

    def iter_activities(
        self,
        *,
        include_tactile: bool = True,
        include_video_timestamps: bool = True,
    ) -> Iterator[SessionSequence]:
        """Yield activity windows one at a time for memory-bounded training."""
        for activity_index, _ in enumerate(self.activities()):
            yield self.load_activity(
                activity_index,
                include_tactile=include_tactile,
                include_video_timestamps=include_video_timestamps,
            )

"""Temporal alignment for :class:`SessionSequence` objects."""

from __future__ import annotations

import numpy as np

from .types import SessionSequence, Stream


def _linear_resample(stream: Stream, timestamps: np.ndarray) -> np.ndarray:
    values = np.asarray(stream.values)
    if values.shape[0] == 0:
        raise ValueError(f"cannot resample empty stream {stream.source}")
    flat = values.reshape(values.shape[0], -1).astype(np.float64, copy=False)
    output = np.empty((len(timestamps), flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(timestamps, stream.timestamps, flat[:, column])
    return output.reshape((len(timestamps),) + values.shape[1:])


def _nearest_resample(stream: Stream, timestamps: np.ndarray) -> np.ndarray:
    values = np.asarray(stream.values)
    if values.shape[0] == 0:
        raise ValueError(f"cannot resample empty stream {stream.source}")
    right = np.searchsorted(stream.timestamps, timestamps, side="left")
    right = np.clip(right, 0, len(stream.timestamps) - 1)
    left = np.clip(right - 1, 0, len(stream.timestamps) - 1)
    use_right = np.abs(stream.timestamps[right] - timestamps) < np.abs(
        timestamps - stream.timestamps[left]
    )
    return values[np.where(use_right, right, left)]


def _window_rms(stream: Stream, timestamps: np.ndarray, step: float) -> np.ndarray:
    """Compute one RMS feature per target interval for high-rate EMG."""
    values = np.asarray(stream.values)
    flat = values.reshape(values.shape[0], -1).astype(np.float64, copy=False)
    starts = np.searchsorted(stream.timestamps, timestamps, side="left")
    ends = np.searchsorted(stream.timestamps, timestamps + step, side="left")
    output = np.empty((len(timestamps), flat.shape[1]), dtype=np.float64)
    for row, (first, last) in enumerate(zip(starts, ends)):
        if first < last:
            output[row] = np.sqrt(np.mean(np.square(flat[first:last]), axis=0))
        else:
            nearest = min(max(int(first), 0), len(flat) - 1)
            output[row] = np.abs(flat[nearest])
    return output.reshape((len(timestamps),) + values.shape[1:])


def _target_timestamps(sequence: SessionSequence, target_hz: float) -> np.ndarray:
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")
    streams = [sequence.emg_left, sequence.emg_right, sequence.hand_pose]
    if sequence.tactile_left is not None and sequence.tactile_right is not None:
        streams.extend([sequence.tactile_left, sequence.tactile_right])
    if sequence.rgb_timestamps is not None:
        streams.append(sequence.rgb_timestamps)
    start = max(stream.start_time for stream in streams)
    end = min(stream.end_time for stream in streams)
    if end <= start:
        raise ValueError("streams have no common time interval")
    step = 1.0 / target_hz
    count = int(np.floor((end - start) * target_hz)) + 1
    return start + np.arange(count, dtype=np.float64) * step


def align_sequence(
    sequence: SessionSequence,
    target_hz: float = 30.0,
    *,
    emg_mode: str = "rms",
) -> SessionSequence:
    """Resample every available stream to one regular timeline.

    EMG defaults to RMS windows because raw Myo samples are about 160 Hz and
    direct interpolation to 30 Hz would discard most of their waveform. Pose
    uses linear interpolation; tactile and RGB frame timing use nearest samples.
    The input sequence is never modified.
    """
    if emg_mode not in {"rms", "linear"}:
        raise ValueError("emg_mode must be 'rms' or 'linear'")
    timestamps = _target_timestamps(sequence, target_hz)
    step = 1.0 / target_hz

    if emg_mode == "rms":
        left_emg = _window_rms(sequence.emg_left, timestamps, step)
        right_emg = _window_rms(sequence.emg_right, timestamps, step)
    else:
        left_emg = _linear_resample(sequence.emg_left, timestamps)
        right_emg = _linear_resample(sequence.emg_right, timestamps)

    tactile_left = tactile_right = None
    if sequence.tactile_left is not None and sequence.tactile_right is not None:
        tactile_left = Stream(
            timestamps=timestamps,
            values=_nearest_resample(sequence.tactile_left, timestamps),
            source=sequence.tactile_left.source,
            headings=sequence.tactile_left.headings,
        )
        tactile_right = Stream(
            timestamps=timestamps,
            values=_nearest_resample(sequence.tactile_right, timestamps),
            source=sequence.tactile_right.source,
            headings=sequence.tactile_right.headings,
        )

    rgb_timestamps = None
    rgb_frames = None
    rgb_frame_indices = None
    if sequence.rgb_timestamps is not None:
        rgb_timestamps = Stream(
            timestamps=timestamps,
            values=_nearest_resample(sequence.rgb_timestamps, timestamps),
            source=sequence.rgb_timestamps.source,
            headings=sequence.rgb_timestamps.headings,
        )
        if sequence.rgb_frames is not None:
            source_times = sequence.rgb_timestamps.timestamps
            right = np.searchsorted(source_times, timestamps, side="left")
            right = np.clip(right, 0, len(source_times) - 1)
            left = np.clip(right - 1, 0, len(source_times) - 1)
            use_right = np.abs(source_times[right] - timestamps) < np.abs(
                timestamps - source_times[left]
            )
            indices = np.where(use_right, right, left)
            rgb_frames = [sequence.rgb_frames[int(index)] for index in indices]
        if sequence.rgb_frame_indices is not None:
            source_times = sequence.rgb_timestamps.timestamps
            source_indices = np.asarray(sequence.rgb_frame_indices).reshape(-1)
            if len(source_indices) != len(source_times):
                raise ValueError("RGB frame indices and timestamps must have the same length")
            index_stream = Stream(
                timestamps=source_times,
                values=source_indices[:, None],
                source=sequence.rgb_video_path or "rgb-video",
            )
            rgb_frame_indices = _nearest_resample(index_stream, timestamps).reshape(-1).astype(
                np.int64,
                copy=False,
            )

    return SessionSequence(
        subject=sequence.subject,
        session=sequence.session,
        activity=sequence.activity,
        start_time=float(timestamps[0]),
        end_time=float(timestamps[-1]),
        emg_left=Stream(
            timestamps, left_emg, sequence.emg_left.source, sequence.emg_left.headings
        ),
        emg_right=Stream(
            timestamps, right_emg, sequence.emg_right.source, sequence.emg_right.headings
        ),
        hand_pose=Stream(
            timestamps,
            _linear_resample(sequence.hand_pose, timestamps),
            sequence.hand_pose.source,
            sequence.hand_pose.headings,
        ),
        tactile_left=tactile_left,
        tactile_right=tactile_right,
        rgb_timestamps=rgb_timestamps,
        rgb_frames=rgb_frames,
        rgb_video_path=sequence.rgb_video_path,
        rgb_frame_indices=rgb_frame_indices,
    )

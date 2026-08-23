"""Prepare ActionSense windows for Meta's EMG2Pose training code.

The exported files intentionally contain one hand and one regular 30 Hz clock.
This matches the shape expected by ``facebookresearch/emg2pose`` while avoiding
its original 2 kHz-to-25 Hz temporal striding.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Literal

import h5py
import numpy as np

from .aligner import _linear_resample, _window_rms
from .data_loader import ActionSenseLoader
from .emg2pose import (
    EMG2POSE_JOINT_NAMES,
    Emg2PoseSequence,
    map_hand_pose_to_emg2pose,
)
from .types import SessionSequence, Stream


def _training_timestamps(
    emg: Stream,
    hand_pose: Stream,
    target_hz: float,
) -> np.ndarray:
    """Build a regular clock from only EMG and pose.

    Optional video and tactile streams must not shorten an EMG2Pose training
    window, so this deliberately ignores them.
    """
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")
    start = max(emg.start_time, hand_pose.start_time)
    end = min(emg.end_time, hand_pose.end_time)
    if end <= start:
        raise ValueError("EMG and hand-pose streams have no common interval")
    step = 1.0 / target_hz
    count = int(np.floor((end - start) * target_hz)) + 1
    return start + np.arange(count, dtype=np.float64) * step


def align_for_emg2pose(
    sequence: SessionSequence,
    *,
    side: Literal["left", "right"],
    target_hz: float = 30.0,
    emg_mode: Literal["rms", "linear"] = "rms",
) -> Emg2PoseSequence:
    """Return one side at a regular training rate with a ``(T, 20)`` target.

    ``rms`` computes one EMG envelope feature in each target-rate interval.
    ``linear`` is available for experiments that explicitly want interpolation.
    No additional temporal downsampling is performed after this function.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if emg_mode not in {"rms", "linear"}:
        raise ValueError("emg_mode must be 'rms' or 'linear'")

    emg = sequence.emg_left if side == "left" else sequence.emg_right
    timestamps = _training_timestamps(emg, sequence.hand_pose, target_hz)
    if emg_mode == "rms":
        emg_values = _window_rms(emg, timestamps, 1.0 / target_hz)
    else:
        emg_values = _linear_resample(emg, timestamps)

    aligned_emg = Stream(
        timestamps=timestamps,
        values=emg_values,
        source=emg.source,
        headings=emg.headings,
    )
    aligned_pose = Stream(
        timestamps=timestamps,
        values=_linear_resample(sequence.hand_pose, timestamps),
        source=sequence.hand_pose.source,
        headings=sequence.hand_pose.headings,
    )
    target = map_hand_pose_to_emg2pose(aligned_pose, side=side)

    return Emg2PoseSequence(
        subject=sequence.subject,
        session=sequence.session,
        activity=sequence.activity,
        side=side,
        emg=aligned_emg,
        target=target,
        emg_preprocessing=(
            f"{target_hz:g} Hz RMS per target interval"
            if emg_mode == "rms"
            else f"{target_hz:g} Hz linear interpolation"
        ),
    )


def _validate_paired_sequence(sequence: Emg2PoseSequence) -> None:
    emg = np.asarray(sequence.emg.values)
    target = np.asarray(sequence.target.joint_angles)
    emg_t = np.asarray(sequence.emg.timestamps)
    target_t = np.asarray(sequence.target.timestamps)
    if emg.ndim != 2 or emg.shape[1] != 8:
        raise ValueError(f"expected EMG shape (T, 8), got {emg.shape}")
    if target.ndim != 2 or target.shape[1] != 20:
        raise ValueError(f"expected target shape (T, 20), got {target.shape}")
    if emg.shape[0] != target.shape[0] or not np.allclose(emg_t, target_t):
        raise ValueError("EMG and target must have the same timestamps")
    if len(emg_t) < 2 or not np.all(np.diff(emg_t) > 0):
        raise ValueError("EMG2Pose export needs at least two increasing timestamps")
    if not np.isfinite(emg).all() or not np.isfinite(target).all():
        raise ValueError("EMG2Pose export contains NaN or infinite values")


def write_emg2pose_hdf5(
    sequence: Emg2PoseSequence,
    path: str | Path,
    *,
    emg_scale: float | None = 128.0,
    overwrite: bool = False,
) -> Path:
    """Write one aligned hand/activity in Meta's compound-HDF5 format.

    ``emg_scale=128`` converts ActionSense's documented Myo integer range to
    approximately ``[-1, 1]``. Pass ``None`` to preserve the RMS values in
    their source units.
    """
    _validate_paired_sequence(sequence)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")

    emg = np.asarray(sequence.emg.values, dtype=np.float32)
    if emg_scale is not None:
        if emg_scale <= 0:
            raise ValueError("emg_scale must be positive or None")
        emg = emg / np.float32(emg_scale)
    target = np.asarray(sequence.target.joint_angles, dtype=np.float32)
    timestamps = np.asarray(sequence.emg.timestamps, dtype=np.float64)

    dtype = np.dtype(
        [
            ("emg", np.float32, (8,)),
            ("joint_angles", np.float32, (20,)),
            ("time", np.float64),
        ]
    )
    timeseries = np.empty(len(timestamps), dtype=dtype)
    timeseries["emg"] = emg
    timeseries["joint_angles"] = target
    timeseries["time"] = timestamps

    with h5py.File(output, "w") as file:
        group = file.create_group("emg2pose")
        group.create_dataset("timeseries", data=timeseries, chunks=True)
        group.attrs.update(
            {
                "session": sequence.session,
                "stage": sequence.activity,
                "user": sequence.subject,
                "side": sequence.side,
                "start": float(timestamps[0]),
                "end": float(timestamps[-1]),
                "num_channels": 8,
                # Epoch-sized float timestamps have a tiny ULP effect on the
                # measured interval; metadata should report the intended
                # regular rate rather than 30.000028 Hz.
                "sample_rate": float(np.round(1.0 / np.median(np.diff(timestamps)), 3)),
                "dataset": "ActionSense",
                "target_joint_names": np.asarray(
                    EMG2POSE_JOINT_NAMES, dtype=h5py.string_dtype()
                ),
                "emg_preprocessing": sequence.emg_preprocessing
                + (
                    "; divided by {emg_scale:g}".format(emg_scale=emg_scale)
                    if emg_scale is not None
                    else "; source units"
                ),
            }
        )
    return output


def export_action_sense_recording(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    sides: Iterable[Literal["left", "right"]] = ("left", "right"),
    target_hz: float = 30.0,
    emg_mode: Literal["rms", "linear"] = "rms",
    emg_scale: float | None = 128.0,
    include_invalid: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Export every labeled ActionSense activity as one file per hand.

    Only one activity window is held in memory at a time. Invalid activity
    labels are skipped by default.
    """
    chosen_sides = tuple(sides)
    if not chosen_sides or any(side not in {"left", "right"} for side in chosen_sides):
        raise ValueError("sides must contain only 'left' and/or 'right'")

    output_root = Path(output_dir)
    written: list[Path] = []
    with ActionSenseLoader(source_path) as loader:
        for activity_index, interval in enumerate(loader.activities()):
            if not include_invalid and interval.valid.strip().lower() not in {"", "good"}:
                continue
            sequence = loader.load_activity(
                activity_index,
                include_tactile=False,
                include_video_timestamps=False,
            )
            for side in chosen_sides:
                paired = align_for_emg2pose(
                    sequence,
                    side=side,
                    target_hz=target_hz,
                    emg_mode=emg_mode,
                )
                filename = (
                    f"{loader.subject}_{loader.session}_activity-{activity_index:03d}_{side}.hdf5"
                )
                written.append(
                    write_emg2pose_hdf5(
                        paired,
                        output_root / filename,
                        emg_scale=emg_scale,
                        overwrite=overwrite,
                    )
                )
    return written


def write_emg2pose_metadata(
    paths: Iterable[str | Path],
    metadata_path: str | Path,
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Path:
    """Write the ``metadata.csv`` consumed by Meta's data-split loader.

    Left/right files from the same ActionSense activity are assigned to the
    same split. Splitting by activity avoids putting two views of one motion
    in different partitions while still allowing subject/session-specific
    experiments. Meta's loader expects filename stems and appends ``.hdf5``.
    """
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must leave training data")

    rows: list[dict[str, object]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with h5py.File(path, "r") as file:
            attrs = file["emg2pose"].attrs
            def attr(name: str, default: object = "") -> object:
                value = attrs.get(name, default)
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="replace")
                return value

            rows.append(
                {
                    "filename": path.stem,
                    "user": attr("user"),
                    "session": attr("session"),
                    "stage": attr("stage"),
                    "side": attr("side"),
                    "sample_rate": attr("sample_rate"),
                    "num_channels": attr("num_channels"),
                    "start": attr("start"),
                    "end": attr("end"),
                    "dataset": attr("dataset", "ActionSense"),
                }
            )

    if not rows:
        raise ValueError("cannot write metadata for an empty path list")

    groups = sorted(
        {
            (row["user"], row["session"], row["stage"])
            for row in rows
        },
        key=str,
    )
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n_groups = len(shuffled)
    n_test = int(round(n_groups * test_fraction))
    n_val = int(round(n_groups * val_fraction))
    if n_groups >= 3:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
        if n_test + n_val >= n_groups:
            n_test, n_val = 1, 1
    split_by_group = {
        group: "test" if index < n_test else
        "val" if index < n_test + n_val else "train"
        for index, group in enumerate(shuffled)
    }
    for row in rows:
        group = (row["user"], row["session"], row["stage"])
        row["split"] = split_by_group[group]

    output = Path(metadata_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename", "user", "session", "stage", "side", "split",
        "sample_rate", "num_channels", "start", "end", "dataset",
    ]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output

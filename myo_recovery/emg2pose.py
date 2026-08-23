"""Convert ActionSense Xsens hand angles to the EMG2Pose target convention.

The target ordering mirrors ``facebookresearch/emg2pose/emg2pose/constants.py``:
20 scalar angles per hand. The Xsens stream already names the corresponding
anatomical components, so this adapter selects by heading rather than relying
on fragile numeric column positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .types import Stream


# Mirrors Meta's JOINTS list in:
# https://github.com/facebookresearch/emg2pose/blob/main/emg2pose/constants.py
EMG2POSE_JOINT_NAMES: tuple[str, ...] = (
    "THUMB_CMC_FE",
    "THUMB_CMC_AA",
    "THUMB_MCP_FE",
    "THUMB_IP_FE",
    "INDEX_MCP_AA",
    "INDEX_MCP_FE",
    "INDEX_PIP_FE",
    "INDEX_DIP_FE",
    "MIDDLE_MCP_AA",
    "MIDDLE_MCP_FE",
    "MIDDLE_PIP_FE",
    "MIDDLE_DIP_FE",
    "RING_MCP_AA",
    "RING_MCP_FE",
    "RING_PIP_FE",
    "RING_DIP_FE",
    "PINKY_MCP_AA",
    "PINKY_MCP_FE",
    "PINKY_PIP_FE",
    "PINKY_DIP_FE",
)


@dataclass(slots=True)
class Emg2PoseTarget:
    """One hand's EMG2Pose-compatible ground-truth angles.

    ``joint_angles`` is ``(T, 20)`` in radians, with the exact ordering in
    ``EMG2POSE_JOINT_NAMES``. The source Xsens stream is degrees, so
    ``joint_angles_degrees`` is provided for auditing and plotting.
    """

    timestamps: np.ndarray
    joint_angles: np.ndarray
    side: Literal["left", "right"]
    source: str

    @property
    def joint_angles_degrees(self) -> np.ndarray:
        return np.rad2deg(self.joint_angles)


@dataclass(slots=True)
class Emg2PoseSequence:
    """One side's EMG plus its mapped EMG2Pose target.

    The native mapper keeps separate clocks; the explicit export path stores
    paired fixed-rate streams and records its preprocessing here.
    """

    subject: str
    session: str
    activity: str
    side: Literal["left", "right"]
    emg: Stream
    target: Emg2PoseTarget
    emg_preprocessing: str = "native-rate"


def _source_headings(side: Literal["left", "right"]) -> tuple[str, ...]:
    hand = side.capitalize()
    return (
        f"{hand} First CMC Flexion/Extension",
        f"{hand} First CMC Abduction/Adduction",
        f"{hand} First MCP Flexion/Extension",
        f"{hand} IP Flexion/Extension",
        f"{hand} Second MCP Abduction/Adduction",
        f"{hand} Second MCP Flexion/Extension",
        f"{hand} Second PIP Flexion/Extension",
        f"{hand} Second DIP Flexion/Extension",
        f"{hand} Third MCP Abduction/Adduction",
        f"{hand} Third MCP Flexion/Extension",
        f"{hand} Third PIP Flexion/Extension",
        f"{hand} Third DIP Flexion/Extension",
        f"{hand} Fourth MCP Abduction/Adduction",
        f"{hand} Fourth MCP Flexion/Extension",
        f"{hand} Fourth PIP Flexion/Extension",
        f"{hand} Fourth DIP Flexion/Extension",
        f"{hand} Fifth MCP Abduction/Adduction",
        f"{hand} Fifth MCP Flexion/Extension",
        f"{hand} Fifth PIP Flexion/Extension",
        f"{hand} Fifth DIP Flexion/Extension",
    )


def map_hand_pose_to_emg2pose(
    hand_pose: Stream,
    *,
    side: Literal["left", "right"],
) -> Emg2PoseTarget:
    """Select Xsens anatomical components into Meta's 20-D hand target.

    This performs only the representation conversion and degrees-to-radians
    conversion. It does not guess sign flips or neutral offsets; those must be
    checked against the recording's calibration and documented separately.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    values = np.asarray(hand_pose.values)
    if values.ndim != 3:
        raise ValueError(f"expected hand pose shape (T, joints, components), got {values.shape}")
    if not hand_pose.headings:
        raise ValueError("hand pose stream has no Data headings metadata")

    flat = values.reshape(values.shape[0], -1)
    if len(hand_pose.headings) != flat.shape[1]:
        raise ValueError(
            f"heading count ({len(hand_pose.headings)}) does not match flattened "
            f"pose width ({flat.shape[1]})"
        )
    heading_to_column = {heading: index for index, heading in enumerate(hand_pose.headings)}
    source_headings = _source_headings(side)
    missing = [heading for heading in source_headings if heading not in heading_to_column]
    if missing:
        raise ValueError("missing Xsens headings required by EMG2Pose: " + "; ".join(missing))

    columns = [heading_to_column[heading] for heading in source_headings]
    angles_degrees = flat[:, columns].astype(np.float64, copy=False)
    angles_radians = np.deg2rad(angles_degrees)
    if not np.isfinite(angles_radians).all():
        raise ValueError("EMG2Pose target contains NaN or infinite joint angles")

    return Emg2PoseTarget(
        timestamps=hand_pose.timestamps.copy(),
        joint_angles=angles_radians,
        side=side,
        source=hand_pose.source,
    )


def make_emg2pose_sequence(
    sequence: "SessionSequence",
    *,
    side: Literal["left", "right"],
) -> Emg2PoseSequence:
    """Pair one native-rate Myo stream with that hand's 20-angle target."""
    # Local import avoids a module cycle at import time.
    from .types import SessionSequence

    if not isinstance(sequence, SessionSequence):
        raise TypeError("sequence must be a SessionSequence")
    emg = sequence.emg_left if side == "left" else sequence.emg_right
    return Emg2PoseSequence(
        subject=sequence.subject,
        session=sequence.session,
        activity=sequence.activity,
        side=side,
        emg=emg,
        target=map_hand_pose_to_emg2pose(sequence.hand_pose, side=side),
        emg_preprocessing="native-rate",
    )

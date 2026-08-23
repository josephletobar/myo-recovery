#!/usr/bin/env python3
"""Render an ActionSense RGB clip beside an EMG2Pose hand reconstruction.

This script is intended to run in the Meta ``emg2pose`` environment on the
Linux workstation.  It runs the trained checkpoint on one prepared
ActionSense HDF5 file, reconstructs the generic hand landmarks from the
predicted and ground-truth joint angles, and writes a small side-by-side MP4.

The source video should already be a short clip for the same activity.  Keeping
the source clip short avoids copying the full multi-gigabyte ActionSense video
to the compute machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np


def _load_model(
    *,
    emg2pose_root: Path,
    checkpoint: Path,
    config: Path,
    device_name: str,
) -> tuple[Any, Any, str]:
    """Load the trained Meta EMG2Pose module from the saved Hydra config."""

    sys.path.insert(0, str(emg2pose_root))
    from omegaconf import OmegaConf
    import torch
    from emg2pose.lightning import Emg2PoseModule

    cfg = OmegaConf.load(config)
    module = Emg2PoseModule.load_from_checkpoint(
        str(checkpoint),
        network_conf=cfg.pose_module,
        optimizer_conf=cfg.optimizer,
        lr_scheduler_conf=cfg.lr_scheduler,
        provide_initial_pos=cfg.provide_initial_pos,
        loss_weights=cfg.loss_weights,
        sample_rate=cfg.sample_rate,
    )
    device = torch.device(device_name)
    module.eval().to(device)
    return module, torch, device


def _infer(
    *,
    prepared_hdf5: Path,
    module: Any,
    torch: Any,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run one complete activity through the trained model."""

    with h5py.File(prepared_hdf5, "r") as file:
        timeseries = file["emg2pose/timeseries"][:]

    timestamps = np.asarray(timeseries["time"], dtype=np.float64)
    emg = np.asarray(timeseries["emg"], dtype=np.float32)
    target = np.asarray(timeseries["joint_angles"], dtype=np.float32)
    if emg.ndim != 2 or emg.shape[1] != 8:
        raise ValueError(f"expected prepared EMG shape (T, 8), got {emg.shape}")
    if target.ndim != 2 or target.shape[1] != 20:
        raise ValueError(f"expected prepared target shape (T, 20), got {target.shape}")
    if len(timestamps) != len(emg) or len(target) != len(emg):
        raise ValueError("prepared EMG, target, and time lengths differ")
    if len(timestamps) < 4 or not np.all(np.diff(timestamps) > 0):
        raise ValueError("prepared timestamps must be strictly increasing")

    emg_tensor = torch.from_numpy(emg.T[None]).to(device)
    initial_pos = torch.zeros((1, 20), dtype=emg_tensor.dtype, device=device)
    with torch.inference_mode():
        prediction = module.model._predict_pose(emg_tensor, initial_pos)
    prediction = prediction[0].detach().cpu().numpy().T

    left_context = int(module.model.left_context)
    prediction_length = min(len(prediction), len(timestamps) - left_context)
    if prediction_length <= 0:
        raise ValueError("activity is shorter than the model's left context")
    prediction = prediction[:prediction_length]
    target = target[left_context : left_context + prediction_length]
    prediction_times = timestamps[left_context : left_context + prediction_length]
    return prediction_times, prediction, target, left_context


def _landmark_batch(
    angles: np.ndarray,
    *,
    side: str,
    torch: Any,
) -> np.ndarray:
    """Convert a batch of 20-D angle vectors to UmeTrack landmarks."""

    from emg2pose.kinematics import forward_kinematics, load_hand_model_from_json
    from emg2pose.visualization import mirror_profile

    model_path = Path(__file__).resolve().parent / "emg2pose" / "UmeTrack" / "dataset" / "generic_hand_model.json"
    if not model_path.is_file():
        # When this file is copied to the Meta checkout, resolve relative to
        # the imported package instead of the MyoRecovery checkout.
        import emg2pose

        model_path = Path(emg2pose.__file__).resolve().parent / "UmeTrack" / "dataset" / "generic_hand_model.json"
    profile = load_hand_model_from_json(str(model_path))
    if side == "left":
        profile = mirror_profile(profile)

    angle_tensor = torch.from_numpy(angles.T[None]).to(dtype=torch.float32)
    with torch.inference_mode():
        landmarks = forward_kinematics(angle_tensor, profile)[0].detach().cpu().numpy()
    return landmarks


def _project_skeleton(
    landmarks: np.ndarray,
    *,
    width: int,
    height: int,
    color: tuple[int, int, int],
    canvas: np.ndarray,
) -> None:
    """Draw a stable x/z projection of the UmeTrack hand landmarks."""

    import cv2

    # UmeTrack's generic hand model is in millimetres.  These fixed bounds
    # keep the hand stable in the panel while it moves.
    x_min, x_max = -210.0, 210.0
    z_min, z_max = -100.0, 125.0

    def point(index: int) -> tuple[int, int]:
        x, _, z = landmarks[index]
        px = int(round((x - x_min) / (x_max - x_min) * (width - 1)))
        py = int(round((z_max - z) / (z_max - z_min) * (height - 1)))
        return px, py

    # Landmark indices: wrist 5; thumb 6-7-0; fingers proximal/intermediate/
    # distal/fingertip are 8-9-10-1, ..., 17-18-19-4.
    chains = (
        (5, 6, 7, 0),
        (5, 8, 9, 10, 1),
        (5, 11, 12, 13, 2),
        (5, 14, 15, 16, 3),
        (5, 17, 18, 19, 4),
    )
    for chain in chains:
        for first, second in zip(chain, chain[1:]):
            cv2.line(canvas, point(first), point(second), color, 3, cv2.LINE_AA)
    for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
        cv2.circle(canvas, point(index), 5, color, -1, cv2.LINE_AA)


def _hand_panel(
    *,
    pred_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    title: str,
    error_degrees: float,
    width: int,
    height: int,
    overlay: bool,
) -> np.ndarray:
    import cv2

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    if overlay:
        _project_skeleton(
            target_landmarks,
            width=width,
            height=height,
            color=(255, 170, 50),
            canvas=panel,
        )
        _project_skeleton(
            pred_landmarks,
            width=width,
            height=height,
            color=(50, 70, 255),
            canvas=panel,
        )
        legend = "GT blue  |  prediction red"
    else:
        _project_skeleton(
            pred_landmarks,
            width=width,
            height=height,
            color=(50, 70, 255),
            canvas=panel,
        )
        legend = "prediction"
    cv2.putText(panel, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(panel, f"angle MAE {error_degrees:.1f} deg", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, legend, (16, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    return panel


def render(
    *,
    video: Path,
    output: Path,
    prediction_times: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    side: str,
    video_start_offset: float,
    panel_width: int,
    output_fps: float | None,
    torch: Any,
) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open source clip: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = output_fps or source_fps
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("source clip has no readable dimensions")

    display_height = 480
    display_width = int(round(source_width * display_height / source_height))
    output_size = (display_width + 2 * panel_width, display_height)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open output video: {output}")

    frame_count = 0
    video_times: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        video_times.append(video_start_offset + frame_count / fps)
        frame_count += 1
    capture.release()
    if not video_times:
        writer.release()
        raise RuntimeError("source clip contains no frames")

    video_times_array = np.asarray(video_times, dtype=np.float64)
    pred_rel = prediction_times - prediction_times[0]
    query = np.clip(video_times_array, pred_rel[0], pred_rel[-1])
    frame_angles = np.column_stack(
        [np.interp(query, pred_rel, prediction[:, joint]) for joint in range(20)]
    )
    target_angles = np.column_stack(
        [np.interp(query, pred_rel, target[:, joint]) for joint in range(20)]
    )
    pred_landmarks = _landmark_batch(frame_angles, side=side, torch=torch)
    target_landmarks = _landmark_batch(target_angles, side=side, torch=torch)

    # Reopen the source so frames and precomputed panels stay synchronized.
    capture = cv2.VideoCapture(str(video))
    for index in range(len(video_times_array)):
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
        error_degrees = float(np.mean(np.abs(frame_angles[index] - target_angles[index])) * 180.0 / np.pi)
        pred_panel = _hand_panel(
            pred_landmarks=pred_landmarks[index],
            target_landmarks=target_landmarks[index],
            title="Predicted hand",
            error_degrees=error_degrees,
            width=panel_width,
            height=display_height,
            overlay=False,
        )
        overlay_panel = _hand_panel(
            pred_landmarks=pred_landmarks[index],
            target_landmarks=target_landmarks[index],
            title="Prediction vs ground truth",
            error_degrees=error_degrees,
            width=panel_width,
            height=display_height,
            overlay=True,
        )
        cv2.putText(
            frame,
            f"ActionSense RGB  |  frame {index + 1}/{len(video_times_array)}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(np.concatenate((frame, pred_panel, overlay_panel), axis=1))
    capture.release()
    writer.release()
    return frame_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-hdf5", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True, help="short activity clip")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--emg2pose-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--video-start-offset",
        type=float,
        default=0.0,
        help="seconds from the first prediction timestamp to the first clip frame",
    )
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    if args.panel_width < 160:
        parser.error("--panel-width must be at least 160")
    for path, label in (
        (args.prepared_hdf5, "prepared HDF5"),
        (args.video, "source video"),
        (args.checkpoint, "checkpoint"),
        (args.config, "config"),
        (args.emg2pose_root, "EMG2Pose checkout"),
    ):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")

    module, torch, device = _load_model(
        emg2pose_root=args.emg2pose_root,
        checkpoint=args.checkpoint,
        config=args.config,
        device_name=args.device,
    )
    prediction_times, prediction, target, left_context = _infer(
        prepared_hdf5=args.prepared_hdf5,
        module=module,
        torch=torch,
        device=device,
    )
    frame_count = render(
        video=args.video,
        output=args.output,
        prediction_times=prediction_times,
        prediction=prediction,
        target=target,
        side=args.side,
        video_start_offset=args.video_start_offset,
        panel_width=args.panel_width,
        output_fps=args.fps,
        torch=torch,
    )
    print(f"wrote {args.output} ({frame_count} frames; model context {left_context} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

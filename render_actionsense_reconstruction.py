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


def _mesh_panel(
    *,
    pred_angles: np.ndarray,
    target_angles: np.ndarray,
    side: str,
    title: str,
    error_degrees: float,
    width: int,
    height: int,
    overlay: bool,
) -> np.ndarray:
    """Render Meta's actual UmeTrack skinned hand mesh to a video panel."""

    import cv2
    import plotly.graph_objects as go
    from emg2pose import visualization

    flip = side == "left"
    prediction_mesh = visualization.generate_hand_mesh_from_joint_angles(
        pred_angles,
        color="crimson",
        opacity=0.92,
        flip=flip,
        name="prediction",
    )
    traces = [prediction_mesh]
    legend = "prediction mesh"
    if overlay:
        target_mesh = visualization.generate_hand_mesh_from_joint_angles(
            target_angles,
            color="deepskyblue",
            opacity=0.72,
            flip=flip,
            name="ground truth",
        )
        traces = [target_mesh, prediction_mesh]
        legend = "GT blue  |  prediction red"

    figure = go.Figure(data=traces)
    figure = visualization._set_3d_plot_layout(
        figure,
        flip=flip,
        clean_background=True,
    )
    figure.update_layout(
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="black",
        plot_bgcolor="black",
        showlegend=False,
    )
    panel = visualization.fig_to_array(figure)[..., :3]
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
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
    # Reopen the source so frames and precomputed panels stay synchronized.
    capture = cv2.VideoCapture(str(video))
    for index in range(len(video_times_array)):
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
        error_degrees = float(np.mean(np.abs(frame_angles[index] - target_angles[index])) * 180.0 / np.pi)
        pred_panel = _mesh_panel(
            pred_angles=frame_angles[index],
            target_angles=target_angles[index],
            side=side,
            title="Predicted hand",
            error_degrees=error_degrees,
            width=panel_width,
            height=display_height,
            overlay=False,
        )
        overlay_panel = _mesh_panel(
            pred_angles=frame_angles[index],
            target_angles=target_angles[index],
            side=side,
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
    )
    print(f"wrote {args.output} ({frame_count} frames; model context {left_context} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render an ActionSense RGB clip beside an EMG2Pose hand reconstruction.

This script is intended to run in the Meta ``emg2pose`` environment on the
Linux workstation.  It runs the trained checkpoint on one prepared
ActionSense HDF5 file and writes a side-by-side MP4 using the same
``plot_hand_mesh``/``joint_angles_to_frames_parallel`` path as Meta's
``notebooks/getting_started.ipynb``.

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


def _mesh_frames(
    angles: np.ndarray,
    *,
    color: str,
    flip: bool,
    width: int,
    height: int,
    n_jobs: int = 4,
) -> np.ndarray:
    """Render frames through Meta's notebook visualization helpers.

    The notebook calls ``joint_angles_to_frames_parallel`` directly.  That
    helper calls ``plot_hand_mesh`` for every frame, which in turn uses the
    UmeTrack generic hand model and Meta's camera/layout.  Keeping this path
    intact avoids a second, subtly different Plotly renderer here.
    """

    import cv2
    from emg2pose import visualization

    frames = visualization.joint_angles_to_frames_parallel(
        angles,
        n_jobs=n_jobs,
        color=color,
        opacity=1.0,
        flip=flip,
        auto_range=False,
        clean_background=True,
    )
    frames = visualization.remove_alpha_channel(frames)
    return np.asarray(
        [
            cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            for frame in frames
        ]
    )


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

    # This is intentionally the same public rendering path used in
    # emg2pose/notebooks/getting_started.ipynb:
    #
    #   visualization.joint_angles_to_frames_parallel(..., color=...)
    #
    # The resulting panels are the notebook's separate GT/pred videos, rather
    # than a custom transparent overlay or a hand-written Mesh3d scene.
    flip = side == "left"
    gt_panels = _mesh_frames(
        target_angles,
        color="gray",
        flip=flip,
        width=panel_width,
        height=display_height,
    )
    pred_panels = _mesh_frames(
        frame_angles,
        color="lightpink",
        flip=flip,
        width=panel_width,
        height=display_height,
    )

    # Reopen the source so frames and precomputed panels stay synchronized.
    capture = cv2.VideoCapture(str(video))
    for index in range(len(video_times_array)):
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
        error_degrees = float(np.mean(np.abs(frame_angles[index] - target_angles[index])) * 180.0 / np.pi)
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
        gt_panel = gt_panels[index].copy()
        pred_panel = pred_panels[index].copy()
        cv2.putText(gt_panel, "Ground truth", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 70, 70), 2, cv2.LINE_AA)
        cv2.putText(gt_panel, f"angle MAE {error_degrees:.1f} deg", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(pred_panel, "Prediction", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 70, 70), 2, cv2.LINE_AA)
        cv2.putText(pred_panel, f"angle MAE {error_degrees:.1f} deg", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
        writer.write(np.concatenate((frame, gt_panel, pred_panel), axis=1))
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

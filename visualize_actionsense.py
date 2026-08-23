#!/usr/bin/env python3
"""Interactively inspect one ActionSense activity.

The viewer keeps the HDF5 data windowed through ``ActionSenseLoader`` and
decodes only the currently selected MP4 frame.  It therefore does not load a
large video or the complete recording into memory.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from myo_recovery import ActionSenseLoader


def _activity_selector(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _decode_frame(video_path: Path, frame_index: int, fps: float) -> np.ndarray:
    """Decode one frame with ffmpeg and return an RGB image array."""
    seconds = max(0.0, frame_index / fps)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{seconds:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"could not decode frame {frame_index}: {detail}")

    # Importing matplotlib here keeps the loader usable without visualization
    # dependencies installed.
    import matplotlib.image as mpimg

    return np.asarray(mpimg.imread(BytesIO(result.stdout), format="png"))


def _prepare_emg(values: np.ndarray, normalize: bool) -> np.ndarray:
    emg = np.asarray(values, dtype=np.float64)
    if not normalize:
        return emg
    center = np.nanmedian(emg, axis=0, keepdims=True)
    scale = np.nanstd(emg, axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0
    return (emg - center) / scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--activity", default="0", help="Activity index or exact label")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument(
        "--raw-emg",
        action="store_true",
        help="Plot source EMG units instead of per-channel z-scores",
    )
    args = parser.parse_args()

    if not args.hdf5.is_file():
        parser.error(f"HDF5 file does not exist: {args.hdf5}")
    if not args.video.is_file():
        parser.error(f"video file does not exist: {args.video}")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    if shutil.which("ffmpeg") is None:
        parser.error("the viewer needs ffmpeg on PATH to decode MP4 frames")

    # Matplotlib is an optional visualization dependency.
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
    except ImportError as exc:
        parser.error(
            "visualization requires matplotlib; install it with "
            "'.venv/bin/python -m pip install matplotlib'"
        )
        raise exc  # pragma: no cover

    with ActionSenseLoader(args.hdf5, video_path=args.video) as loader:
        sequence = loader.load_activity(
            _activity_selector(args.activity),
            include_tactile=False,
            include_video_timestamps=True,
        )

    if sequence.rgb_timestamps is None or sequence.rgb_frame_indices is None:
        parser.error("the HDF5 activity has no world-video frame timestamps")
    if len(sequence.rgb_frame_indices) == 0:
        parser.error("the selected activity has no video frames")

    emg_stream = sequence.emg_left if args.side == "left" else sequence.emg_right
    emg_time = emg_stream.timestamps - sequence.start_time
    emg = _prepare_emg(emg_stream.values, normalize=not args.raw_emg)
    max_points = 8_000
    stride = max(1, len(emg_time) // max_points)
    plot_time = emg_time[::stride]
    plot_emg = emg[::stride]

    video_time = sequence.rgb_timestamps.timestamps - sequence.start_time
    frame_indices = np.asarray(sequence.rgb_frame_indices, dtype=np.int64)

    figure = plt.figure(figsize=(12, 8))
    grid = figure.add_gridspec(2, 1, height_ratios=(1.25, 1.0), hspace=0.28)
    image_axis = figure.add_subplot(grid[0])
    emg_axis = figure.add_subplot(grid[1])
    figure.subplots_adjust(bottom=0.16)

    image_axis.set_axis_off()
    image_axis.set_title(f"{sequence.subject} · {sequence.activity} · {args.side} hand")
    emg_axis.set_xlabel("Time from activity start (s)")
    emg_axis.set_ylabel("EMG channel (z-score)" if not args.raw_emg else "EMG channel")
    offsets = np.arange(emg.shape[1], dtype=float) * 3.0
    for channel in range(emg.shape[1]):
        emg_axis.plot(
            plot_time,
            plot_emg[:, channel] + offsets[channel],
            linewidth=0.7,
            label=f"ch {channel + 1}",
        )
    emg_axis.set_yticks(offsets)
    emg_axis.set_yticklabels([f"{i + 1}" for i in range(emg.shape[1])])
    emg_axis.grid(axis="x", alpha=0.25)
    cursor = emg_axis.axvline(float(video_time[0]), color="tab:red", linewidth=1.2)

    slider_axis = figure.add_axes((0.12, 0.07, 0.68, 0.035))
    slider = Slider(
        slider_axis,
        "Video frame",
        0,
        len(frame_indices) - 1,
        valinit=0,
        valstep=1,
    )
    button_axis = figure.add_axes((0.83, 0.055, 0.10, 0.065))
    play_button = Button(button_axis, "Play")
    image_artist: Any = None
    state = {"playing": False}

    def update(local_index: float) -> None:
        nonlocal image_artist
        index = int(round(local_index))
        try:
            image = _decode_frame(args.video, int(frame_indices[index]), args.video_fps)
            if image_artist is None:
                image_artist = image_axis.imshow(image)
            else:
                image_artist.set_data(image)
            image_axis.set_title(
                f"{sequence.subject} · {sequence.activity} · {args.side} hand · "
                f"t={video_time[index]:.2f}s · frame={frame_indices[index]}"
            )
        except RuntimeError as exc:
            image_axis.clear()
            image_axis.set_axis_off()
            image_axis.text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True)
            image_artist = None
        cursor.set_xdata([video_time[index], video_time[index]])
        figure.canvas.draw_idle()

    def toggle_play(_event: Any) -> None:
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")

    def tick(_event: Any) -> None:
        if state["playing"]:
            next_index = int(slider.val) + 1
            if next_index >= len(frame_indices):
                state["playing"] = False
                play_button.label.set_text("Play")
            else:
                slider.set_val(next_index)

    slider.on_changed(update)
    play_button.on_clicked(toggle_play)
    timer = figure.canvas.new_timer(interval=max(1, int(round(1000 / args.video_fps))))
    timer.add_callback(tick, None)
    timer.start()
    update(0)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

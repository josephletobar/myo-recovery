#!/usr/bin/env python3
"""Export ActionSense activities into EMG2Pose-compatible HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

from myo_recovery.emg2pose_export import (
    export_action_sense_recording,
    write_emg2pose_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="ActionSense wearable HDF5 file")
    parser.add_argument("output_dir", type=Path, help="directory for per-activity HDF5 files")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--target-hz", type=float, default=30.0)
    parser.add_argument("--emg-mode", choices=("rms", "linear"), default="rms")
    parser.add_argument(
        "--angle-calibration",
        choices=("none", "xsens-open-hand"),
        default="none",
        help="apply the recorded Manus open-hand neutral offset",
    )
    parser.add_argument(
        "--emg-scale",
        type=float,
        default=128.0,
        help="divide EMG features by this value; use 0 to preserve source units",
    )
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="metadata.csv path; defaults to OUTPUT_DIR/metadata.csv",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sides = ("left", "right") if args.side == "both" else (args.side,)
    paths = export_action_sense_recording(
        args.source,
        args.output_dir,
        sides=sides,
        target_hz=args.target_hz,
        emg_mode=args.emg_mode,
        emg_scale=None if args.emg_scale == 0 else args.emg_scale,
        angle_calibration=args.angle_calibration,
        include_invalid=args.include_invalid,
        overwrite=args.overwrite,
    )
    metadata_path = args.metadata or args.output_dir / "metadata.csv"
    write_emg2pose_metadata(
        paths,
        metadata_path,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"wrote {len(paths)} files and metadata to {args.output_dir}")


if __name__ == "__main__":
    main()

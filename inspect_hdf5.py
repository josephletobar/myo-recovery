#!/usr/bin/env python3
"""Memory-safe structural inspector for large HDF5 files."""

from __future__ import annotations

import argparse
import ast
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import h5py
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it with: python3 -m pip install -r requirements.txt"
    ) from exc


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    raise AssertionError("unreachable")


def short(value: Any, limit: int) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    elif isinstance(value, np.ndarray):
        value = np.array2string(
            value, threshold=12, edgeitems=3, max_line_width=100
        )
    else:
        value = str(value)
    value = value.replace("\n", "\\n")
    if len(value) > limit:
        return value[: max(0, limit - 1)] + "…"
    return value


def shape_text(dataset: h5py.Dataset) -> str:
    if dataset.shape is None:
        return "null"
    return "scalar" if dataset.shape == () else "×".join(map(str, dataset.shape))


def dataset_line(path: str, dataset: h5py.Dataset) -> str:
    storage = dataset.id.get_storage_size()
    compression = dataset.compression or "none"
    return (
        f"{path}  shape={shape_text(dataset)}  dtype={dataset.dtype}  "
        f"stored={human_bytes(storage)}  compression={compression}"
    )


def print_attrs(obj: h5py.Group | h5py.Dataset, limit: int) -> None:
    if not obj.attrs:
        print("  attributes: none")
        return
    print("  attributes:")
    for key in sorted(obj.attrs):
        print(f"    {key}: {short(obj.attrs[key], limit)}")


def collect(file: h5py.File) -> tuple[list[str], list[tuple[str, h5py.Dataset]]]:
    groups: list[str] = []
    datasets: list[tuple[str, h5py.Dataset]] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        path = f"/{name}"
        if isinstance(obj, h5py.Group):
            groups.append(path)
        else:
            datasets.append((path, obj))

    file.visititems(visitor)
    return groups, datasets


def print_summary(
    file: h5py.File,
    file_path: Path,
    groups: list[str],
    datasets: list[tuple[str, h5py.Dataset]],
    top: int,
) -> None:
    stat = file_path.stat()
    storage = sum(dataset.id.get_storage_size() for _, dataset in datasets)
    logical = sum(
        (dataset.size or 0) * dataset.dtype.itemsize for _, dataset in datasets
    )

    print(f"File: {file_path}")
    print(f"Size: {human_bytes(stat.st_size)} ({stat.st_size:,} bytes)")
    print(f"Modified: {datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat()}")
    print(f"HDF5 library: {h5py.version.hdf5_version}")
    print(f"Objects: {len(groups):,} groups, {len(datasets):,} datasets")
    print(f"Dataset storage: {human_bytes(storage)}")
    print(f"Logical dataset data: {human_bytes(logical)}")

    root_names = list(file.keys())
    print(f"Root entries ({len(root_names)}):")
    print("  " + ", ".join(root_names))

    if top:
        print(f"Largest datasets (top {min(top, len(datasets))}):")
        ranked = sorted(
            datasets, key=lambda item: item[1].id.get_storage_size(), reverse=True
        )
        for path, dataset in ranked[:top]:
            print("  " + dataset_line(path, dataset))


def print_tree(
    file: h5py.File,
    groups: list[str],
    datasets: list[tuple[str, h5py.Dataset]],
    attrs: bool,
    attr_limit: int,
) -> None:
    objects: list[tuple[str, h5py.Group | h5py.Dataset]] = [
        (path, file[path]) for path in groups
    ] + datasets
    for path, obj in sorted(objects, key=lambda item: item[0]):
        if isinstance(obj, h5py.Group):
            print(f"[group]   {path}")
        else:
            print(f"[dataset] {dataset_line(path, obj)}")
        if attrs and obj.attrs:
            print_attrs(obj, attr_limit)


def sample_dataset(dataset: h5py.Dataset, rows: int) -> tuple[Any, Any | None]:
    if dataset.shape is None:
        return "<null dataspace>", None
    if dataset.shape == ():
        return dataset[()], None
    count = min(rows, dataset.shape[0])
    head = dataset[:count]
    tail = dataset[-count:] if dataset.shape[0] > count else None
    return head, tail


def decode_sample(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return np.char.decode(value, "utf-8", errors="replace")
    return value


def print_object(file: h5py.File, path: str, samples: int, attr_limit: int) -> None:
    normalized = "/" + path.strip("/") if path != "/" else "/"
    if normalized not in file:
        raise KeyError(f"no object exists at {normalized}")
    obj = file[normalized]
    kind = "group" if isinstance(obj, h5py.Group) else "dataset"
    print(f"Selected {kind}: {normalized}")
    if isinstance(obj, h5py.Group):
        print_attrs(obj, attr_limit)
        print(f"  children ({len(obj)}):")
        for name in obj:
            child = obj[name]
            if isinstance(child, h5py.Dataset):
                print("    " + dataset_line(child.name, child))
            else:
                print(f"    {child.name}/")
        return

    print("  " + dataset_line(normalized, obj))
    print(f"  chunks: {obj.chunks}")
    print(f"  max shape: {obj.maxshape}")
    print_attrs(obj, attr_limit)
    if samples:
        head, tail = sample_dataset(obj, samples)
        print(f"  first {min(samples, obj.shape[0] if obj.shape else 1)}:")
        print("    " + short(decode_sample(head), 2_000))
        if tail is not None:
            print(f"  last {min(samples, obj.shape[0])}:")
            print("    " + short(decode_sample(tail), 2_000))


def edge_timing(dataset: h5py.Dataset) -> tuple[float, float, float]:
    """Return start, end, and effective average rate using two scalar reads."""
    if dataset.shape is None or not dataset.shape or dataset.shape[0] == 0:
        raise ValueError(f"no timestamp rows in {dataset.name}")
    start = float(np.asarray(dataset[0]).reshape(-1)[0])
    end = float(np.asarray(dataset[-1]).reshape(-1)[0])
    rate = (dataset.shape[0] - 1) / (end - start) if end > start else float("nan")
    return start, end, rate


def iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="milliseconds")


def describe_stream(
    file: h5py.File, data_path: str, time_path: str
) -> tuple[float, float] | None:
    if data_path not in file or time_path not in file:
        print(f"    MISSING: {data_path} or {time_path}")
        return None
    data = file[data_path]
    time = file[time_path]
    start, end, rate = edge_timing(time)
    print(f"    data: {data_path}")
    print(f"    time: {time_path}")
    print(
        f"    shape={shape_text(data)}, dtype={data.dtype}, rows={data.shape[0]:,}, "
        f"effective rate={rate:.3f} Hz"
    )
    print(f"    coverage={iso_time(start)} to {iso_time(end)} ({end - start:.3f} s)")
    return start, end


def parse_headings(group: h5py.Group) -> list[str]:
    raw = group.attrs.get("Data headings", "[]")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        parsed = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def companion_videos(file_path: Path) -> list[Path]:
    prefix = file_path.name.split("_streamLog", 1)[0]
    extensions = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
    try:
        return sorted(
            path
            for path in file_path.parent.glob(f"{prefix}*")
            if path.is_file() and path.suffix.lower() in extensions
        )
    except OSError:
        return []


def print_myorecovery(file: h5py.File, file_path: Path) -> None:
    """Report the ActionSense streams needed by the MyoRecovery evaluation."""
    coverage: list[tuple[float, float]] = []

    print("MyoRecovery evaluation inputs")
    print("=============================")

    print("\n[PARTIAL] Egocentric video")
    world_data = "/eye-tracking-video-world/frame_timestamp/data"
    world_time = "/eye-tracking-video-world/frame_timestamp/time_s"
    span = describe_stream(file, world_data, world_time)
    if span:
        coverage.append(span)
    video_groups = (
        "/eye-tracking-video-world",
        "/eye-tracking-video-worldGaze",
        "/eye-tracking-video-eye",
    )
    image_datasets: list[str] = []
    for group_path in video_groups:
        if group_path not in file:
            continue
        file[group_path].visititems(
            lambda name, obj, root=group_path: image_datasets.append(f"{root}/{name}")
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 3
            else None
        )
    if image_datasets:
        print("    image-like datasets: " + ", ".join(image_datasets))
    else:
        print("    WARNING: this HDF5 stores frame timestamps, not video pixels.")
        matches = companion_videos(file_path)
        if matches:
            print("    same-prefix companion video(s):")
            for match in matches:
                print(f"      {match}")
        else:
            print("    No same-prefix companion video was found beside the HDF5 file.")

    print("\n[READY] Myo sEMG")
    for side in ("left", "right"):
        print(f"  {side.capitalize()} arm (8 channels, normalized integer range [-128, 127]):")
        span = describe_stream(
            file, f"/myo-{side}/emg/data", f"/myo-{side}/emg/time_s"
        )
        if span:
            coverage.append(span)
        original = f"/myo-{side}/emg/time_s_original"
        if original in file:
            print(f"    original device/arrival timing also available: {original}")

    print("\n[READY] Ground-truth hand pose")
    joint_group = "/xsens-joints/rotation_xzy_deg"
    joint_data = f"{joint_group}/data"
    joint_time = f"{joint_group}/time_s"
    span = describe_stream(file, joint_data, joint_time)
    if span:
        coverage.append(span)
    headings = parse_headings(file[joint_group]) if joint_group in file else []
    joint_count = len(headings) // 3
    finger_joint_count = max(0, joint_count - 22)
    print(
        f"    primary target: {joint_count} articulated joints × 3 Euler components; "
        f"{finger_joint_count} are finger joints (19 left + 19 right)"
    )
    print("    finger slices: data[:, 22:41, :] = left; data[:, 41:60, :] = right")
    print("    alternate Euler convention: /xsens-joints/rotation_zxy_deg/data")
    segment_group = "/xsens-segments/position_cm"
    if segment_group in file:
        segment_headings = parse_headings(file[segment_group])
        print(
            f"    segment positions: /xsens-segments/position_cm/data "
            f"({len(segment_headings) // 3} segments × xyz, cm)"
        )
        print("    hand slices: data[:, 23:43, :] = left; data[:, 43:63, :] = right")
        print("    matching segment Euler angles and quaternions are also available.")
    print(
        "    NOTE: there is no explicit Manus-named stream; finger kinematics are "
        "embedded in XsensStreamer output."
    )

    print("\n[READY, OPTIONAL] Tactile / force-related sensing")
    for side in ("left", "right"):
        print(f"  {side.capitalize()} glove (32×32 ADC pressure matrix, range [0, 4095]):")
        span = describe_stream(
            file,
            f"/tactile-glove-{side}/tactile_data/data",
            f"/tactile-glove-{side}/tactile_data/time_s",
        )
        if span:
            coverage.append(span)
    if "/tactile-calibration-scale" in file:
        print("    calibration metadata: /tactile-calibration-scale")

    print("\n[READY] Synchronization metadata")
    print("    shared alignment key: epoch seconds in each stream's time_s dataset")
    print("    world camera: frame_timestamp/data and frame_timestamp/time_s")
    print("    Myo: corrected time_s plus time_s_original")
    print("    Xsens: time_s, xsens_sample_number, and xsens_time_since_start_s")
    print("    tactile gloves: tactile_data/time_s")
    if coverage:
        overlap_start = max(start for start, _ in coverage)
        overlap_end = min(end for _, end in coverage)
        if overlap_end > overlap_start:
            print(
                f"    all-stream overlap={iso_time(overlap_start)} to "
                f"{iso_time(overlap_end)} ({overlap_end - overlap_start:.3f} s)"
            )
        else:
            print("    WARNING: the required streams have no common overlap.")

    print("\nOverall: numeric evaluation inputs are present; egocentric video pixels must")
    print("be supplied as a companion recording and aligned to the world-frame timestamps.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a large HDF5 file without loading entire datasets into memory. "
            "The default view summarizes metadata and the largest datasets."
        )
    )
    parser.add_argument("file", type=Path, help="HDF5 file to inspect")
    parser.add_argument(
        "--dataset", "--object", dest="object_path", metavar="PATH",
        help="show one group or dataset, including bounded edge samples",
    )
    parser.add_argument(
        "--myorecovery", action="store_true",
        help="audit the streams needed for the MyoRecovery evaluation",
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="rows to read from each edge with --dataset (default: 3)",
    )
    parser.add_argument(
        "--tree", action="store_true", help="list every group and dataset"
    )
    parser.add_argument(
        "--attrs", action="store_true", help="include attributes in --tree output"
    )
    parser.add_argument(
        "--attr-limit", type=int, default=240,
        help="maximum characters per attribute (default: 240)",
    )
    parser.add_argument(
        "--top", type=int, default=12,
        help="number of largest datasets in summary (default: 12; 0 disables)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.samples < 0 or args.top < 0 or args.attr_limit < 20:
        raise SystemExit("--samples/--top must be nonnegative; --attr-limit must be >= 20")
    if not args.file.is_file():
        raise SystemExit(f"Not a file: {args.file}")

    try:
        with h5py.File(args.file, "r") as file:
            groups, datasets = collect(file)
            if args.myorecovery:
                print_myorecovery(file, args.file)
            elif args.object_path:
                print_object(file, args.object_path, args.samples, args.attr_limit)
            else:
                print_summary(file, args.file, groups, datasets, args.top)
                if args.tree:
                    print_tree(file, groups, datasets, args.attrs, args.attr_limit)
    except (OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# myo-recovery

Small, bounded-memory tools for examining the ActionNet HDF5 recording.

## Environment

The local environment is already created. Activate it with:

```sh
source .venv/bin/activate
```

To rebuild it on a normal Python installation:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On this Mac, Homebrew Python currently needs its matching Expat library while
bootstrapping pip:

```sh
DYLD_LIBRARY_PATH="$(brew --prefix expat)/lib" python3.12 -m venv .venv
DYLD_LIBRARY_PATH="$(brew --prefix expat)/lib" .venv/bin/python -m pip install -r requirements.txt
```

The interactive RGB/EMG viewer adds Matplotlib:

```sh
.venv/bin/python -m pip install -r requirements-viz.txt
```

## Inspect a file

Summary and largest datasets:

```sh
python inspect_hdf5.py "/Volumes/Crucial X9/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5"
```

Audit the inputs needed for the MyoRecovery evaluation:

```sh
python inspect_hdf5.py FILE.hdf5 --myorecovery
```

This focused view reports egocentric-video timestamps, bilateral Myo sEMG,
finger joint angles and segment positions, tactile matrices, and the common
timestamp overlap. It also warns when video pixels or other required assets
are not embedded in the HDF5 file.

## Load activity windows in Python

The loader opens the HDF5 and returns one `SessionSequence` per activity interval.
It reads only that interval, so the full 4 GB recording is not loaded into
memory:

```python
from myo_recovery import ActionSenseLoader

path = "/Volumes/Crucial X9/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5"

with ActionSenseLoader(path) as loader:
    sequence = loader.load_activity(0)
    print(sequence.subject, sequence.session, sequence.activity)
    print(sequence.emg_left.values.shape)   # (N, 8)
    print(sequence.hand_pose.values.shape)  # (N, 60, 3)

    aligned = sequence.align(target_hz=30.0)
    print(aligned.emg_left.timestamps.shape) # (T,)
    print(aligned.combined_emg().shape)      # (T, 16), left + right
    print(aligned.hand_pose.values.shape)    # (T, 60, 3)
    print(aligned.combined_tactile().shape)  # (T, 2, 32, 32), if enabled

    for sequence in loader.iter_activities():
        # Train or export one activity window at a time.
        pass
```

To attach a separately downloaded ActionSense world-camera MP4, pass it when
constructing the loader. The sequence keeps the video path and frame indices
for the selected activity; it does not load the video pixels into memory:

```python
# Use the MP4 from the same ActionSense session as ``path``.
video = "/Volumes/Crucial X9/MATCHING_SESSION_eye-tracking-video-world_frame_repaired.mp4"
with ActionSenseLoader(path, video_path=video) as loader:
    sequence = loader.load_activity(0)
    print(sequence.rgb_video_path)
    print(sequence.rgb_frame_indices.shape)  # one source index per HDF5 timestamp
```

## Inspect RGB and EMG together

Use the standalone viewer with an HDF5 recording and the matching world-camera
MP4. It shows the current RGB frame above eight EMG traces, with a shared time
cursor, frame slider, and play/pause control. Only the selected MP4 frame is
decoded, so a multi-gigabyte recording is not loaded into memory:

```sh
python visualize_actionsense.py \
  "/path/to/matching_session.hdf5" \
  --video "/path/to/matching_world_video.mp4" \
  --activity 0 \
  --side left
```

`--activity` accepts either an activity index or its exact label. EMG is
per-channel z-scored for readability by default; pass `--raw-emg` to plot the
source units. The HDF5 and MP4 must belong to the same ActionSense session for
the frame references to be synchronized.

Raw and aligned values use the same `SessionSequence` structure. Each `Stream`
retains its own timestamps before alignment; after `sequence.align(30.0)`, all
available streams have the same regular timestamp vector, and RGB references
are nearest-neighbor selected on that clock. The loader does not
silently resample data. Alignment is explicit: pose is linearly interpolated,
tactile uses nearest samples, and EMG defaults to a per-window RMS feature
(`emg_mode="linear"` is available when linearly interpolated EMG is specifically
desired).

For an EMG2Pose-style target, select one hand and map the Xsens headings into
the 20 scalar joint angles used by Meta's architecture:

```python
from myo_recovery import make_emg2pose_sequence

emg2pose = make_emg2pose_sequence(sequence, side="left")
print(emg2pose.emg.values.shape)            # (N, 8), native Myo samples
print(emg2pose.target.joint_angles.shape)   # (T, 20), radians
```

That mapper is the native-rate representation adapter: it is useful for
inspection, but its EMG and pose clocks are still different. For training at
30 Hz, use the explicit preparation step below. It computes one EMG feature
per 30 Hz interval, resamples the Xsens angles onto the same clock, and writes
one hand/activity in the compound HDF5 layout expected by Meta's loader:

```sh
python prepare_emg2pose.py \
  "/Volumes/Crucial X9/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5" \
  ./prepared_emg2pose \
  --side both --target-hz 30 --emg-mode rms
```

Each output contains `emg` with shape `(T, 8)`, `joint_angles` with shape
`(T, 20)` in radians, and `time` with shape `(T,)`. The 20 columns use the
same ordering as Meta's EMG2Pose target. The default `--emg-scale 128`
normalizes the ActionSense Myo features; pass `--emg-scale 0` to preserve
their source units. The CLI also writes `metadata.csv`, grouped by activity
so paired left/right files stay in the same train/validation/test split.

The export is deliberately explicit: it produces a 30 Hz RMS envelope and
does not ask the model to downsample it again. Use
`config/emg2pose_actionsense_tds.yaml` in the Meta repository as the network
override. It changes the input width to 8 channels and sets every temporal
stride to 1. Also select Meta's `basic` transform rather than the optional
`channel_downsampling` transform, because ActionSense already has the desired
eight channels per hand. Use
`config/emg2pose_actionsense_datamodule.yaml` as the matching datamodule
override: Meta's default window counts are expressed in 2,000-Hz samples, so
they are too large for a 30-Hz file and can produce no windows for a short
activity. The supplied override uses 120-sample training windows and
600-sample validation/test windows. Finally, use
`config/emg2pose_actionsense_pose_stateful.yaml` for the stateful decoder.
The Meta pose module is patched on the Linux checkout's `main` branch
(commit `af618b3`) so its decoder accepts
`input_sample_rate: 30` and `rollout_freq: 30`. Its derivative metrics use the
same explicit rate; no 2,000-Hz compatibility setting or hidden resampling is
needed.

The resulting Meta-repository selection is:

```sh
python -m emg2pose.train train=True eval=True \
  experiment=regression_actionsense \
  transforms=basic
```

## Render a qualitative reconstruction

For a frame-synchronous qualitative check, run
`render_actionsense_reconstruction.py` in the Linux EMG2Pose environment. It
keeps the heavy checkpoint on the GPU, reconstructs the 20 joint angles, and
uses Meta's UmeTrack skinning model to write a 30 Hz MP4 with three panels:
ActionSense RGB, the predicted hand mesh, and the prediction overlaid on the
ground-truth mesh. The input video can be a short clip for one activity rather
than the full multi-gigabyte recording.

```sh
python render_actionsense_reconstruction.py \
  --prepared-hdf5 /data/datasets/actionsense/prepared_s04/S04_2022-06-14_16-38-43_activity-019_left.hdf5 \
  --video /data/datasets/actionsense/actionsense_activity019_source.mp4 \
  --checkpoint /data/datasets/actionsense/full_train_s04_20260823/lightning_logs/version_0/checkpoints/epoch=121-step=1464.ckpt \
  --config /data/datasets/actionsense/full_train_s04_20260823/hydra_configs/config.yaml \
  --emg2pose-root /home/joseph/projects/emg2pose \
  --output /data/datasets/actionsense/actionsense_activity019_reconstruction.mp4 \
  --side left --fps 30
```

The ActionSense model has zero right context, so this playback is causal: the
prediction at each displayed time only uses EMG up to that time. Inference is
batched to make export fast, but the resulting reconstruction is rendered at
the model's 30 Hz clock.

List the complete object tree:

```sh
python inspect_hdf5.py FILE.hdf5 --tree
```

Inspect one dataset and read only three rows from each edge:

```sh
python inspect_hdf5.py FILE.hdf5 \
  --dataset /experiment-activities/activities/data \
  --samples 3
```

Use `--tree --attrs` to include truncated attributes. Run `--help` for all
options. The inspector never reads a whole dataset unless that dataset itself
has at most the requested number of sample rows.

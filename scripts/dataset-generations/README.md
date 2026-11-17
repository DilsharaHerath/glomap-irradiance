# Facade datasets for GLOMAP

Standalone camera planning and Mitsuba rendering based on
`solar-drone/drone-prep-new/facade-camera-grid_4ODGS_withpanels-completed.ipynb`.
No imports from that project or notebook execution are required. Only the city
scene assets remain external. Source assets and existing ODGS datasets are read
only. `configs/reference.json` records reference file hashes.

The included targets use the same catalogue, triangle-plane extraction, side
conventions and facade IDs as the reference:

| Target | Building | Facade | Origin |
| --- | --- | --- | --- |
| `north_28` (default) | `SM_Building_17_01_LOD0_28` | `facade_001`, north | Active notebook selection |
| `east` | `SM_Building_17_01_LOD0` | `facade_003`, east | Earlier notebook sequences in `made_seqs.json` |

## How the images differ from ODGS

The reference uses outward-facing, full-sphere panoramas and closely spaced
layers 0.2–4 m from the wall. This generator uses **perspective cameras facing
the selected facade**, at 8 m and 12 m by default. It estimates poses from
images later; known Mitsuba poses are saved separately for evaluation.

Default settings are 1280×960, 70° horizontal field of view, 256 SPP in four
64-SPP passes, original textured materials, path tracing with 32 maximum
bounces, and the reference sun/sky settings. The reference's median firefly
rejection is retained. There is no denoiser or synthetic sensor noise. All final
PNGs use one exposure computed from pooled HDR samples, followed by sRGB transfer.

Metric camera spacing is chosen from the perspective footprint for **75% nominal
overlap on the reference plane**. Gaps are bounded above, rather than below as
in the ODGS layout. Borders extend 2 m left/right/top. Odd rows are staggered;
border cameras toe in toward the nearest point inside the facade. Actual
overlap depends on scene relief, occluders and those border rotations. Before
rendering, candidates are rejected when inside geometry, within 0.25 m of a
surface, or with less than 15% sampled facade coverage. These are heuristic
checks, not a guarantee of successful SfM. The two targets are separate datasets.

The full default plans have **181 north candidates and 122 east candidates**
with the supplied meshes. Occlusion checks may reduce those counts.

## Environment

Use the existing **`env_1`** environment for generation: it already contains
Mitsuba 3.9.1, NumPy and Pillow. Keep the C++ reconstruction environment separate.

Alternatively, create a dedicated rendering environment:

```bash
conda env create -f scripts/dataset-generations/environment.yml
conda activate glomap-render
```

For an existing compatible Python environment, the package list is in
`requirements.txt`. Rendering needs a working NVIDIA driver for `cuda_ad_rgb`,
or LLVM for `llvm_ad_rgb`. Planning only needs NumPy. No Qt or GUI is required.

## Plan the camera positions

Run these examples from the cloned GLOMAP repository:

```bash
cd /local/compvis/dilshara/SVM/glomap
conda activate env_1

# Read assets and print a summary; do not create files or render.
python scripts/dataset-generations/generate_dataset.py plan --target all --dry-run

# Save a plan for the notebook's current facade.
python scripts/dataset-generations/generate_dataset.py plan --target north_28
```

Plans go to `data/facades/<target>/metadata/dataset.json`. To create both plans,
use `--target all`; use `--target east` for the previous building only.

The scene path is configured in `configs/facades.json`. It currently points to:

```text
/local/compvis/dilshara/Datasets/new-mitsuba-scenes/city-340m/city-340m.xml
```

Use `--scene-xml /your/path/city-340m.xml` if the assets move. For different
resolution, offsets, overlap, lighting, depth output or render quality, copy
`configs/facades.json`, edit the copy, set its `catalogue` to the supplied
`buildings.json` (relative to your config or absolute), and pass `--config`.
`min_height_above_building_m` is optional, relative to `bbox_min[1]`.
The defaults do not apply a height cutoff, matching the active notebook.

## Render, inspect and resume

```bash
python scripts/dataset-generations/generate_dataset.py render \
    --dataset data/facades/north_28 --gpu 0
```

For CPU rendering, replace `--gpu 0` with `--variant llvm_ad_rgb`. The script
uses one GPU per process. Separate datasets can be rendered on different GPUs,
but simultaneous renderers may not write into the same dataset.

To inspect a small pilot first, give it a separate output root:

```bash
python scripts/dataset-generations/generate_dataset.py plan \
    --target north_28 --limit-views 12 --output-root data/facade-pilot
python scripts/dataset-generations/generate_dataset.py render \
    --dataset data/facade-pilot/north_28 --gpu 0
```

`--limit-views` evenly samples across the complete camera list; it can reduce
overlap and is for checking renders, not assessing full reconstruction quality.
For a partial full-quality run, use `render --max-new-views 3`. It saves HDRs
and fixed-exposure inspection PNGs in `previews/`, then stops. Repeat the render
command without that limit to finish. Inspection PNGs never enter reconstruction.

Completed HDR/depth files are checked by hash and reused. Final shared-exposure
PNGs are written only when every candidate has been processed. Interrupted
outputs are republished atomically. Different plan settings or changed source
assets require a new output root; the generator does not overwrite an unrelated
dataset. Resume using the same Mitsuba version and CPU/GPU backend to avoid
mixing render implementations. Review rejection reasons in `metadata/status.json` and adjust camera
distances if nearby buildings block the facade.

```text
data/facades/north_28/
├── metadata/
│   ├── dataset.json          # Immutable plan, calibration, poses, provenance
│   ├── status.json           # Accepted/rejected views, reasons, file hashes
│   └── exposure.json         # One shared final scale
├── hdr/                      # Linear RGB EXR renders, retained for resume
├── previews/                 # Provisional PNGs; never used by COLMAP
├── images/                   # Final calibrated RGB PNGs for COLMAP
├── depth/                    # Optional float32 .npy, camera-Z metres, sky=0
└── ground_truth/
    ├── poses.json            # Mitsuba c2w and COLMAP w2c, accepted views only
    └── colmap/
        ├── cameras.txt
        ├── images.txt        # Poses only; no observed 2D tracks
        └── points3D.txt      # Empty; no fabricated reference reconstruction
```

World units are metres and world +Y is up. Mitsuba camera coordinates are
left/up/forward; COLMAP coordinates are right/down/forward. The conversion
flips camera X/Y, inverts c2w and exports Hamilton quaternions `(qw,qx,qy,qz)`.
Pixel coordinates use the same corner-based convention as COLMAP; the principal
point is `(width/2,height/2)`. Intrinsics are shared, distortion-free `PINHOLE`.
See [Mitsuba cameras](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_sensors.html)
and [COLMAP model format](https://colmap.github.io/format.html).

## Reconstruct the completed dataset

Use the **`glomap`** environment containing the executables we built:

```bash
conda activate glomap
python scripts/dataset-generations/reconstruct.py \
    --dataset data/facades/north_28 --dry-run

python scripts/dataset-generations/reconstruct.py \
    --dataset data/facades/north_28
```

The runner checks completeness, image dimensions and hashes, then calls
`colmap feature_extractor`, `colmap exhaustive_matcher` and `glomap mapper`.
It supplies known intrinsics and keeps them fixed. It never supplies ground-truth
camera extrinsics or geometry to the mapper. Flags match the COLMAP 3.14 dev
revision pinned by this GLOMAP checkout. Use `--cpu` to disable GPU processing.
The estimated reconstruction has arbitrary similarity scale; align it to the
saved ground-truth poses before interpreting its coordinates as metres.

Outputs go to `DATASET/reconstruction/database.db`, `sparse/0`, and `logs/`.
For another run, supply `--output /path/to/new/run`; existing nonempty outputs
are protected. GLOMAP returns a sparse reconstruction, not a finished facade
mesh. Dense stereo/meshing is a separate later step.

## Validation

```bash
conda activate env_1
python -B -m unittest discover -s scripts/dataset-generations/tests -v
```

Tests cover rotation/quaternion conversion (including 180° rotations), camera
centers and projection, overlap spacing, manifest changes, XML include handling,
native Mitsuba pixel rays versus exported calibration, camera-Z depth,
occlusion rejection, partial-render recovery, shared-exposure output integrity,
and keeping ground-truth poses out of reconstruction inputs. Rendering tests
use a tiny temporary synthetic scene on LLVM; no city rendering is triggered.

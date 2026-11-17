#!/usr/bin/env python3
"""Run COLMAP features/matching and GLOMAP on a completed generated dataset.

Uses only the Python standard library, so run this in the existing glomap
environment. Ground-truth extrinsics are never passed to reconstruction.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys


def checked_dataset(dataset):
    plan = json.loads((dataset / "metadata/dataset.json").read_text())
    specification = {k: v for k, v in plan.items() if k != "fingerprint"}
    fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if fingerprint != plan["fingerprint"]:
        raise ValueError("Dataset plan was modified after generation")
    state = json.loads((dataset / "metadata/status.json").read_text())
    if state.get("status") != "complete" or state.get("fingerprint") != fingerprint:
        raise ValueError("Render the entire dataset before reconstruction; it is incomplete or has mismatched settings")
    accepted = [v for v in plan["views"] if state["views"].get(v["name"], {}).get("status") == "rendered"]
    if len(accepted) < 3:
        raise ValueError("At least three accepted views are required")
    expected = {v["name"] for v in accepted}
    actual = {p.name for p in (dataset / "images").iterdir() if p.is_file()}
    if actual != expected:
        raise ValueError("images/ differs from the accepted view list; do not mix datasets or include previews")
    for name in sorted(expected):
        path = dataset / "images" / name
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != state["views"][name].get("png_sha256"):
            raise ValueError(f"Image changed after finalization: {name}")
        if data[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", data[16:24]) != (plan["calibration"]["width"], plan["calibration"]["height"]):
            raise ValueError(f"Image dimensions do not match calibration: {name}")
    return plan


def commands_for(plan, dataset, output, colmap, glomap, cpu=False, gpu_index="0", threads=16):
    calibration = plan["calibration"]
    gpu = "0" if cpu else "1"
    params = ",".join(format(v, ".17g") for v in calibration["params"])
    database, sparse = str(output / "database.db"), str(output / "sparse")
    return [
        [colmap, "feature_extractor", "--database_path", database, "--image_path", str(dataset / "images"),
         "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "1",
         "--ImageReader.camera_params", params, "--FeatureExtraction.use_gpu", gpu,
         "--FeatureExtraction.gpu_index", gpu_index, "--FeatureExtraction.num_threads", str(threads),
         "--SiftExtraction.max_image_size", str(max(calibration["width"], calibration["height"]))],
        [colmap, "exhaustive_matcher", "--database_path", database,
         "--FeatureMatching.use_gpu", gpu, "--FeatureMatching.gpu_index", gpu_index,
         "--FeatureMatching.num_threads", str(threads), "--TwoViewGeometry.compute_relative_pose", "1"],
        [glomap, "mapper", "--database_path", database, "--image_path", str(dataset / "images"),
         "--output_path", sparse, "--skip_view_graph_calibration", "1",
         "--BundleAdjustment.optimize_intrinsics", "0",
         "--GlobalPositioning.use_gpu", gpu, "--BundleAdjustment.use_gpu", gpu,
         "--GlobalPositioning.gpu_index", gpu_index, "--BundleAdjustment.gpu_index", gpu_index],
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New reconstruction directory; default: DATASET/reconstruction")
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable built alongside this GLOMAP")
    parser.add_argument("--glomap", default="glomap")
    parser.add_argument("--cpu", action="store_true", help="Disable GPU extraction, matching and optimization")
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without running them")
    args = parser.parse_args(argv)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    dataset = args.dataset.resolve()
    plan = checked_dataset(dataset)
    output = (args.output or dataset / "reconstruction").resolve()
    if output == dataset or dataset.is_relative_to(output):
        raise ValueError("Reconstruction output must not replace or contain the source dataset")
    for name in ("images", "metadata", "ground_truth", "hdr", "depth", "previews"):
        if output.is_relative_to(dataset / name):
            raise ValueError(f"Reconstruction output must be outside the dataset's {name}/ directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Reconstruction output is nonempty: {output}; use --output with a new directory")
    commands = commands_for(plan, dataset, output, args.colmap, args.glomap, args.cpu, args.gpu_index, args.threads)
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    for executable in (args.colmap, args.glomap):
        if shutil.which(executable) is None:
            raise ValueError(f"Executable not found: {executable}; activate the glomap environment")
    (output / "logs").mkdir(parents=True)
    (output / "sparse").mkdir()
    record = dict(dataset_fingerprint=plan["fingerprint"], commands=commands, status="running")
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    for index, command in enumerate(commands, 1):
        log = output / "logs" / f"{index:02d}_{command[1]}.log"
        print(f"Running {command[1]}; log: {log}", flush=True)
        with log.open("w") as stream:
            try:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError:
                record.update(status="failed", failed_step=command[1], log=str(log))
                (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
                raise
    models = sorted(p.parent for p in (output / "sparse").glob("*/cameras.bin"))
    if not models:
        raise RuntimeError("Mapper produced no sparse model; inspect the mapper log and image overlap")
    record.update(status="complete", models=[str(p) for p in models])
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Reconstruction complete: {models}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

"""Mitsuba rendering, scene clearance checks and shared-exposure PNG export.

Lighting, material adapters and median firefly filtering follow drone-prep-new.
All modifications to source scene XML happen in memory.
"""

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

from .planning import atomic_json, digest, expanded_scene, file_hash, integer, read_plan


def init_backend(variant, gpu, threads):
    integer(threads, "threads")
    if gpu is not None:
        if variant == "llvm_ad_rgb":
            raise ValueError("--gpu cannot be combined with the CPU variant")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import mitsuba as mi
    import drjit as dr
    if variant == "auto":
        for choice in ("cuda_ad_rgb", "llvm_ad_rgb"):
            try:
                mi.set_variant(choice)
                break
            except Exception:
                if gpu is not None:
                    raise
        else:
            raise RuntimeError("No CUDA or LLVM Mitsuba backend is available")
    else:
        mi.set_variant(variant)
    dr.set_thread_count(threads)
    return mi


def load_scene(mi, xml_root, settings):
    """Retain the city geometry/materials; replace sensor, integrator and environment."""
    import copy
    root = copy.deepcopy(xml_root)
    for tag in ("sensor", "integrator"):
        for item in list(root.findall(tag)):
            root.remove(item)
    for emitter in list(root.findall("emitter")):
        if emitter.get("type") in ("envmap", "sunsky", "constant"):
            root.remove(emitter)
    integrator = ET.SubElement(root, "integrator", {"type": "path" if settings["mode"] == "path" else "direct"})
    if settings["mode"] == "path":
        ET.SubElement(integrator, "integer", {"name": "max_depth", "value": str(settings["max_depth"])})
    else:
        # Same conversion as reference render_city.configure_render_mode.
        adapters = {"twosided", "mask", "normalmap", "bumpmap", "blendbsdf", "diffuse", "null"}
        for bsdf in list(root.iter("bsdf")):
            if bsdf.get("type") in adapters:
                continue
            color = next((child for name in ("base_color", "reflectance", "diffuse_reflectance")
                          for child in bsdf if child.get("name") == name), None)
            color = copy.deepcopy(color) if color is not None else ET.Element("rgb", {"value": "0.5"})
            color.set("name", "reflectance")
            attrs = {k: v for k, v in bsdf.attrib.items() if k in ("id", "name")}
            bsdf.clear()
            bsdf.attrib.update(attrs, type="diffuse")
            bsdf.append(color)
    elevation, azimuth = np.deg2rad([settings["sun_elevation"], settings["sun_azimuth"]])
    direction = [-np.cos(elevation)*np.sin(azimuth), np.sin(elevation), np.cos(elevation)*np.cos(azimuth)]
    sky = ET.SubElement(root, "emitter", {"type": "sunsky", "id": "sfm_sky"})
    ET.SubElement(sky, "vector", {"name": "sun_direction", "value": ",".join(map(str, direction))})
    ET.SubElement(sky, "float", {"name": "turbidity", "value": str(settings["turbidity"])})
    transform = ET.SubElement(sky, "transform", {"name": "to_world"})
    ET.SubElement(transform, "rotate", {"x": "1", "angle": "-90"})
    return mi.load_string(ET.tostring(root, encoding="unicode"))


def sensor_spec(mi, view, calibration, spp):
    return dict(type="perspective", fov_axis="x", to_world=mi.ScalarTransform4f(view["cam_to_world"]),
                fov=math.degrees(2*math.atan(calibration["width"] / (2*calibration["params"][0]))),
                near_clip=0.01, far_clip=10000.0,
                film=dict(type="hdrfilm", width=calibration["width"], height=calibration["height"],
                          pixel_format="rgb", component_format="float32", rfilter=dict(type="box")),
                sampler=dict(type="multijitter", sample_count=spp))


def cast(mi, scene, origins, directions):
    ray = mi.Ray3f()
    ray.o = mi.Point3f(*[mi.Float(np.ascontiguousarray(origins[:, i], dtype=np.float32)) for i in range(3)])
    ray.d = mi.Vector3f(*[mi.Float(np.ascontiguousarray(directions[:, i], dtype=np.float32)) for i in range(3)])
    interaction = scene.ray_intersect(ray)
    valid = np.asarray(interaction.is_valid()).ravel().astype(bool)
    distances = np.where(valid, np.asarray(interaction.t).ravel(), np.inf)
    points = np.asarray(interaction.p).reshape(3, -1).T
    normals = np.asarray(interaction.n).reshape(3, -1).T
    return distances, valid, points, normals


def check_view(mi, scene, view, plan):
    """Reject inside/near-surface cameras and views with too little facade coverage."""
    checks, plane = plan["checks"], plan["plane"]
    c2w = np.asarray(view["cam_to_world"])
    origin = c2w[:3, 3]
    n = checks["probe_rays"]
    i = np.arange(n) + 0.5
    y = 1 - 2*i/n
    radius = np.sqrt(1-y*y)
    phi = np.pi*(3-np.sqrt(5))*i
    directions = np.column_stack((radius*np.cos(phi), y, radius*np.sin(phi)))
    distances, valid, _, normals = cast(mi, scene, np.tile(origin, (n, 1)), directions)
    nearest = float(distances.min())
    back = float(((normals[valid] * directions[valid]).sum(1) > 0).mean()) if valid.any() else 0.0
    metrics = dict(nearest_surface_m=nearest if math.isfinite(nearest) else None, backface_fraction=back)
    if back > 0.5:
        return "inside_geometry", metrics
    if nearest < checks["clearance_m"]:
        return "insufficient_clearance", metrics
    width, height = checks["visibility_grid"]
    # Pixel centers across the entire field of view, in Mitsuba camera axes.
    u, v = np.meshgrid((np.arange(width)+0.5)/width, (np.arange(height)+0.5)/height)
    fx, fy, cx, cy = plan["calibration"]["params"]
    w, h = plan["calibration"]["width"], plan["calibration"]["height"]
    local = np.column_stack(((cx-u.ravel()*w)/fx, (cy-v.ravel()*h)/fy, np.ones(u.size)))
    directions = local @ c2w[:3, :3].T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    _, hit, points, _ = cast(mi, scene, np.tile(origin, (len(local), 1)), directions)
    along, depth = points @ np.asarray(plane["right"]), points @ np.asarray(plane["normal"]) - plane["distance"]
    facade = (hit & (along >= plane["u_min"]) & (along <= plane["u_max"])
              & (points[:, 1] >= plane["y_min"]) & (points[:, 1] <= plane["y_max"])
              & (np.abs(depth) <= checks["facade_depth_tolerance_m"]))
    metrics["facade_fraction"] = float(facade.mean())
    return (None if metrics["facade_fraction"] >= checks["min_facade_fraction"] else "facade_occluded"), metrics


def combine_passes(stack, factor):
    """Reference median firefly rule, with row chunks to bound temporary memory."""
    result = np.zeros(stack.shape[1:], np.float32)
    touched = 0
    for start in range(0, result.shape[0], 32):
        stop = start + 32
        block = stack[:, start:stop]
        luminance = block @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        keep = luminance <= factor*np.median(luminance, axis=0) + 0.005
        total = result[start:stop]
        for sample, mask in zip(block, keep):
            total += sample * mask[..., None]
        total /= np.maximum(keep.sum(0), 1)[..., None]
        touched += int((~keep).any(0).sum())
    return result, touched / (result.shape[0]*result.shape[1])


def depth_map(mi, scene, sensor, c2w, width, height):
    """Pixel-center camera-Z depth in metres, zero for no hit. Not radial distance."""
    depth = np.zeros((height, width), dtype=np.float32)
    c2w = np.asarray(c2w)
    for start in range(0, height, 32):
        stop = min(start+32, height)
        x, y = np.meshgrid((np.arange(width)+0.5)/width, (np.arange(start, stop)+0.5)/height)
        ray, _ = sensor.sample_ray(0, 0, mi.Point2f(mi.Float(x.ravel()), mi.Float(y.ravel())), mi.Point2f(0))
        si = scene.ray_intersect(ray)
        points = np.asarray(si.p).reshape(3, -1).T
        valid = np.asarray(si.is_valid()).ravel().astype(bool)
        values = np.zeros(len(points), np.float32)
        values[valid] = (points[valid] - c2w[:3, 3]) @ c2w[:3, 2]
        depth[start:stop] = values.reshape(stop-start, width)
    return depth


def save_png(path, hdr, scale):
    from PIL import Image
    linear = np.clip(hdr*scale, 0, 1)
    srgb = np.where(linear <= 0.0031308, 12.92*linear, 1.055*linear**(1/2.4)-0.055)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(np.rint(srgb*255).astype(np.uint8)).save(temporary)
    temporary.replace(path)


def export_ground_truth(output, plan, accepted):
    destination = output / "ground_truth/colmap"
    destination.mkdir(parents=True, exist_ok=True)
    calibration = plan["calibration"]
    params = " ".join(format(v, ".17g") for v in calibration["params"])
    (destination / "cameras.txt").write_text(
        "# Ground-truth calibration; not estimated by GLOMAP.\n"
        f"1 PINHOLE {calibration['width']} {calibration['height']} {params}\n")
    lines = ["# Ground-truth world-to-camera poses; empty 2D observation lines."]
    for view in accepted:
        pose = view["colmap"]
        values = " ".join(format(x, ".17g") for x in pose["qvec"] + pose["tvec"])
        lines.extend([f"{view['id']} {values} 1 {view['name']}", ""])
    (destination / "images.txt").write_text("\n".join(lines) + "\n")
    (destination / "points3D.txt").write_text("# No ground-truth tracks are fabricated.\n")
    atomic_json(output / "ground_truth/poses.json", dict(
        units="metres", mitsuba_axes="left, up, forward", colmap_axes="right, down, forward",
        depth="camera-Z in metres, 0 = no hit" if plan["render"]["save_depth"] else None,
        calibration=calibration, views=accepted))


def finalize(mi, output, plan, state):
    accepted = [v for v in plan["views"] if state["views"].get(v["name"], {}).get("status") == "rendered"]
    if not accepted:
        raise ValueError("All cameras were rejected. Inspect status.json and adjust offsets or checks in a new plan.")
    # Exact percentile from pooled [::4,::4] RGB samples, backed by disk.
    count = 0
    with tempfile.TemporaryFile(dir=output) as pool:
        for view in accepted:
            hdr = np.asarray(mi.Bitmap(str(output / "hdr" / Path(view["name"]).with_suffix(".exr"))))
            samples = np.ascontiguousarray(hdr[::4, ::4, :3], dtype=np.float32).ravel()
            samples.tofile(pool)
            count += samples.size
        pool.flush()
        values = np.memmap(pool, dtype=np.float32, mode="r+", shape=(count,))
        white = max(float(np.percentile(values, plan["render"]["exposure_percentile"], overwrite_input=True)), 1e-6)
        del values
    (output / "images").mkdir(exist_ok=True)
    for view in accepted:
        hdr = np.asarray(mi.Bitmap(str(output / "hdr" / Path(view["name"]).with_suffix(".exr"))))
        save_png(output / "images" / view["name"], hdr[..., :3], 1/white)
        state["views"][view["name"]]["png_sha256"] = file_hash(output / "images" / view["name"])
    atomic_json(output / "metadata/exposure.json", dict(
        scale=1/white, white=white, percentile=plan["render"]["exposure_percentile"],
        pooled_over=[v["name"] for v in accepted], transfer="linear RGB -> sRGB"))
    export_ground_truth(output, plan, accepted)
    state.update(status="complete", accepted=len(accepted), rejected=len(plan["views"])-len(accepted))


@contextmanager
def dataset_lock(output):
    with (Path(output) / ".render.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another renderer is already using this dataset") from error
        yield


def render_dataset(output, *, variant="auto", gpu=None, threads=8, max_new_views=None):
    output = Path(output).resolve()
    plan = read_plan(output)
    if max_new_views is not None:
        integer(max_new_views, "max_new_views")
    xml, dependencies = expanded_scene(plan["scene_xml"])
    if digest(dependencies) != plan["scene_signature"]:
        raise ValueError("Scene assets changed since planning; create a new dataset plan")
    scene_dir = Path(plan["scene_xml"]).parent
    if any(file_hash(scene_dir / name) != value for name, value in plan["target_mesh_sha256"].items()):
        raise ValueError("Target facade geometry changed since planning; create a new dataset plan")
    mi = init_backend(variant, gpu, threads)
    with dataset_lock(output):
        state_path = output / "metadata/status.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(fingerprint=plan["fingerprint"], views={})
        if state["fingerprint"] != plan["fingerprint"]:
            raise ValueError("Render state belongs to another dataset plan")
        if any(v.get("status") == "rendered" for v in state["views"].values()):
            if state.get("variant") != mi.variant() or state.get("mitsuba_version") != mi.__version__:
                raise ValueError("Resume with the same Mitsuba version/backend, or use a new dataset directory")
        state.update(status="rendering", mitsuba_version=mi.__version__, variant=mi.variant())
        atomic_json(state_path, state)
        for name in ("hdr", "previews", "depth"):
            if name != "depth" or plan["render"]["save_depth"]:
                (output / name).mkdir(exist_ok=True)
        print(f"Loading scene using {mi.variant()} ...", flush=True)
        scene = load_scene(mi, xml, plan["render"])
        width, height = plan["calibration"]["width"], plan["calibration"]["height"]
        settings = plan["render"]
        # Normalize missing/corrupt completed files before enforcing a partial-run
        # limit, so stale 'rendered' records cannot prematurely finalize a dataset.
        for view in plan["views"]:
            record = state["views"].get(view["name"], {})
            if record.get("status") != "rendered":
                continue
            files = [(output / "hdr" / Path(view["name"]).with_suffix(".exr"), "hdr_sha256")]
            if settings["save_depth"]:
                files.append((output / "depth" / Path(view["name"]).with_suffix(".npy"), "depth_sha256"))
            if any(not p.exists() or file_hash(p) != record.get(key) for p, key in files):
                record["status"] = "pending"
        stack = np.empty((settings["passes"], height, width, 3), np.float32)
        new_views = 0
        for index, view in enumerate(plan["views"], 1):
            record = state["views"].get(view["name"], {})
            if record.get("status") == "rejected":
                continue
            hdr_path = output / "hdr" / Path(view["name"]).with_suffix(".exr")
            depth_path = output / "depth" / Path(view["name"]).with_suffix(".npy")
            if record.get("status") == "rendered":
                continue
            if max_new_views is not None and new_views >= max_new_views:
                break
            reason, metrics = check_view(mi, scene, view, plan)
            record = dict(metrics=metrics, status="rejected" if reason else "rendering")
            state["views"][view["name"]] = record
            if reason:
                record["reason"] = reason
                atomic_json(state_path, state)
                print(f"[{index}/{len(plan['views'])}] REJECTED {view['name']}: {reason}", flush=True)
                continue
            started = time.perf_counter()
            sensor = mi.load_dict(sensor_spec(mi, view, plan["calibration"], settings["spp"]//settings["passes"]))
            for sample in range(settings["passes"]):
                seed = (settings["seed"] + 104729*view["id"] + sample) % (2**32)
                frame = np.asarray(mi.render(scene, sensor=sensor, seed=seed))
                stack[sample] = np.nan_to_num(frame[..., :3], nan=0, posinf=0, neginf=0)
            image, fireflies = combine_passes(stack, settings["firefly_factor"])
            np.maximum(image, 0, out=image)
            temporary = hdr_path.with_suffix(".tmp.exr")
            mi.Bitmap(image).write(str(temporary))
            temporary.replace(hdr_path)
            if settings["save_depth"]:
                depth = depth_map(mi, scene, sensor, view["cam_to_world"], width, height)
                temporary = depth_path.with_suffix(".tmp.npy")
                np.save(temporary, depth)
                temporary.replace(depth_path)
                record["depth_sha256"] = file_hash(depth_path)
            # Immediate previews have a fixed scale and never enter feature extraction.
            save_png(output / "previews" / view["name"], image, 1.0)
            record.update(status="rendered", hdr_sha256=file_hash(hdr_path),
                          seconds=round(time.perf_counter()-started, 3), firefly_fraction=fireflies)
            new_views += 1
            atomic_json(state_path, state)
            print(f"[{index}/{len(plan['views'])}] {view['name']}: {record['seconds']:.2f}s", flush=True)
        finished = all(state["views"].get(v["name"], {}).get("status") in ("rendered", "rejected") for v in plan["views"])
        if finished:
            try:
                finalize(mi, output, plan, state)
            except Exception:
                state["status"] = "failed"
                atomic_json(state_path, state)
                raise
        else:
            state["status"] = "partial"
        atomic_json(state_path, state)
        print(f"Dataset {state['status']}: {output}", flush=True)
        return state

"""Renderer-independent configuration, asset provenance and dataset plans."""

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from . import __version__
from .cameras import intrinsics, plan_views
from .geometry import facade_planes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/facades.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_config(config):
    c, r, checks = config["camera"], config["render"], config["checks"]
    for key in ("width", "height"):
        integer(c[key], key)
    positive(c["fov_x_deg"], "fov_x_deg")
    if c["fov_x_deg"] >= 150:
        raise ValueError("fov_x_deg must be below 150 degrees")
    if isinstance(c["overlap"], bool) or not 0 < c["overlap"] < 1:
        raise ValueError("overlap must be between 0 and 1")
    positive(c["border_m"], "border_m", zero=True)
    if not c["offsets_m"] or len(set(c["offsets_m"])) != len(c["offsets_m"]):
        raise ValueError("offsets_m must contain distinct positive distances")
    for offset in c["offsets_m"]:
        positive(offset, "offset_m")
    if c["min_height_above_building_m"] is not None:
        positive(c["min_height_above_building_m"], "min_height_above_building_m", zero=True)
    if not isinstance(c["stagger"], bool) or not isinstance(r["save_depth"], bool):
        raise ValueError("stagger and save_depth must be booleans")
    for key in ("spp", "passes", "max_depth"):
        integer(r[key], key)
    integer(r["seed"], "seed", minimum=0)
    if r["spp"] % r["passes"] or math.isqrt(r["spp"] // r["passes"]) ** 2 != r["spp"] // r["passes"]:
        raise ValueError("spp / passes must be a square number for the multijitter sampler")
    positive(r["firefly_factor"], "firefly_factor")
    if r["firefly_factor"] < 1:
        raise ValueError("firefly_factor must be >= 1")
    if r["mode"] not in ("path", "lambertian"):
        raise ValueError("render mode must be path or lambertian")
    if not 0 < r["exposure_percentile"] <= 100:
        raise ValueError("exposure_percentile must be in (0,100]")
    if not math.isfinite(r["sun_elevation"]) or not -90 <= r["sun_elevation"] <= 90:
        raise ValueError("Invalid sun_elevation")
    if not math.isfinite(r["sun_azimuth"]):
        raise ValueError("Invalid sun_azimuth")
    if not 1 <= r["turbidity"] <= 10:
        raise ValueError("turbidity must be in [1,10]")
    positive(checks["clearance_m"], "clearance_m", zero=True)
    positive(checks["facade_depth_tolerance_m"], "facade_depth_tolerance_m")
    integer(checks["probe_rays"], "probe_rays", minimum=16)
    if len(checks["visibility_grid"]) != 2:
        raise ValueError("visibility_grid must contain width and height")
    for count in checks["visibility_grid"]:
        integer(count, "visibility_grid")
    if not 0 < checks["min_facade_fraction"] <= 1:
        raise ValueError("min_facade_fraction must be in (0,1]")


def load_config(path=DEFAULT_CONFIG, scene_override=None):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    validate_config(config)
    config["scene_xml"] = str(Path(scene_override or config["scene_xml"]).expanduser().resolve())
    config["catalogue"] = str((path.parent / config["catalogue"]).resolve())
    return config


def expanded_scene(scene_xml):
    """Inline XML includes and absolutize asset paths, without writing beside assets.

    Supports the reference scene's $default substitutions, including emission_off.
    Return the input dependencies as well as the in-memory XML tree.
    """
    xml_files, asset_files = {}, set()
    defaults = {}

    def substitute(value):
        def replace(match):
            if match[1] not in defaults:
                raise ValueError(f"Unknown XML default ${match[1]} in a file path")
            return defaults[match[1]]
        return re.sub(r"\$(\w+)", replace, value)

    def read(path, chain=()):
        path = Path(path).resolve(strict=True)
        if path in chain:
            raise ValueError(f"Recursive scene include: {path}")
        xml_files[str(path)] = file_hash(path)
        root = ET.parse(path).getroot()
        for default in root.findall("default"):
            defaults.setdefault(default.attrib["name"], default.attrib["value"])

        def expand(parent):
            for child in list(parent):
                if child.tag == "include":
                    included = read(path.parent / substitute(child.attrib["filename"]), (*chain, path))
                    index = list(parent).index(child)
                    parent.remove(child)
                    for item in reversed(list(included)):
                        parent.insert(index, item)
                else:
                    if child.tag == "string" and child.get("name") == "filename":
                        asset = (path.parent / substitute(child.attrib["value"])).resolve(strict=True)
                        child.set("value", str(asset))
                        asset_files.add(str(asset))
                    expand(child)
        expand(root)
        return root

    root = read(scene_xml)
    # Asset metadata catches changed meshes/textures without hashing gigabytes per resume.
    dependencies = dict(xml_sha256=xml_files, assets=[
        dict(path=p, size=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns)
        for p in sorted(asset_files)])
    return root, dependencies


def make_plan(config, target_name, limit_views=None):
    if target_name not in config["targets"]:
        raise ValueError(f"Unknown target {target_name!r}; choices: {list(config['targets'])}")
    if limit_views is not None:
        integer(limit_views, "limit_views")
    target = config["targets"][target_name]
    catalogue = json.loads(Path(config["catalogue"]).read_text())
    building = next((b for b in catalogue["buildings"] if b["id"] == target["building_id"]), None)
    if building is None:
        raise ValueError(f"Building missing from catalogue: {target['building_id']}")
    planes = facade_planes(building, Path(config["scene_xml"]).parent, min_area=5)
    normal = dict(north=[0, 0, 1], south=[0, 0, -1], east=[-1, 0, 0], west=[1, 0, 0])[target["side"]]
    candidates = [p for p in planes if np.dot(p["normal"], normal) > math.cos(math.radians(15))]
    if target.get("facade_id"):
        candidates = [p for p in candidates if p["id"] == target["facade_id"]]
    if not candidates:
        raise ValueError("Selected facade does not match this scene; check catalogue and scene assets")
    plane = max(candidates, key=lambda p: p["area"])
    _, dependencies = expanded_scene(config["scene_xml"])
    part_hashes = {p["filename"]: file_hash(Path(config["scene_xml"]).parent / p["filename"])
                   for p in building["parts"]}
    views, layers = plan_views(plane, building, config["camera"])
    total = len(views)
    if limit_views is not None and limit_views < total:
        indices = np.linspace(0, total-1, limit_views, dtype=int)
        views = [views[i] for i in indices]
    spec = copy.deepcopy(dict(
        schema_version=1, generator_version=__version__, target_name=target_name,
        target=target, scene_xml=config["scene_xml"], building=building, plane=plane,
        coordinate_system="Mitsuba world: metres, +Y up, +Z north, +X west",
        camera=config["camera"], render=config["render"], checks=config["checks"],
        calibration=intrinsics(config["camera"]["width"], config["camera"]["height"],
                               config["camera"]["fov_x_deg"]),
        layers=layers, full_grid_count=total, limit_views=limit_views, views=views,
        scene_signature=digest(dependencies), target_mesh_sha256=part_hashes,
    ))
    spec["fingerprint"] = digest(spec)
    return spec


def save_plan(plan, output):
    output = Path(output)
    manifest = output / "metadata/dataset.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["fingerprint"] != plan["fingerprint"]:
            raise ValueError(f"Different configuration already exists in {output}; use a new output directory")
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to reuse a nonempty directory without a dataset manifest: {output}")
    atomic_json(manifest, plan)
    return manifest


def read_plan(output):
    manifest = Path(output) / "metadata/dataset.json"
    plan = json.loads(manifest.read_text())
    fingerprint = plan.pop("fingerprint")
    if digest(plan) != fingerprint:
        raise ValueError("Dataset manifest changed after planning; regenerate in a new directory")
    plan["fingerprint"] = fingerprint
    return plan

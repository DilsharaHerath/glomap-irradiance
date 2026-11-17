"""Perspective placement and explicit Mitsuba -> COLMAP pose conversion."""

import math

import numpy as np


def look_at(origin, target):
    """Mitsuba camera axes: +X left, +Y up, +Z forward; world +Y up."""
    origin, target = np.asarray(origin, float), np.asarray(target, float)
    forward = target - origin
    length = np.linalg.norm(forward)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError("Camera origin and target must be distinct finite points")
    forward /= length
    up = np.array([0.0, 1.0, 0.0])
    if abs(forward @ up) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((left, np.cross(forward, left), forward))
    transform[:3, 3] = origin
    return transform


def intrinsics(width, height, fov_x_deg):
    focal = width / (2 * math.tan(math.radians(fov_x_deg) / 2))
    return dict(model="PINHOLE", width=width, height=height,
                params=[focal, focal, width / 2, height / 2],
                K=[[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])


def rotation_quaternion(rotation):
    """Hamilton unit quaternion (w,x,y,z), stable at rotations near 180 degrees."""
    r = np.asarray(rotation)
    # Eigenvector construction avoids divisions by small quaternion components.
    k = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]],
        [r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]],
        [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def colmap_pose(cam_to_world):
    """COLMAP axes are +X right, +Y down, +Z forward; poses are world-to-camera."""
    c2w = np.asarray(cam_to_world, dtype=float)
    rotation = np.diag([-1.0, -1.0, 1.0]) @ c2w[:3, :3].T
    translation = -rotation @ c2w[:3, 3]
    return dict(rotation=rotation.tolist(), qvec=rotation_quaternion(rotation).tolist(),
                tvec=translation.tolist())


def axis_positions(low, high, max_spacing):
    """Cover both edges with gaps <= max_spacing (an overlap constraint)."""
    if high < low:
        raise ValueError("Height cutoff leaves no camera rows")
    if high == low:
        return np.array([low])
    return np.linspace(low, high, math.ceil((high-low) / max_spacing) + 1)


def plan_views(plane, building, camera):
    normal, right = np.asarray(plane["normal"]), np.asarray(plane["right"])
    border = camera["border_m"]
    bottom, top = plane["y_min"], plane["y_max"] + border
    minimum = camera["min_height_above_building_m"]
    if minimum is not None:
        bottom = max(bottom, building["bbox_min"][1] + minimum)
    tangent = math.tan(math.radians(camera["fov_x_deg"]) / 2)
    views, layers = [], []
    for layer, offset in enumerate(camera["offsets_m"], 1):
        footprint_width = 2 * offset * tangent
        footprint_height = footprint_width * camera["height"] / camera["width"]
        dx, dy = np.array([footprint_width, footprint_height]) * (1 - camera["overlap"])
        us = axis_positions(plane["u_min"] - border, plane["u_max"] + border, dx)
        ys = axis_positions(bottom, top, dy)
        layers.append(dict(layer=layer, offset_m=offset, columns=len(us), rows=len(ys),
                           footprint_m=[footprint_width, footprint_height],
                           max_spacing_m=[float(dx), float(dy)]))
        for row, y in enumerate(ys):
            row_us = (us[:-1] + us[1:]) / 2 if camera["stagger"] and row % 2 and len(us) > 1 else us
            # Serpentine ordering keeps neighboring filenames spatially nearby.
            columns = list(enumerate(row_us))
            if row % 2:
                columns.reverse()
            for column, u in columns:
                wall = right * u + normal * plane["distance"] + np.array([0, y, 0])
                origin = wall + normal * offset
                # At the expanded border, toe in toward the closest facade point.
                tu = np.clip(u, plane["u_min"] + 0.01, plane["u_max"] - 0.01)
                ty = np.clip(y, plane["y_min"] + 0.01, plane["y_max"] - 0.01)
                target = right * tu + normal * plane["distance"] + np.array([0, ty, 0])
                c2w = look_at(origin, target)
                name = f"L{layer:02d}_r{row:03d}_c{column:03d}.png"
                views.append(dict(id=len(views)+1, name=name, layer=layer, row=row,
                                  column=column, offset_m=offset, wall_u_m=float(u),
                                  world_y_m=float(y), cam_to_world=c2w.tolist(),
                                  colmap=colmap_pose(c2w)))
    return views, layers

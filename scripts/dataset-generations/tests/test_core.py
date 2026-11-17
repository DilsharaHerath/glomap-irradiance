import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facade_dataset.cameras import axis_positions, colmap_pose, intrinsics, look_at
from facade_dataset.planning import DEFAULT_CONFIG, expanded_scene, make_plan, read_plan, save_plan, validate_config
from reconstruct import commands_for


def fixture(root):
    """A tiny standalone facade asset, no dependency on the city or reference repo."""
    vertices = np.array([[-2, 1, 0], [2, 1, 0], [2, 5, 0], [-2, 5, 0]], dtype='<f4')
    triangles = np.array([(3, [0, 1, 2]), (3, [0, 2, 3])], dtype=[('n', 'u1'), ('indices', '<i4', 3)])
    ply = root / 'wall.ply'
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex 4\n'
              'property float x\nproperty float y\nproperty float z\n'
              'element face 2\nproperty list uchar int vertex_indices\nend_header\n')
    ply.write_bytes(header.encode() + vertices.tobytes() + triangles.tobytes())
    building = dict(id='fixture', center=[0, 3, -0.5], bbox_min=[-2, 1, -1], bbox_max=[2, 5, 0],
                    parts=[dict(filename='wall.ply')])
    scene = root / 'scene.xml'
    scene.write_text('''<scene version="3.0.0">
      <integrator type="path"/>
      <shape type="ply"><string name="filename" value="wall.ply"/>
        <bsdf type="diffuse"><texture name="reflectance" type="checkerboard">
          <rgb name="color0" value="0.2,0.3,0.6"/><rgb name="color1" value="0.8,0.7,0.4"/>
        </texture></bsdf>
      </shape></scene>''')
    catalogue = root / 'buildings.json'
    catalogue.write_text(json.dumps(dict(buildings=[building])))
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(scene_xml=str(scene), catalogue=str(catalogue),
                  targets={'fixture': dict(building_id='fixture', side='north', facade_id='facade_000')})
    config['camera'].update(width=64, height=48, fov_x_deg=65, offsets_m=[2., 3.], border_m=0., overlap=0.4)
    config['render'].update(spp=4, passes=1, save_depth=True, max_depth=2)
    validate_config(config)
    return config


def quaternion_matrix(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class CameraTests(unittest.TestCase):
    def test_colmap_projection_and_camera_centers(self):
        # Include 180-degree and non-axis-aligned rotations.
        for target in ([0, 2, 0], [4, 3, 10], [-8, 4, -1]):
            c2w = look_at([1, 2, 5], target)
            self.assertAlmostEqual(np.linalg.det(c2w[:3, :3]), 1)
            pose = colmap_pose(c2w)
            r, t = np.array(pose['rotation']), np.array(pose['tvec'])
            np.testing.assert_allclose(quaternion_matrix(pose['qvec']), r, atol=1e-12)
            np.testing.assert_allclose(-r.T @ t, c2w[:3, 3], atol=1e-12)
            local = np.array([0.3, -0.2, 2.])
            world = c2w[:3, :3] @ local + c2w[:3, 3]
            np.testing.assert_allclose(r @ world + t, [-0.3, 0.2, 2], atol=1e-12)
            f, _, cx, cy = intrinsics(800, 600, 60)['params']
            xy = (r @ world + t)[:2] / (r @ world + t)[2]
            np.testing.assert_allclose(xy*f + [cx, cy], [cx-local[0]*f/2, cy-local[1]*f/2])

    def test_spacing_bounds_maximum_for_overlap(self):
        for span in (0, .2, 2.1, 10):
            positions = axis_positions(1, 1+span, .7)
            self.assertTrue(np.all(np.diff(positions) <= .7+1e-12))
            self.assertEqual(positions[0], 1)
            self.assertEqual(positions[-1], 1+span)


class PlanTests(unittest.TestCase):
    def test_plan_is_deterministic_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = fixture(root)
            plan = make_plan(config, 'fixture', 5)
            self.assertEqual(plan, make_plan(config, 'fixture', 5))
            self.assertEqual(len(plan['views']), 5)
            for view in plan['views']:
                self.assertLess(np.array(view['cam_to_world'])[:3, 2] @ plan['plane']['normal'], 0)
            out = root / 'dataset'
            save_plan(plan, out)
            self.assertEqual(read_plan(out), plan)
            config['camera']['fov_x_deg'] = 60
            with self.assertRaisesRegex(ValueError, 'Different configuration'):
                save_plan(make_plan(config, 'fixture', 5), out)
            path = out / 'metadata/dataset.json'
            edited = json.loads(path.read_text())
            edited['render']['spp'] = 16
            path.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError, 'changed after planning'):
                read_plan(out)

    def test_defaults_include_the_two_reference_facades(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        self.assertEqual(config['targets']['north_28']['facade_id'], 'facade_001')
        self.assertEqual(config['targets']['east']['facade_id'], 'facade_003')

    def test_invalid_settings_are_rejected(self):
        base = json.loads(DEFAULT_CONFIG.read_text())
        for group, key, value in [('camera', 'overlap', 1), ('camera', 'overlap', float('nan')),
                                  ('camera', 'offsets_m', [0]), ('camera', 'width', True),
                                  ('render', 'spp', 7), ('checks', 'min_facade_fraction', -1)]:
            config = copy.deepcopy(base)
            config[group][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config(config)

    def test_scene_includes_are_read_only_and_paths_are_absolute(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            scene = root / 'scene.xml'
            scene.rename(root / 'shapes.xml')
            scene.write_text('<scene version="3.0.0"><default name="part" value="shapes"/>'
                             '<include filename="$part.xml"/></scene>')
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            tree, deps = expanded_scene(scene)
            self.assertEqual(tree.find('.//string').get('value'), str(root / 'wall.ply'))
            self.assertEqual(len(deps['xml_sha256']), 2)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})

    def test_reconstruction_uses_intrinsics_without_ground_truth_poses(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = make_plan(fixture(root), 'fixture', 5)
            commands = commands_for(plan, root/'dataset', root/'run', 'colmap', 'glomap', cpu=True)
            self.assertTrue(all(isinstance(command, list) for command in commands))
            self.assertNotIn('ground_truth', str(commands))
            self.assertIn('PINHOLE', commands[0])
            self.assertIn('--BundleAdjustment.optimize_intrinsics', commands[-1])
            self.assertEqual(commands[0][commands[0].index('--FeatureExtraction.use_gpu')+1], '0')


if __name__ == '__main__':
    unittest.main()

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facade_dataset.cameras import look_at
from facade_dataset.planning import expanded_scene, make_plan, save_plan
from facade_dataset.rendering import check_view, depth_map, init_backend, load_scene, render_dataset, sensor_spec
from reconstruct import checked_dataset
from test_core import fixture


@unittest.skipUnless(importlib.util.find_spec('mitsuba'), 'Requires Mitsuba; use the rendering environment')
class RenderTests(unittest.TestCase):
    def test_native_sensor_rays_and_depth_match_exported_calibration(self):
        mi = init_backend('llvm_ad_rgb', None, 2)
        with tempfile.TemporaryDirectory() as folder:
            config = fixture(Path(folder))
            plan = make_plan(config, 'fixture', 4)
            xml, _ = expanded_scene(config['scene_xml'])
            scene = load_scene(mi, xml, plan['render'])
            view = dict(cam_to_world=look_at([0, 3, 2], [0, 3, 0]).tolist())
            sensor = mi.load_dict(sensor_spec(mi, view, plan['calibration'], 4))
            calibration = plan['calibration']
            w, h = calibration['width'], calibration['height']
            uv = np.array([[.4, .3], [.5, .5], [.6, .7]], dtype=np.float32)
            ray, _ = sensor.sample_ray(0, 0, mi.Point2f(mi.Float(uv[:, 0]), mi.Float(uv[:, 1])), mi.Point2f(0))
            d = np.asarray(ray.d).reshape(3, -1).T
            c2w = np.asarray(view['cam_to_world'])
            local = d @ c2w[:3, :3]
            fx, fy, cx, cy = calibration['params']
            pixels = np.column_stack((cx-fx*local[:, 0]/local[:, 2], cy-fy*local[:, 1]/local[:, 2]))
            np.testing.assert_allclose(pixels, uv * [w, h], atol=2e-5)
            z = depth_map(mi, scene, sensor, c2w, w, h)
            np.testing.assert_allclose(z[z > 0], 2.0, atol=1e-5)
            reason, metrics = check_view(mi, scene, view, plan)
            self.assertIsNone(reason)
            self.assertGreater(metrics['facade_fraction'], .5)
            reverse = dict(cam_to_world=look_at([0, 3, 2], [0, 3, 4]).tolist())
            self.assertEqual(check_view(mi, scene, reverse, plan)[0], 'facade_occluded')

    def test_partial_render_resume_and_integrity(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = fixture(root)
            plan = make_plan(config, 'fixture', 4)
            out = root / 'dataset'
            save_plan(plan, out)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
            state = render_dataset(out, variant='llvm_ad_rgb', threads=2, max_new_views=1)
            self.assertEqual(state['status'], 'partial')
            first = next((out/'hdr').glob('*.exr'))
            first_time = first.stat().st_mtime_ns
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                checked_dataset(out)
            state = render_dataset(out, variant='llvm_ad_rgb', threads=2)
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(first.stat().st_mtime_ns, first_time)
            self.assertEqual(checked_dataset(out)['fingerprint'], plan['fingerprint'])
            pngs = sorted((out/'images').glob('*.png'))
            self.assertEqual(len(pngs), 4)
            for image in pngs:
                with Image.open(image) as opened:
                    self.assertEqual(opened.size, (64, 48))
            self.assertEqual(len(list((out/'depth').glob('*.npy'))), 4)
            self.assertEqual(len(json.loads((out/'ground_truth/poses.json').read_text())['views']), 4)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
            # A partial repair must not finalize while another required depth is missing.
            depths = sorted((out/'depth').glob('*.npy'))
            depths[0].unlink()
            depths[1].unlink()
            state = render_dataset(out, variant='llvm_ad_rgb', threads=2, max_new_views=1)
            self.assertEqual(state['status'], 'partial')
            state = render_dataset(out, variant='llvm_ad_rgb', threads=2)
            self.assertEqual(state['status'], 'complete')
            pngs[0].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed after finalization'):
                checked_dataset(out)


if __name__ == '__main__':
    unittest.main()

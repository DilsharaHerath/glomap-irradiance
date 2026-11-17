#!/usr/bin/env python3
"""Plan and render calibrated perspective images of the reference facades."""

import argparse
from pathlib import Path
import sys

from facade_dataset.planning import DEFAULT_CONFIG, ROOT, load_config, make_plan, save_plan


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Read geometry and save camera plans; no Mitsuba or GPU needed")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--scene-xml", type=Path, help="Override the scene path in the configuration")
    plan.add_argument("--target", nargs="+", default=["north_28"], help="north_28 (default), east, or all")
    plan.add_argument("--output-root", type=Path, default=ROOT.parents[1] / "data/facades")
    plan.add_argument("--limit-views", type=int, help="Evenly subsample a small, separate pilot dataset")
    plan.add_argument("--dry-run", action="store_true", help="Print plan summaries without writing files")
    render = commands.add_parser("render", help="Render or resume one previously planned dataset")
    render.add_argument("--dataset", type=Path, required=True)
    render.add_argument("--variant", choices=["auto", "cuda_ad_rgb", "llvm_ad_rgb"], default="auto")
    render.add_argument("--gpu", help="CUDA device ID or UUID; selected before Mitsuba is imported")
    render.add_argument("--threads", type=int, default=8)
    render.add_argument("--max-new-views", type=int, help="Stop after this many new images; rerun to resume")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "plan":
        config = load_config(args.config, args.scene_xml)
        targets = list(config["targets"]) if args.target == ["all"] else args.target
        if len(set(targets)) != len(targets) or ("all" in targets and len(targets) != 1):
            raise ValueError("Choose distinct targets or 'all' on its own")
        for target in targets:
            plan = make_plan(config, target, args.limit_views)
            output = (args.output_root / target).resolve()
            print(f"{target}: {plan['target']['building_id']}, {plan['plane']['id']} ({plan['plane']['side']})")
            print(f"  {len(plan['views'])} candidate cameras; {plan['full_grid_count']} in full grid")
            print(f"  {plan['camera']['width']}x{plan['camera']['height']}, horizontal FOV {plan['camera']['fov_x_deg']:g} deg")
            print(f"  offsets {plan['camera']['offsets_m']} m; nominal overlap {100*plan['camera']['overlap']:g}%")
            print(f"  output: {output}")
            if not args.dry_run:
                save_plan(plan, output)
    else:
        from facade_dataset.rendering import render_dataset
        render_dataset(args.dataset, variant=args.variant, gpu=args.gpu,
                       threads=args.threads, max_new_views=args.max_new_views)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

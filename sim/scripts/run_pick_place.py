"""Run the scripted (classic) pick-and-place for N randomised episodes and record the results.

  python sim/scripts/run_pick_place.py --viewer                 # watch one episode
  python sim/scripts/run_pick_place.py --episodes 20            # success rate, writes a CSV
  python sim/scripts/run_pick_place.py --episodes 1 --gif media/sim_pick_place.gif
"""
import argparse
import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from so101_sim.pick_place import SimRobot, pick_and_place
from so101_sim.scene import BLOCKS, SLOT_SITES, load_scene, placement_error, reset

ROOT = Path(__file__).resolve().parents[2]


class GifRecorder:
    """Grabs frames from a scene camera every `1/fps` simulated seconds."""

    def __init__(self, model, camera="front", fps=12, size=(320, 240)):
        self.renderer = mujoco.Renderer(model, height=size[1], width=size[0])
        self.camera, self.period, self.next_t, self.frames = camera, 1.0 / fps, 0.0, []

    def __call__(self, model, data):
        if data.time >= self.next_t:
            self.renderer.update_scene(data, camera=self.camera)
            self.frames.append(self.renderer.render().copy())
            self.next_t += self.period

    def save(self, path: Path, speedup: float = 2.0):
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in self.frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0,
                     duration=int(1000 * self.period / speedup))
        print(f"saved {path} ({len(imgs)} frames, {speedup:g}x speed)")


def main(args):
    rows, n_ok = [], 0
    viewer = None
    for ep in range(args.episodes):
        model, data = load_scene()
        rng = np.random.default_rng(args.seed + ep)
        reset(model, data, rng=rng, xy_jitter=args.xy_jitter, yaw_jitter_deg=args.yaw_jitter)

        hooks = []
        recorder = GifRecorder(model) if args.gif and ep == 0 else None
        if recorder:
            hooks.append(recorder)
        if args.viewer:
            viewer = mujoco.viewer.launch_passive(model, data)

            def sync(m, d, v=viewer):
                v.sync()
                time.sleep(m.opt.timestep / args.speed)
            hooks.append(sync)

        robot = SimRobot(model, data, on_step=lambda m, d: [h(m, d) for h in hooks])
        for block, slot in zip(BLOCKS, SLOT_SITES):
            ok, msg = pick_and_place(robot, block, slot)
            e = placement_error(model, data, block.name, slot)
            n_ok += ok
            rows.append({"episode": ep, "block": block.name, "width_mm": block.width * 1000,
                         "slot": slot, "success": int(ok), "message": msg,
                         "err_across_mm": round(e["across"] * 1000, 2),
                         "err_along_mm": round(e["along"] * 1000, 2),
                         "err_yaw_deg": round(float(np.degrees(e["yaw"])), 2)})
            print(f"episode {ep:3d}  {block.name:<10} -> {slot}: {'OK  ' if ok else 'FAIL'} {msg:<28}"
                  f" along {e['along'] * 1000:+5.1f} mm  across {e['across'] * 1000:+5.1f} mm")
        if recorder:
            recorder.save(ROOT / args.gif)
        if viewer is not None:
            # Let the user close the window: shutting the viewer down from code while its
            # render thread is still running can crash (segfault) on Linux/WSL.
            print("Close the viewer window to continue.")
            while viewer.is_running():
                time.sleep(0.05)
            viewer = None

    total = len(rows)
    print(f"\nSuccess: {n_ok}/{total} = {100 * n_ok / total:.1f} %")
    for b in BLOCKS:
        r = [x for x in rows if x["block"] == b.name]
        ok = sum(x["success"] for x in r)
        print(f"  {b.name:<10} ({b.width * 1000:4.1f} mm): {ok}/{len(r)}")
    if args.csv:
        out = ROOT / args.csv
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--xy-jitter", type=float, default=0.01, help="m, random block offset in supply area")
    p.add_argument("--yaw-jitter", type=float, default=10.0, help="deg, random block rotation")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--speed", type=float, default=1.0, help="viewer playback speed")
    p.add_argument("--gif", type=str, default=None, help="record episode 0 to this GIF (repo-relative)")
    p.add_argument("--csv", type=str, default="experiments/sim_classic_pick_place.csv")
    main(p.parse_args())

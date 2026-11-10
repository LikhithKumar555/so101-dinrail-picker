"""Open the DIN-rail scene in the viewer, or render the two scene cameras to PNG files.

  python sim/scripts/view_scene.py              # interactive viewer
  python sim/scripts/view_scene.py --render     # save media/scene_overhead.png and media/scene_front.png
"""
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
from PIL import Image

from so101_sim.scene import load_scene, reset

MEDIA = Path(__file__).resolve().parents[2] / "media"


def main(render: bool):
    model, data = load_scene()
    reset(model, data, arm_pose="park" if render else "ready")
    if render:
        MEDIA.mkdir(exist_ok=True)
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for _ in range(int(0.5 / model.opt.timestep)):   # let the arm settle in the park pose
                mujoco.mj_step(model, data)
            for cam in ("overhead", "front"):
                renderer.update_scene(data, camera=cam)
                out = MEDIA / f"scene_{cam}.png"
                Image.fromarray(renderer.render()).save(out)
                print("saved", out)
        return
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    main(parser.parse_args().render)

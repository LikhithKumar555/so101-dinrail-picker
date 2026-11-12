#!/usr/bin/env python3
"""Import the official SO-101 URDF + meshes from TheRobotStudio/SO-ARM100 into this package.

Usage:  python3 scripts/import_urdf.py <path-to-SO-ARM100-checkout>

What it does (so the step is reproducible and documented):
  * copies the STL meshes into meshes/
  * rewrites mesh paths  assets/x.stl  ->  package://so101_description/meshes/x.stl
  * removes the ROS 1 style <transmission> blocks (ros2_control is configured separately)
  * writes urdf/so101_arm.urdf
"""
import re
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SRC_REL = Path("Simulation/SO101")
URDF_NAME = "so101_new_calib.urdf"


def main(repo: Path):
    src = repo / SRC_REL
    urdf = (src / URDF_NAME).read_text()

    (PKG / "meshes").mkdir(exist_ok=True)
    meshes = sorted((src / "assets").glob("*.stl"))
    for stl in meshes:
        shutil.copy2(stl, PKG / "meshes" / stl.name)

    urdf = urdf.replace('filename="assets/', 'filename="package://so101_description/meshes/')
    urdf = re.sub(r"\s*<transmission.*?</transmission>", "", urdf, flags=re.S)
    header = ("<!-- Imported from TheRobotStudio/SO-ARM100 (Apache-2.0), "
              f"{SRC_REL / URDF_NAME}, by scripts/import_urdf.py -->\n")
    urdf = urdf.replace("<robot ", header + "<robot ", 1)
    (PKG / "urdf" / "so101_arm.urdf").write_text(urdf)
    shutil.copy2(repo / "LICENSE", PKG / "LICENSE.SO-ARM100")
    print(f"copied {len(meshes)} meshes, wrote urdf/so101_arm.urdf")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).expanduser().resolve())

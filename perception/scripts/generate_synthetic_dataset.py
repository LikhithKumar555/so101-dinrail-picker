"""Generate a labelled synthetic dataset (YOLO-OBB format) from the MuJoCo scene.

  python perception/scripts/generate_synthetic_dataset.py --n 3000
Writes data/synthetic_blocks/{images,labels}/{train,val,test} + dataset.yaml (data/ is git-ignored)
and media/synthetic_examples.png (a few samples with their labels drawn).
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from so101_perception.synthetic import (CLASS_NAMES, SyntheticRenderer, draw_labels,
                                        yolo_obb_line)

ROOT = Path(__file__).resolve().parents[2]


def main(a):
    out = ROOT / a.out
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    for s in splits:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    r = SyntheticRenderer(a.width, a.height)
    rng = np.random.default_rng(a.seed)
    counts, previews, t0 = np.zeros(len(CLASS_NAMES), int), [], time.time()
    for i in range(a.n):
        frac = i / a.n
        split = "train" if frac < splits["train"] else "val" if frac < splits["train"] + splits["val"] else "test"
        s = r.sample(rng)
        name = f"{split}_{i:05d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / split / f"{name}.txt").write_text(
            "\n".join(yolo_obb_line(c, k, a.width, a.height) for c, k in s.labels))
        for c, _ in s.labels:
            counts[c] += 1
        if len(previews) < 6:
            previews.append(draw_labels(s.image, s.labels))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{a.n} images  ({(time.time() - t0) / (i + 1) * 1000:.0f} ms/image)")
    (out / "dataset.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    grid = np.vstack([np.hstack(previews[0:3]), np.hstack(previews[3:6])])
    grid = cv2.resize(grid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    (ROOT / "media").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "media" / "synthetic_examples.png"), grid)
    print(f"done: {a.n} images in {time.time() - t0:.0f} s -> {out}")
    print("labelled blocks per class:", dict(zip(CLASS_NAMES, counts.tolist())))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default="data/synthetic_blocks")
    main(p.parse_args())

"""Train a YOLO26-OBB detector on the synthetic dataset.

  python perception/scripts/train_yolo.py                      # GPU, defaults below
  python perception/scripts/train_yolo.py --epochs 3 --fraction 0.1   # quick smoke test
Results (weights, curves, confusion matrix) go to outputs/yolo/<name>/ (git-ignored).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def main(a):
    model = YOLO(a.model)                       # COCO/DOTA-pretrained weights: transfer learning
    model.train(
        data=str(ROOT / a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        workers=a.workers, fraction=a.fraction, patience=15, seed=0,
        project=str(ROOT / "outputs" / "yolo"), name=a.name, exist_ok=True,
        # augmentations on top of our domain randomisation
        degrees=10.0, flipud=0.5, fliplr=0.5, hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, mosaic=1.0,
    )
    print("best weights:", ROOT / "outputs" / "yolo" / a.name / "weights" / "best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26n-obb.pt")
    p.add_argument("--data", default="data/synthetic_blocks/dataset.yaml")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8, help="lower to 4 if the GPU runs out of memory")
    p.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' = CPU")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--fraction", type=float, default=1.0, help="use only part of the training set")
    p.add_argument("--name", default="blocks_obb")
    main(p.parse_args())

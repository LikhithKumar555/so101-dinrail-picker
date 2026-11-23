"""Write the printable block test sheet (step 4.2b) as a vector PDF, plus its ground truth.

  python perception/scripts/make_test_sheet.py --out /mnt/d/so101_data/block_test_sheet.pdf
  python perception/scripts/make_test_sheet.py --grayscale ...   # for a black-and-white printer

Print on A4 at 100 % / "Actual size" (NOT "Fit to page"). Then MEASURE the 100 mm scale bar
and one checkerboard square (15 mm). A uniform scale error does not change the measured
positions in board millimetres (board and blocks scale together), but note it in the results.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from so101_perception.print_sheet import (BACKGROUNDS, BOARD_ORIGIN_MM, LAYOUT, SHEET_BOARD,
                                          block_corners_mm, save_sheet_pdf, truth)
from so101_perception.synthetic import CLASS_NAMES


def main(a):
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_sheet_pdf(out, a.background, a.grayscale)
    gt = {
        "frame": "board frame: origin at inner corner (0,0), x = page right, y = page down, "
                 "z into the paper; metres",
        "board": {"inner_corners": [SHEET_BOARD.cols, SHEET_BOARD.rows], "square_m": SHEET_BOARD.square,
                  "origin_on_page_mm": BOARD_ORIGIN_MM.tolist()},
        "blocks": [{"class": CLASS_NAMES[cls], "centre_m": np.round(xy, 6).tolist(),
                    "yaw_deg": round(float(np.degrees(yaw)), 3), "width_mm": width,
                    "colour": b.colour, "corners_on_page_mm": np.round(block_corners_mm(b), 3).tolist()}
                   for b, (cls, xy, yaw, width) in zip(LAYOUT, truth())],
    }
    gt_path = out.with_suffix(".json")
    gt_path.write_text(json.dumps(gt, indent=1))
    print(f"wrote {out} (A4 landscape) and {gt_path}")
    print("print at 100 % / 'Actual size', then measure: scale bar = 100 mm, one square = 15 mm")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/mnt/d/so101_data/block_test_sheet.pdf")
    p.add_argument("--background", choices=sorted(BACKGROUNDS), default="dark",
                   help="grey level of the 'table' behind the blocks")
    p.add_argument("--grayscale", action="store_true", help="no blue blocks (black-and-white printer)")
    main(p.parse_args())

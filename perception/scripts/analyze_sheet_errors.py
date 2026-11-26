"""Where do the real-photo errors of the sheet test come from? (step 4.2b, diagnosis)

  python perception/scripts/analyze_sheet_errors.py
  python perception/scripts/analyze_sheet_errors.py --real experiments/sheet_test_real.csv \\
      --dry experiments/sheet_test_dryrun.csv

Splits every block's position error over all photos into
  * a SYSTEMATIC part: the mean error vector of that block (the same in every photo).
    It comes from geometry: print scale (x and y may differ), a sheet that is not flat,
    calibration errors. It grows with the distance from the checkerboard (lever arm).
  * a RANDOM part: the scatter around that mean (changes from photo to photo).
    It comes from the detector: blur, noise, light.
Also: which blocks are found how often (real vs dry run), and the class confusion.
Writes media/sheet_test_error_field.png: the mean error arrows drawn on the sheet, enlarged.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from so101_perception.print_sheet import BOARD_ORIGIN_MM, LAYOUT, SHEET_BOARD

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["found"] = r["found"] == "True"
    return rows


def per_block(rows):
    out = {}
    for i in range(len(LAYOUT)):
        rb = [r for r in rows if int(r["block"]) == i]
        f = [r for r in rb if r["found"] and r.get("est_x_mm")]
        e = (np.array([[float(r["est_x_mm"]) - float(r["true_x_mm"]),
                        float(r["est_y_mm"]) - float(r["true_y_mm"])] for r in f])
             if f else np.zeros((0, 2)))
        out[i] = {"found": len([r for r in rb if r["found"]]) / max(len(rb), 1), "n": len(f), "e": e}
    return out


def main(a):
    real = load(ROOT / a.real)
    if "est_x_mm" not in real[0]:
        raise SystemExit("this CSV has no positions - run evaluate_sheet_test.py again (new version)")
    dry = load(ROOT / a.dry) if (ROOT / a.dry).exists() else None
    pb, pd = per_block(real), (per_block(dry) if dry else None)
    board_centre = BOARD_ORIGIN_MM + SHEET_BOARD.square * 1000 * np.array(
        [(SHEET_BOARD.cols - 1) / 2, (SHEET_BOARD.rows - 1) / 2])

    print(f"{'#':>2} {'block':<12} {'found real':>10} {'dry':>5}  {'mean error (dx, dy) mm':>23} "
          f"{'|mean|':>6} {'scatter':>7} {'dist board':>10}")
    sys_mag, scat, dists = [], [], []
    for i, b in enumerate(LAYOUT):
        r = pb[i]
        d = float(np.linalg.norm(np.asarray(b.centre_mm) - board_centre))
        name = f"{b.width_mm:4.1f} {b.colour}"
        dry_txt = f"{pd[i]['found']:5.0%}" if pd else "    -"
        if r["n"] >= 2:
            m = r["e"].mean(axis=0)
            sd = float(np.sqrt(np.mean(np.sum((r["e"] - m) ** 2, axis=1))))
            sys_mag.append(np.linalg.norm(m))
            scat.append(sd)
            dists.append(d)
            print(f"{i:2d} {name:<12} {r['found']:10.0%} {dry_txt}  ({m[0]:+6.2f}, {m[1]:+6.2f}) "
                  f"{'':>7} {np.linalg.norm(m):6.2f} {sd:7.2f} {d:8.0f} mm")
        else:
            print(f"{i:2d} {name:<12} {r['found']:10.0%} {dry_txt}  {'(too few detections)':>23}")
    if sys_mag:
        sys_mag, scat, dists = map(np.array, (sys_mag, scat, dists))
        print(f"\nsystematic part (|mean error| per block): median {np.median(sys_mag):.2f} mm")
        print(f"random part (scatter per block)         : median {np.median(scat):.2f} mm")
        if len(dists) >= 3:
            c = np.corrcoef(dists, sys_mag)[0, 1]
            slope = np.polyfit(dists, sys_mag, 1)[0]
            print(f"systematic error vs distance from the board: correlation {c:+.2f}, "
                  f"slope {slope * 100:+.2f} mm per 100 mm")
        # a pure scale error looks like  error = (s_x - 1) * x,  (s_y - 1) * y  in board coordinates
        all_e, all_p = [], []
        for i, b in enumerate(LAYOUT):
            if pb[i]["n"] >= 2:
                all_e.append(pb[i]["e"].mean(axis=0))
                all_p.append(np.asarray(b.centre_mm) - BOARD_ORIGIN_MM)
        E, P = np.array(all_e), np.array(all_p)
        if len(E) >= 3:
            sx = np.linalg.lstsq(P[:, [0]], E[:, 0], rcond=None)[0][0]
            sy = np.linalg.lstsq(P[:, [1]], E[:, 1], rcond=None)[0][0]
            resid = E - P * [sx, sy]
            print(f"best-fit scale error: x {sx * 100:+.2f} %, y {sy * 100:+.2f} %  -> explains "
                  f"{1 - np.sum(resid ** 2) / np.sum(E ** 2):.0%} of the systematic error")

    for label, rows in (("real", real), ("dry run", dry)):
        if rows:
            f = [r for r in rows if r["found"]]
            c = Counter((int(r["true_class"]), int(r["det_class"])) for r in f)
            print(f"\n{label}: true -> detector class (0 = 5.2, 1 = 8.2, 2 = 12.2 mm): "
                  + ", ".join(f"{t}->{d}: {n}" for (t, d), n in sorted(c.items())))

    # error field picture
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from so101_perception.print_sheet import draw_sheet

    fig = plt.figure(figsize=(297 / 25.4, 210 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    draw_sheet(ax, labels=False)
    k = a.arrow_scale
    for i, b in enumerate(LAYOUT):
        if pb[i]["n"] >= 2:
            m = pb[i]["e"].mean(axis=0)
            cx, cy = b.centre_mm
            ax.annotate("", xy=(cx + k * m[0], cy + k * m[1]), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
            ax.text(cx + 3, cy - 4, f"{np.linalg.norm(m):.1f} mm", color="yellow", fontsize=6)
    ax.text(150, 205, f"red arrows: mean position error per block, enlarged {k:g}x", fontsize=6,
            ha="center")
    out = ROOT / "media/sheet_test_error_field.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real", default="experiments/sheet_test_real.csv")
    p.add_argument("--dry", default="experiments/sheet_test_dryrun.csv")
    p.add_argument("--arrow-scale", type=float, default=10.0)
    main(p.parse_args())

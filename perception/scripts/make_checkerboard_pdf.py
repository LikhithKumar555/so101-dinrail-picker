"""Write a printable A4 checkerboard PDF with exact physical dimensions (vector graphics).

  python perception/scripts/make_checkerboard_pdf.py --out /mnt/d/so101_data/checkerboard_9x6_25mm.pdf

Print at 100 % / "Actual size" (NOT "Fit to page"), then MEASURE: the 100 mm scale bar must be
100 mm, and 9 squares must measure 225 mm. Use your measured square size for calibration.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

A4_W, A4_H = 297.0, 210.0  # mm, landscape


def main(a):
    nx, ny = a.cols + 1, a.rows + 1                 # squares = inner corners + 1
    bw, bh = nx * a.square_mm, ny * a.square_mm
    x0, y0 = (A4_W - bw) / 2, (A4_H - bh) / 2 + 4   # leave room for text at the bottom
    fig = plt.figure(figsize=(A4_W / 25.4, A4_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W)
    ax.set_ylim(0, A4_H)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                ax.add_patch(Rectangle((x0 + i * a.square_mm, y0 + j * a.square_mm),
                                       a.square_mm, a.square_mm, color="black", lw=0))
    # 100 mm scale bar to verify the print scale
    ax.plot([x0, x0 + 100], [y0 - 7, y0 - 7], color="black", lw=1.5)
    for x in (x0, x0 + 100):
        ax.plot([x, x], [y0 - 9, y0 - 5], color="black", lw=1.5)
    ax.text(x0 + 105, y0 - 8.2, "100 mm", fontsize=8, va="center")
    ax.text(x0 + bw, y0 - 8.2,
            f"{a.cols}x{a.rows} inner corners, {a.square_mm:g} mm squares - print at 100 %",
            fontsize=8, va="center", ha="right")
    fig.savefig(a.out)
    print(f"wrote {a.out}: board {bw:.0f} x {bh:.0f} mm on A4 landscape")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--square-mm", type=float, default=25.0)
    p.add_argument("--out", default="checkerboard_9x6_25mm.pdf")
    main(p.parse_args())

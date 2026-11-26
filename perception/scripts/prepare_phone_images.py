"""Phone photos -> images the pipeline can use (calibration and the 4.2b sheet test).

  python perception/scripts/prepare_phone_images.py --src /mnt/d/so101_data/phone_raw/calib \\
      --dst /mnt/d/so101_data/phone_calib
  python perception/scripts/prepare_phone_images.py --src /mnt/d/so101_data/phone_raw/sheet \\
      --dst /mnt/d/so101_data/phone_sheet

Why this step exists - camera intrinsics are only valid for ONE image geometry:
  * same lens: phones switch silently between main, ultrawide (macro) and tele lenses.
    The EXIF focal length tells which lens took a photo; photos from another lens are dropped.
  * same orientation: a portrait photo is the landscape photo rotated by 90 deg, which swaps
    fx/fy and cx/cy - a different K. Portrait photos are dropped. Landscape held the other way
    round (rotated 180 deg) mirrors cx, cy - photos with the less common EXIF rotation are dropped.
  * same size: every photo is resized to the same width (default 1280 px). Resizing by a factor
    s simply scales fx, fy, cx, cy by s, so calibration and test photos must get the SAME size.
  * resolution modes: many phones switch between full resolution (e.g. 50 MP, 8160 x 6144) and
    2 x 2 pixel binning (12.5 MP, 4080 x 3072). Binning combines neighbouring pixels of the SAME
    sensor area, so the field of view is identical - after resizing to the same width both give
    the same K. Accepted if the aspect ratio is identical (a crop would change it).
OpenCV applies the EXIF rotation when it reads a JPEG, so "landscape" means landscape as viewed.
"""
import argparse
from pathlib import Path

import cv2
from PIL import Image

EXIF_IFD, FOCAL_LENGTH, FOCAL_35MM, MODEL, ORIENTATION = 0x8769, 37386, 41989, 272, 274


def lens_info(path: Path):
    """(model, focal length [mm], 35 mm-equivalent focal length, EXIF rotation) - None if missing."""
    try:
        exif = Image.open(path).getexif()
        sub = exif.get_ifd(EXIF_IFD)
        f = sub.get(FOCAL_LENGTH)
        return (exif.get(MODEL), (round(float(f), 2) if f is not None else None), sub.get(FOCAL_35MM),
                exif.get(ORIENTATION, 1))
    except Exception:
        return None, None, None, 1


def main(a):
    src, dst = Path(a.src), Path(a.dst)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        heic = list(src.glob("*.heic")) + list(src.glob("*.HEIC"))
        raise SystemExit(f"no JPG/PNG images in {src}" + (
            " - found HEIC files: on iPhone set Settings > Camera > Formats > 'Most Compatible'" if heic else ""))
    dst.mkdir(parents=True, exist_ok=True)

    infos = {f: lens_info(f) for f in files}
    focals = [i[1] for i in infos.values() if i[1] is not None]
    main_focal = max(set(focals), key=focals.count) if focals else None   # the most used lens
    rots = [i[3] for i in infos.values()]
    main_rot = max(set(rots), key=rots.count)                             # the usual way to hold it
    ref_size, kept, skipped, modes = None, 0, [], {}
    for f in files:
        model, focal, f35, rot = infos[f]
        if main_focal is not None and focal is not None and focal != main_focal:
            skipped.append((f.name, f"other lens (focal {focal} mm, main {main_focal} mm)"))
            continue
        if rot != main_rot:
            skipped.append((f.name, f"phone held differently (EXIF rotation {rot}, usual {main_rot})"))
            continue
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)            # applies the EXIF rotation
        if img is None:
            skipped.append((f.name, "cannot read"))
            continue
        h, w = img.shape[:2]
        if h > w:
            skipped.append((f.name, f"portrait ({w} x {h}) - hold the phone in landscape"))
            continue
        if ref_size is None:
            ref_size = (w, h)
        if abs(w / h - ref_size[0] / ref_size[1]) > 1e-3:          # different crop = different camera
            skipped.append((f.name, f"aspect ratio of {w} x {h} differs from {ref_size[0]} x {ref_size[1]}"))
            continue
        if (w, h) != ref_size:
            modes[(w, h)] = modes.get((w, h), 0) + 1
        s = a.width / w
        out = cv2.resize(img, (a.width, round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / f"{f.stem}.png"), out)
        kept += 1

    model, focal, f35, _ = next(iter(infos.values()))
    print(f"camera: {model or 'unknown'}, main lens focal length {main_focal} mm"
          + (f" ({f35} mm equivalent)" if f35 else ""))
    if ref_size:
        print(f"original {ref_size[0]} x {ref_size[1]} -> {a.width} x {round(ref_size[1] * a.width / ref_size[0])}")
    for (w, h), n in modes.items():
        print(f"  {n} photos in another resolution mode ({w} x {h}, same aspect ratio = same field of view): kept")
    print(f"kept {kept} of {len(files)} photos -> {dst}")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder with the phone photos (JPG)")
    p.add_argument("--dst", required=True, help="output folder (PNG, resized)")
    p.add_argument("--width", type=int, default=1280, help="output width; use the SAME for all sets")
    main(p.parse_args())

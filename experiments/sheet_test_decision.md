# 4.2b real-camera test: threshold fixed before the real photos

Decided on 2026-11-23, from the synthetic dry run only (30 views of the printed sheet,
real webcam calibration, seed 0), BEFORE any real photo was taken.

Rule: highest graspable rate with at most 1 false detection per 10 images (after the size check).

| conf | graspable | false detections (after size check) |
|------|-----------|-------------------------------------|
| 0.5  | 63.9 %    | 0 |
| 0.25 | 81.9 %    | 0 |
| 0.1  | 90.3 %    | 1 |

=> Real test runs with conf = 0.1, imgsz = 960, width tolerance 1.5 mm, size check ON.

Dry-run baseline at conf 0.1: found 90.3 %, fused class 100 %, xy error 0.36 / 1.07 mm (median / p95),
yaw 0.17 / 0.69 deg. Detector class errors are all "one size too small" (28/28), explained by
the missing side faces of a flat print.

"""Debug: inspect lane pixel counts and fit residuals at a specific frame."""
import sys
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils.utils import LoadImages, driving_area_mask, lane_line_mask, select_device
from lane_metrics import (
    CX, CY, IMG_H, IMG_W, Y_EVAL, CAM_H, FOCAL_PX,
    MIN_LANE_PX, MAX_RESIDUAL,
    SOURCE, WEIGHTS, IMG_SZ,
    find_lane_clusters,
)

TARGET_FRAME = 2000

device = select_device('0')
half   = device.type != 'cpu'
model  = torch.jit.load(WEIGHTS)
model  = model.to(device)
if half:
    model.half()
model.eval()

dummy = torch.zeros(1, 3, IMG_SZ, IMG_SZ).to(device)
if half:
    dummy = dummy.half()
with torch.no_grad():
    for _ in range(3):
        model(dummy)

dataset = LoadImages(SOURCE, img_size=IMG_SZ, stride=32)
frame_idx = 0

with torch.no_grad():
    for path, img, im0, cap in dataset:
        if frame_idx < TARGET_FRAME:
            frame_idx += 1
            continue

        img_t = torch.from_numpy(img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        [_, __], seg, ll = model(img_t)
        da_mask = driving_area_mask(seg)
        ll_mask  = lane_line_mask(ll)

        print(f"=== Frame {frame_idx} ===")
        print(f"ll_mask shape: {ll_mask.shape}, dtype: {ll_mask.dtype}")
        print(f"ll_mask total pixels (before crop): {np.sum(ll_mask > 0)}")

        ll_mask[:int(IMG_H * 0.4)] = 0
        total_px = int(np.sum(ll_mask > 0))
        print(f"ll_mask total pixels (after top-40% crop): {total_px}")

        ys, xs = np.where(ll_mask > 0)
        if len(xs) > 0:
            print(f"  x range: {xs.min()} - {xs.max()},  y range: {ys.min()} - {ys.max()}")
            left_cx = int(CX * 1.3)
            lmask = xs < left_cx
            rmask = xs >= CX
            lx, ly = xs[lmask], ys[lmask]
            rx, ry = xs[rmask], ys[rmask]
            print(f"  left cluster (xs<{left_cx}): {len(lx)} pixels")
            print(f"  right cluster (xs>={CX}): {len(rx)} pixels")

            # Try fits with verbose residual
            for name, (cxs, cys) in [('LEFT', (lx, ly)), ('RIGHT', (rx, ry))]:
                if len(cxs) >= MIN_LANE_PX:
                    try:
                        coeffs = np.polyfit(cys, cxs, 2)
                        residual = np.mean((np.polyval(coeffs, cys) - cxs) ** 2)
                        print(f"  {name}: {len(cxs)}px, residual={residual:.1f} (limit={MAX_RESIDUAL}) -> {'OK' if residual <= MAX_RESIDUAL else 'FAIL residual too high'}")
                    except Exception as e:
                        print(f"  {name}: fit error: {e}")
                else:
                    print(f"  {name}: {len(cxs)}px < {MIN_LANE_PX} min -> FAIL not enough points")
        else:
            print("  No ll_mask pixels!")

        break

"""Quick preview: processes first MAX_FRAMES from the video and saves a screenshot."""
import os, sys
import numpy as np
import cv2
import torch
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils.utils import (
    LoadImages, driving_area_mask, lane_line_mask,
    select_device, time_synchronized,
)

# import everything from lane_metrics
from lane_metrics import (
    find_lane_clusters, fit_from_points, poly_x, curvature_px, curvature_m,
    lane_type, yaw_angle, draw_overlay, draw_center_lines, draw_deviation_arrow,
    draw_metrics_panel, Smoother, px_to_world, _f,
    CX, CY, IMG_W, IMG_H, Y_EVAL, FOCAL_PX, CAM_H, HALF_FRONT,
    OUTLIER_R_PX, SMOOTH_N,
    SOURCE, WEIGHTS, IMG_SZ, OUT_DIR,
)

MAX_FRAMES   = 620
SAVE_FRAME   = 600         # which frame to save as screenshot
OUT_PNG      = str(OUT_DIR / 'screenshot.png')

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
print("Warmup...")
with torch.no_grad():
    for _ in range(5):
        model(dummy)
torch.cuda.synchronize()
print("Warmup done.")

dataset = LoadImages(SOURCE, img_size=IMG_SZ, stride=32)
fps_src = 30.0

sm_keys = [
    'lateral_offset_m', 'lateral_offset_px',
    'lane_centre_dist_m', 'lane_centre_dist_px',
    'yaw_deg',
    'curv_left_m', 'curv_left_px',
    'curv_right_m', 'curv_right_px',
    'lane_width_m', 'lane_width_px',
    'dist_left_wheel_m', 'dist_right_wheel_m',
    'detect_dist_m',
]
sm = {k: Smoother() for k in sm_keys}
prev_curv_left_m = None
frame_idx = 0

OUT_DIR.mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    for path, img, im0, cap in dataset:
        if frame_idx >= MAX_FRAMES:
            break
        if cap is not None:
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

        img_t = torch.from_numpy(img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        [pred, anchor_grid], seg, ll = model(img_t)
        da_mask = driving_area_mask(seg)
        ll_mask  = lane_line_mask(ll)
        ll_mask[:int(IMG_H * 0.4)] = 0

        lx, ly, rx, ry = find_lane_clusters(ll_mask)
        left_c  = fit_from_points(lx, ly)
        right_c = fit_from_points(rx, ry)

        y_b = Y_EVAL
        left_x_px  = poly_x(left_c,  y_b)
        right_x_px = poly_x(right_c, y_b)

        if left_x_px is not None and right_x_px is not None:
            lane_cx_px = (left_x_px + right_x_px) / 2.0
        elif left_x_px is not None:
            lane_cx_px = left_x_px + 190.0
        elif right_x_px is not None:
            lane_cx_px = right_x_px - 190.0
        else:
            lane_cx_px = None

        if lane_cx_px is not None:
            off_px_raw = float(CX - lane_cx_px)
            dv         = y_b - CY
            off_m_raw  = off_px_raw * CAM_H / dv if dv > 0 else None
        else:
            off_px_raw = off_m_raw = None

        lat_off_px = sm['lateral_offset_px'].update(off_px_raw)
        lat_off_m  = sm['lateral_offset_m'].update(off_m_raw)
        lcd_px     = sm['lane_centre_dist_px'].update(abs(off_px_raw) if off_px_raw is not None else None)
        lcd_m      = sm['lane_centre_dist_m'].update(abs(off_m_raw) if off_m_raw is not None else None)

        yaw_raw = yaw_angle(left_c, right_c)
        yaw_s   = sm['yaw_deg'].update(yaw_raw) if yaw_raw is not None else None

        cl_px_raw = curvature_px(left_c,  y_b)
        cr_px_raw = curvature_px(right_c, y_b)
        cl_m_raw  = curvature_m(left_c,   y_b)
        cr_m_raw  = curvature_m(right_c,  y_b)

        cl_px_s = sm['curv_left_px'].update(cl_px_raw)
        cr_px_s = sm['curv_right_px'].update(cr_px_raw)
        cl_m_s  = sm['curv_left_m'].update(cl_m_raw)
        cr_m_s  = sm['curv_right_m'].update(cr_m_raw)

        if cl_m_s is not None and prev_curv_left_m is not None:
            curv_change = cl_m_s - prev_curv_left_m
        else:
            curv_change = None
        prev_curv_left_m = cl_m_s

        if left_x_px is not None and right_x_px is not None:
            w_px_raw = right_x_px - left_x_px
            dv2 = y_b - CY
            w_m_raw = (w_px_raw * CAM_H / dv2) if (w_px_raw > 0 and dv2 > 0) else None
        else:
            w_px_raw = w_m_raw = None
        lw_px_s = sm['lane_width_px'].update(w_px_raw)
        lw_m_s  = sm['lane_width_m'].update(w_m_raw)

        dv3 = y_b - CY
        spm = dv3 / CAM_H if dv3 > 0 else None
        dl_m_raw = (CX - HALF_FRONT * spm - left_x_px) / spm if (spm and left_x_px is not None) else None
        dr_m_raw = (right_x_px - CX - HALF_FRONT * spm) / spm if (spm and right_x_px is not None) else None
        dl_m_s = sm['dist_left_wheel_m'].update(dl_m_raw)
        dr_m_s = sm['dist_right_wheel_m'].update(dr_m_raw)

        lt_left  = lane_type(lx, ly)
        lt_right = lane_type(rx, ry)

        dd_raw = None
        for (xs_, ys_) in [(lx, ly), (rx, ry)]:
            if xs_ is not None and len(ys_) > 0:
                top_y = int(np.min(ys_))
                _, z  = px_to_world(CX, top_y)
                if z is not None:
                    dd_raw = max(dd_raw or 0.0, z)
        dd_m_s = sm['detect_dist_m'].update(dd_raw)

        if lat_off_m is not None:
            v_lane = 'sol serit' if lat_off_m < -0.15 else ('sag serit' if lat_off_m > 0.15 else 'merkez')
        else:
            v_lane = 'belirsiz'

        adj_left  = 'var' if (lx is not None and len(lx) >= 20 and left_x_px is not None and np.min(lx) < (left_x_px - 80)) else 'yok'
        adj_right = 'var' if (rx is not None and len(rx) >= 20 and right_x_px is not None and np.max(rx) > (right_x_px + 80)) else 'yok'
        _dv_rel  = Y_EVAL - CY
        _spm_rel = _dv_rel / CAM_H if _dv_rel > 0 else None
        _lat_m_rel = (abs(lat_off_px) / _spm_rel) if (_spm_rel and lat_off_px is not None) else None
        reliability = (
            'dusuk' if (left_c is None and right_c is None)
            else 'dusuk' if (_lat_m_rel is not None and _lat_m_rel > 0.8)
            else 'yuksek'
        )
        timestamp   = frame_idx / fps_src

        metrics = {
            'frame': frame_idx, 'timestamp': round(timestamp, 3),
            'lateral_offset_px': lat_off_px, 'lateral_offset_m': lat_off_m,
            'lane_centre_dist_px': lcd_px, 'lane_centre_dist_m': lcd_m,
            'yaw_deg': yaw_s,
            'curv_left_px': cl_px_s, 'curv_left_m': cl_m_s,
            'curv_right_px': cr_px_s, 'curv_right_m': cr_m_s,
            'curv_change': curv_change,
            'lane_width_px': lw_px_s, 'lane_width_m': lw_m_s,
            'dist_left_wheel_m': dl_m_s, 'dist_right_wheel_m': dr_m_s,
            'lane_type_left': lt_left, 'lane_type_right': lt_right,
            'vehicle_lane': v_lane, 'adj_left_lane': adj_left, 'adj_right_lane': adj_right,
            'detect_dist_m': dd_m_s, 'reliability': reliability,
        }

        frame = im0.copy()
        draw_overlay(frame, da_mask, ll_mask)
        draw_center_lines(frame, lane_cx_px)
        fps_live = 30.0
        draw_metrics_panel(frame, metrics, fps_live)
        draw_deviation_arrow(frame, lat_off_m)

        if frame_idx == SAVE_FRAME or (frame_idx == MAX_FRAMES - 1 and not os.path.exists(OUT_PNG)):
            cv2.imwrite(OUT_PNG, frame)
            print("Screenshot saved:", OUT_PNG)
            print("  yaw_deg =", _f(yaw_s, 'deg'))
            print("  lat_off =", _f(lat_off_m, 'm'))
            print("  left_c  =", left_c is not None)
            print("  right_c =", right_c is not None)
            print("  reliability =", reliability)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print("Frame", frame_idx)

print("Done. Preview complete.")

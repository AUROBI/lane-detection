#!/usr/bin/env python3
"""
Lane Metrics - CLRNet Edition
YoloPv2  → driving-area maskesi (yeşil arka plan)
CLRNet   → şerit tespiti + metrik hesaplamaları
"""

import sys, time, csv
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import deque

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.utils import LoadImages, driving_area_mask
from utils.clrnet_infer import CLRNetPredictor

# ── Araç / kamera sabitleri ───────────────────────────────────────────────
TRACK_FRONT = 0.886      # m — ön iz genişliği
CAM_H       = 0.685      # m — kamera yüksekliği
HALF_FRONT  = TRACK_FRONT / 2.0

IMG_W, IMG_H = 1280, 720
FOCAL_PX     = 700.0 * IMG_W / 1920.0
CX           = IMG_W // 2
CY           = IMG_H // 2
Y_EVAL       = int(IMG_H * 0.85)   # metrik hesaplama satırı

# ── Pipeline ayarları ─────────────────────────────────────────────────────
SMOOTH_N      = 8
WARMUP_N      = 20
MIN_FIT_PTS   = 4
MAX_RESIDUAL  = 150
MAX_CURV_PX   = 8000
MAX_SLOPE_DEG = 65.0
MIN_LANE_W_PX = 180
MAX_LANE_W_PX = 950
LANE_THRESH   = 0.30

SOURCE  = str(ROOT / 'data/demo/test_short.mp4')
WEIGHTS = str(ROOT / 'data/weights/yolopv2.pt')
CLRNET  = str(ROOT / 'data/weights/culane_r18.pth')
IMG_SZ  = 640
OUT_DIR = ROOT / 'runs/metrics_clrnet'

CSV_FIELDS = [
    'frame', 'timestamp',
    'lateral_offset_px', 'lateral_offset_m',
    'lane_centre_dist_px', 'lane_centre_dist_m',
    'yaw_deg',
    'curv_left_px', 'curv_left_m', 'curv_left_raw',
    'curv_right_px', 'curv_right_m', 'curv_right_raw',
    'curv_change',
    'lane_width_px', 'lane_width_m',
    'dist_left_wheel_m', 'dist_right_wheel_m',
    'lane_type_left', 'lane_type_right',
    'vehicle_lane', 'adj_left_lane', 'adj_right_lane',
    'detect_dist_m', 'reliability',
    'clrnet_lane_count', 'clrnet_max_conf',
]


# ── Yardımcı sınıflar ─────────────────────────────────────────────────────

class Smoother:
    def __init__(self, n=SMOOTH_N):
        self.buf = deque(maxlen=n)

    def update(self, val):
        if val is not None and np.isfinite(float(val)):
            self.buf.append(float(val))
        return float(np.mean(self.buf)) if self.buf else None


# ── Şerit / polinom yardımcıları ─────────────────────────────────────────

def _x_at(pts, y_ref):
    """pts (N,2) içinde y_ref'e en yakın noktanın x değeri."""
    idx = np.argmin(np.abs(pts[:, 1] - y_ref))
    return float(pts[idx, 0])


def _validate_poly(coeffs, side):
    if coeffs is None:
        return None
    x_bot = np.polyval(coeffs, Y_EVAL)
    x_top = np.polyval(coeffs, int(IMG_H * 0.4))
    if not (0 <= x_bot <= IMG_W) or not (0 <= x_top <= IMG_W):
        return None
    if side == 'left'  and x_bot > CX * 1.4:
        return None
    if side == 'right' and x_bot < CX * 0.6:
        return None
    dy    = Y_EVAL - int(IMG_H * 0.4)
    angle = abs(np.degrees(np.arctan2(abs(x_bot - x_top), dy)))
    return coeffs if angle <= MAX_SLOPE_DEG else None


def fit_poly(xs, ys, side):
    if xs is None or len(xs) < MIN_FIT_PTS:
        return None
    try:
        c = np.polyfit(ys, xs, 2)
        res = float(np.mean((np.polyval(c, ys) - xs) ** 2))
        return _validate_poly(c, side) if res <= MAX_RESIDUAL else None
    except Exception:
        return None


def poly_x(c, y):
    return float(np.polyval(c, y)) if c is not None else None


def lanes_to_polys(clrnet_lanes):
    """CLRNet şerit noktaları → sol/sağ polinom + ham noktalar."""
    left_cands, right_cands = [], []
    for pts in clrnet_lanes:
        if len(pts) < MIN_FIT_PTS:
            continue
        x_ev = _x_at(pts, Y_EVAL)
        xs, ys = pts[:, 0].astype(float), pts[:, 1].astype(float)
        dist = abs(x_ev - CX)
        if x_ev < CX:
            left_cands.append((dist, xs, ys))
        else:
            right_cands.append((dist, xs, ys))

    lxs = lys = rxs = rys = None
    left_c = right_c = None

    if left_cands:
        left_cands.sort(key=lambda t: t[0])   # merkeze en yakın
        _, lxs, lys = left_cands[0]
        left_c = fit_poly(lxs, lys, 'left')

    if right_cands:
        right_cands.sort(key=lambda t: t[0])
        _, rxs, rys = right_cands[0]
        right_c = fit_poly(rxs, rys, 'right')

    return left_c, right_c, lxs, lys, rxs, rys


# ── Geometri / metrik hesaplamaları ──────────────────────────────────────

def px_to_world(u, v):
    dv = float(v) - CY
    if dv <= 0:
        return None, None
    return (float(u) - CX) * CAM_H / dv, FOCAL_PX * CAM_H / dv


def curvature_px(c):
    if c is None:
        return None
    A, B, _ = c
    if abs(A) < 1e-10:
        return None
    R = (1.0 + (2*A*Y_EVAL + B)**2)**1.5 / abs(2*A)
    return float(R) if R <= MAX_CURV_PX else None


def curvature_m(c):
    R_px = curvature_px(c)
    if R_px is None:
        return None
    dv = Y_EVAL - CY
    return R_px * (FOCAL_PX * CAM_H / dv) / FOCAL_PX if dv > 0 else None


def curv_label(R_m):
    if R_m is None or R_m > 500: return 'duz yol'
    if R_m > 150: return 'hafif viraj'
    if R_m > 50:  return 'orta viraj'
    return 'sert viraj'


def calc_yaw(left_c, right_c):
    c = left_c if left_c is not None else right_c
    if c is None:
        return None
    y_bot, y_top = Y_EVAL, max(int(IMG_H * 0.35), Y_EVAL - 350)
    xb, xt = poly_x(c, y_bot), poly_x(c, y_top)
    if xb is None or xt is None:
        return None
    return float(np.degrees(np.arctan2(xb - xt, float(y_bot - y_top))))


def lane_type(xs, ys):
    if xs is None or len(xs) < 8:
        return 'belirsiz'
    ys_b = ys[ys >= int(IMG_H * 0.4)]
    if len(ys_b) < 6:
        return 'belirsiz'
    band = 12
    y_min, y_max = int(ys_b.min()), int(ys_b.max())
    dolu = bos = 0
    for y0 in range(y_min, y_max, band):
        (dolu if np.any((ys_b >= y0) & (ys_b < y0 + band)) else bos).__class__  # count
        if np.any((ys_b >= y0) & (ys_b < y0 + band)):
            dolu += 1
        else:
            bos += 1
    total = dolu + bos
    if total == 0:
        return 'belirsiz'
    r = bos / total
    return 'surekli' if r < 0.25 else ('kesikli' if r < 0.60 else 'belirsiz')


# ── Çizim ─────────────────────────────────────────────────────────────────

def draw_da_overlay(frame, da_mask):
    ov = np.zeros_like(frame, dtype=np.uint8)
    ov[da_mask == 1] = (0, 180, 0)
    cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)


def draw_clrnet_lanes(frame, clrnet_lanes, confs, left_c, right_c):
    for i, pts in enumerate(clrnet_lanes):
        x_ev  = _x_at(pts, Y_EVAL)
        color = (255, 100, 0) if x_ev < CX else (0, 100, 255)  # turuncu=sol, mavi=sağ
        for j in range(len(pts) - 1):
            cv2.line(frame,
                     (int(pts[j, 0]),   int(pts[j, 1])),
                     (int(pts[j+1, 0]), int(pts[j+1, 1])),
                     color, 3, cv2.LINE_AA)
        # Güven skoru etiketi
        if i < len(confs):
            mid = len(pts) // 2
            cv2.putText(frame, f"{confs[i]:.2f}",
                        (int(pts[mid, 0]) + 4, int(pts[mid, 1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # Polinom çizgisi (ince, farklı renk)
    for c, color in [(left_c, (0, 230, 255)), (right_c, (255, 230, 0))]:
        if c is None:
            continue
        pts_p = [(int(poly_x(c, y)), y)
                 for y in range(int(IMG_H*0.4), IMG_H, 3)
                 if 0 <= poly_x(c, y) <= IMG_W]
        if len(pts_p) > 2:
            cv2.polylines(frame, [np.array(pts_p)], False, color, 2, cv2.LINE_AA)


def draw_center_lines(frame, lane_cx):
    cv2.line(frame, (CX, 400), (CX, IMG_H-1), (255, 255, 255), 1, cv2.LINE_AA)
    if lane_cx is not None:
        lc = int(lane_cx)
        cv2.line(frame, (lc, 400), (lc, IMG_H-1), (0, 220, 220), 2, cv2.LINE_AA)
        cv2.line(frame, (CX, IMG_H-55), (lc, IMG_H-55), (255, 255, 0), 1, cv2.LINE_AA)


def draw_arrow(frame, offset_m):
    if offset_m is None:
        return
    cy_row = IMG_H - 35
    col    = ((0,200,0) if abs(offset_m) < 0.15 else
              (0,165,255) if abs(offset_m) < 0.40 else (0,0,220))
    length = int(min(abs(offset_m) * 250, 130))
    tip    = (CX + (1 if offset_m > 0 else -1) * length, cy_row)
    cv2.arrowedLine(frame, (CX, cy_row), tip, col, 3, tipLength=0.3)
    cv2.putText(frame, f"{offset_m:+.2f}m",
                (CX-35, cy_row-8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2, cv2.LINE_AA)


def draw_panel(frame, m, fps):
    pw = 340
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (pw, IMG_H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    def put(txt, row, col=(220, 220, 220)):
        cv2.putText(frame, txt, (6, 20 + row*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

    def f(v, unit=''):
        return '-' if v is None else (f"{v:.2f}{unit}" if isinstance(v, float) else f"{v}{unit}")

    cl_lbl = curv_label(m['curv_left_raw'])
    cr_lbl = curv_label(m['curv_right_raw'])

    put(f"FPS: {fps:.1f}  [CLRNet + YOLOPv2]",                                  0, (0, 220, 100))
    put(f"T: {m['timestamp']:.2f}s   F:{m['frame']}",                           1)
    put(f" 1. Sapma          : {f(m['lateral_offset_m'],'m')} / {f(m['lateral_offset_px'],'px')}", 2)
    put(f" 2. Merkez uzaklik : {f(m['lane_centre_dist_m'],'m')} / {f(m['lane_centre_dist_px'],'px')}", 3)
    put(f" 3. Yaw acisi      : {f(m['yaw_deg'],'deg')}",                        4)
    put(f" 4. Sol egrilik    : {f(m['curv_left_m'],'m')}  [{cl_lbl}]",          5)
    put(f" 5. Sag egrilik    : {f(m['curv_right_m'],'m')}  [{cr_lbl}]",         6)
    put(f" 6. Egrilik degisim: {f(m['curv_change'],'m')}",                      7)
    put(f" 7. Serit genisligi: {f(m['lane_width_m'],'m')} / {f(m['lane_width_px'],'px')}", 8)
    put(f" 8. Sol teker uzak : {f(m['dist_left_wheel_m'],'m')}",                9)
    put(f" 9. Sag teker uzak : {f(m['dist_right_wheel_m'],'m')}",              10)
    put(f"10. Sol serit tipi : {m['lane_type_left']}",                          11)
    put(f"11. Sag serit tipi : {m['lane_type_right']}",                         12)
    put('-'*47,                                                                  13, (70, 70, 70))
    put(f"12. Zaman          : {m['timestamp']:.3f}s",                          14)
    put(f"13. Arac seridi    : {m['vehicle_lane']}",                            15)
    put(f"14. Komsu sol/sag  : {m['adj_left_lane']} / {m['adj_right_lane']}",  16)
    put(f"15. Algilama mes.  : {f(m['detect_dist_m'],'m')}",                   17)
    put(f"16. CLRNet serit   : {m['clrnet_lane_count']}  conf:{f(m['clrnet_max_conf'],'')}", 18, (100, 220, 255))
    rel = m['reliability']
    put(f"17. Guvenilirlik   : {rel}",
        19, (0, 255, 0) if rel == 'yuksek' else (0, 0, 255))


# ── Ana döngü ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = str(OUT_DIR / 'output_clrnet.mp4')
    csv_path   = str(OUT_DIR / 'metrics_clrnet.csv')

    # YoloPv2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("UYARI: CUDA bulunamadi, CPU modunda calisiliyor. FPS dusuk olabilir.")
    half = device.type == 'cuda'
    yolo   = torch.jit.load(WEIGHTS).to(device)
    if half:
        yolo.half()
    yolo.eval()
    dummy = (torch.zeros(1, 3, IMG_SZ, IMG_SZ).to(device).half()
             if half else torch.zeros(1, 3, IMG_SZ, IMG_SZ).to(device))
    print(f"YoloPv2 ısınıyor ({WARMUP_N} iter)...")
    with torch.no_grad():
        for _ in range(WARMUP_N):
            yolo(dummy)
    if half:
        torch.cuda.synchronize()

    # CLRNet
    print("CLRNet yükleniyor...")
    clrnet = CLRNetPredictor(CLRNET, device=str(device))
    clrnet.predict(np.zeros((720, 1280, 3), dtype=np.uint8))   # warmup
    print("Hazır! İşleniyor...\n")

    dataset    = LoadImages(SOURCE, img_size=IMG_SZ, stride=32)
    vid_writer = None
    fps_src    = 30.0

    sm = {k: Smoother() for k in [
        'lateral_offset_m', 'lateral_offset_px',
        'lane_centre_dist_m', 'lane_centre_dist_px',
        'yaw_deg',
        'curv_left_m', 'curv_left_px',
        'curv_right_m', 'curv_right_px',
        'lane_width_m', 'lane_width_px',
        'dist_left_wheel_m', 'dist_right_wheel_m',
        'detect_dist_m', 'clrnet_max_conf',
    ]}

    prev_curv_m = None
    frame_idx   = 0
    t_start     = time.time()

    with open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=CSV_FIELDS)
        writer.writeheader()

        with torch.no_grad():
            for path, img, im0, cap in dataset:
                t_frame = time.time()
                if cap is not None:
                    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

                if vid_writer is None:
                    h, w = im0.shape[:2]
                    vid_writer = cv2.VideoWriter(
                        video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                        fps_src, (w, h))

                # ── YoloPv2 → driving area ──
                img_t = (torch.from_numpy(img).to(device).half()
                         if half else torch.from_numpy(img).to(device).float()) / 255.0
                if img_t.ndimension() == 3:
                    img_t = img_t.unsqueeze(0)
                [_, _], seg, _ = yolo(img_t)
                da_mask = driving_area_mask(seg)

                # ── CLRNet → şeritler + güven ──
                clrnet_lanes, confs = clrnet.predict_with_conf(im0)
                max_conf = max(confs) if confs else 0.0

                # Polinom katsayıları
                left_c, right_c, lxs, lys, rxs, rys = lanes_to_polys(clrnet_lanes)

                lx_px = poly_x(left_c,  Y_EVAL)
                rx_px = poly_x(right_c, Y_EVAL)

                # Serit genişliği mantık kontrolü
                if lx_px is not None and rx_px is not None:
                    w_check = rx_px - lx_px
                    if not (MIN_LANE_W_PX <= w_check <= MAX_LANE_W_PX):
                        left_c, lx_px = None, None

                lx_px = poly_x(left_c,  Y_EVAL)
                rx_px = poly_x(right_c, Y_EVAL)

                lane_cx = (
                    (lx_px + rx_px) / 2.0 if lx_px and rx_px else
                    lx_px + 200.0          if lx_px is not None else
                    rx_px - 200.0          if rx_px is not None else None
                )

                # ── Sapma ──
                off_px = float(CX - lane_cx) if lane_cx is not None else None
                dv     = Y_EVAL - CY
                off_m  = off_px * CAM_H / dv if (off_px is not None and dv > 0) else None
                lat_px = sm['lateral_offset_px'].update(off_px)
                lat_m  = sm['lateral_offset_m'].update(off_m)
                lcd_px = sm['lane_centre_dist_px'].update(abs(off_px) if off_px is not None else None)
                lcd_m  = sm['lane_centre_dist_m'].update(abs(off_m)  if off_m  is not None else None)

                yaw_s = sm['yaw_deg'].update(calc_yaw(left_c, right_c))

                # ── Eğrilik ──
                cl_px      = sm['curv_left_px'].update(curvature_px(left_c))
                cr_px      = sm['curv_right_px'].update(curvature_px(right_c))
                cl_m_raw   = curvature_m(left_c)
                cr_m_raw   = curvature_m(right_c)
                cl_m       = sm['curv_left_m'].update(cl_m_raw)
                cr_m       = sm['curv_right_m'].update(cr_m_raw)

                avg_c = ((cl_m + cr_m) / 2.0 if (cl_m and cr_m) else cl_m or cr_m)
                curv_chg = None
                if avg_c is not None and prev_curv_m is not None:
                    dc = avg_c - prev_curv_m
                    curv_chg = dc if abs(dc) < 50 else None
                prev_curv_m = avg_c

                # ── Serit genişliği ──
                w_px = (rx_px - lx_px) if (lx_px and rx_px) else None
                w_m  = (w_px * CAM_H / dv) if (w_px and w_px > 0 and dv > 0) else None
                lw_px = sm['lane_width_px'].update(w_px)
                lw_m  = sm['lane_width_m'].update(w_m)

                # ── Teker uzaklıkları ──
                spm = dv / CAM_H if dv > 0 else None
                dl_m = dr_m = None
                if spm:
                    if lx_px is not None:
                        dl_m = (CX - HALF_FRONT * spm - lx_px) / spm
                    if rx_px is not None:
                        dr_m = (rx_px - (CX + HALF_FRONT * spm)) / spm
                dl_m_s = sm['dist_left_wheel_m'].update(dl_m)
                dr_m_s = sm['dist_right_wheel_m'].update(dr_m)

                lt_l = lane_type(lxs, lys)
                lt_r = lane_type(rxs, rys)

                # ── Algılama mesafesi ──
                dd_m = None
                for pts in clrnet_lanes:
                    if len(pts) > 0:
                        _, z = px_to_world(CX, int(pts[:, 1].min()))
                        if z is not None:
                            dd_m = max(dd_m or 0.0, z)
                dd_m_s = sm['detect_dist_m'].update(dd_m)

                conf_s = sm['clrnet_max_conf'].update(max_conf)

                # ── Araç şeridi / komşu şerit ──
                v_lane = ('sol serit' if lat_m is not None and lat_m < -LANE_THRESH else
                          'sag serit' if lat_m is not None and lat_m >  LANE_THRESH else
                          'merkez'    if lat_m is not None else 'belirsiz')

                has_far_left  = any(_x_at(p, Y_EVAL) < CX * 0.30 for p in clrnet_lanes)
                has_far_right = any(_x_at(p, Y_EVAL) > IMG_W - CX * 0.30 for p in clrnet_lanes)

                reliability = ('dusuk' if (left_c is None and right_c is None)
                               or (lat_m is not None and abs(lat_m) > 0.9)
                               else 'yuksek')

                metrics = {
                    'frame':               frame_idx,
                    'timestamp':           round(frame_idx / fps_src, 3),
                    'lateral_offset_px':   lat_px,
                    'lateral_offset_m':    lat_m,
                    'lane_centre_dist_px': lcd_px,
                    'lane_centre_dist_m':  lcd_m,
                    'yaw_deg':             yaw_s,
                    'curv_left_px':        cl_px,
                    'curv_left_m':         cl_m,
                    'curv_left_raw':       cl_m_raw,
                    'curv_right_px':       cr_px,
                    'curv_right_m':        cr_m,
                    'curv_right_raw':      cr_m_raw,
                    'curv_change':         curv_chg,
                    'lane_width_px':       lw_px,
                    'lane_width_m':        lw_m,
                    'dist_left_wheel_m':   dl_m_s,
                    'dist_right_wheel_m':  dr_m_s,
                    'lane_type_left':      lt_l,
                    'lane_type_right':     lt_r,
                    'vehicle_lane':        v_lane,
                    'adj_left_lane':       'var' if has_far_left  else 'yok',
                    'adj_right_lane':      'var' if has_far_right else 'yok',
                    'detect_dist_m':       dd_m_s,
                    'reliability':         reliability,
                    'clrnet_lane_count':   len(clrnet_lanes),
                    'clrnet_max_conf':     conf_s,
                }

                # ── Kareyi çiz ──
                frame = im0.copy()
                draw_da_overlay(frame, da_mask)
                draw_clrnet_lanes(frame, clrnet_lanes, confs, left_c, right_c)
                draw_center_lines(frame, lane_cx)
                fps_live = 1.0 / (time.time() - t_frame + 1e-9)
                draw_panel(frame, metrics, fps_live)
                draw_arrow(frame, lat_m)

                vid_writer.write(frame)

                row = {k: (f"{v:.4f}" if isinstance(v, float) else
                           (str(v) if v is not None else ''))
                       for k, v in metrics.items()}
                writer.writerow(row)

                frame_idx += 1
                if frame_idx % 300 == 0:
                    elapsed = time.time() - t_start
                    fps_avg = frame_idx / elapsed
                    eta     = (3601 - frame_idx) / fps_avg
                    print(f"[{frame_idx:5d}/3601] {elapsed:5.1f}s  "
                          f"fps_avg={fps_avg:.1f}  ETA={eta:.0f}s  "
                          f"lat={f'{lat_m:.2f}m' if lat_m else 'N/A'}  "
                          f"serit={len(clrnet_lanes)}  "
                          f"conf={max_conf:.2f}  rel={reliability}")

    if vid_writer:
        vid_writer.release()

    total = time.time() - t_start
    print(f"\nTamamlandı — {frame_idx} kare, {total:.1f}s, "
          f"ortalama {frame_idx/total:.1f} fps")
    print(f"  Video : {video_path}")
    print(f"  CSV   : {csv_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Lane Metrics v4 - Bee1 / YOLOPv2
Connected components + optimize edilmis morfoloji + duzeltilmis egrilik esikleri
"""

import sys, time, csv
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import deque

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils.utils import (
    LoadImages, driving_area_mask, lane_line_mask,
    select_device, time_synchronized,
)

# ── Bee1 / ZED2 sabitleri ──────────────────────────────────────────────────
TRACK_FRONT  = 0.886
CAM_H        = 0.685
HALF_FRONT   = TRACK_FRONT / 2.0

IMG_W, IMG_H = 1280, 720
FOCAL_PX     = 700.0 * IMG_W / 1920.0
CX           = IMG_W // 2
CY           = IMG_H // 2
Y_EVAL       = int(IMG_H * 0.85)

# ── Pipeline ayarlari ──────────────────────────────────────────────────────
SMOOTH_N       = 8
WARMUP_N       = 30
MIN_COMP_PX    = 80
MIN_FIT_PX     = 20
MAX_RESIDUAL   = 120     # dusuk = yanlis fit reddedilir
MAX_CURV_PX    = 8000
LANE_THRESH    = 0.30
MORPH_K        = 3       # kucuk = daha hizli

SOURCE   = 'data/demo/test.mp4'
WEIGHTS  = 'data/weights/yolopv2.pt'
IMG_SZ   = 640
OUT_DIR  = Path('runs/metrics')

CSV_FIELDS = [
    'frame', 'timestamp',
    'lateral_offset_px', 'lateral_offset_m',
    'lane_centre_dist_px', 'lane_centre_dist_m',
    'yaw_deg',
    'curv_left_px', 'curv_left_m',
    'curv_right_px', 'curv_right_m',
    'curv_change',
    'lane_width_px', 'lane_width_m',
    'dist_left_wheel_m', 'dist_right_wheel_m',
    'lane_type_left', 'lane_type_right',
    'vehicle_lane', 'adj_left_lane', 'adj_right_lane',
    'detect_dist_m', 'reliability',
]


class Smoother:
    def __init__(self, n=SMOOTH_N):
        self.buf = deque(maxlen=n)

    def update(self, val):
        if val is not None and np.isfinite(float(val)):
            self.buf.append(float(val))
        return float(np.mean(self.buf)) if self.buf else None


def preprocess_mask(ll_mask):
    """Hafif morfoloji - gurultu temizle, serit cizgilerini guclendir."""
    mask = (ll_mask > 0).astype(np.uint8) * 255
    # Kucuk gurultuyu sil
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    # Dikey olarak hafifce uzat
    k_dil  = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K*2))
    mask   = cv2.dilate(mask, k_dil, iterations=1)
    return mask


def find_lane_components(ll_mask):
    """
    Connected components ile sol/sag serit cizgisini bul.
    Her komponenti merkez x konumuna gore sol/sag sinifla.
    En buyuk sol ve sag komponenti dondur.
    Ek filtre: komponentin y araliginin en az %30'u goruntu altinda olmali.
    """
    processed = preprocess_mask(ll_mask)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        processed, connectivity=8
    )

    left_cands  = []
    right_cands = []
    y_bot_lim   = int(IMG_H * 0.5)   # alt yari kontrolu

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_COMP_PX:
            continue

        ys, xs = np.where(labels == i)

        # Alt bolgede yeterli piksel olmayan komponentleri atla
        if np.sum(ys >= y_bot_lim) < MIN_FIT_PX:
            continue

        # Sadece alt %60'daki pikselleri kullan
        bot_mask = ys >= int(IMG_H * 0.40)
        ys_b = ys[bot_mask]
        xs_b = xs[bot_mask]
        if len(xs_b) < MIN_FIT_PX:
            continue

        cx_comp = float(centroids[i][0])

        if cx_comp < CX:
            left_cands.append((area, xs_b, ys_b))
        else:
            right_cands.append((area, xs_b, ys_b))

    lxs = lys = rxs = rys = None
    if left_cands:
        left_cands.sort(key=lambda x: x[0], reverse=True)
        _, lxs, lys = left_cands[0]
    if right_cands:
        right_cands.sort(key=lambda x: x[0], reverse=True)
        _, rxs, rys = right_cands[0]

    return lxs, lys, rxs, rys


def fit_poly(xs, ys):
    """x = A*y^2 + B*y + C  polinom fit. Dusuk residual kontrolu."""
    if xs is None or len(xs) < MIN_FIT_PX:
        return None
    try:
        coeffs   = np.polyfit(ys, xs, 2)
        residual = float(np.mean((np.polyval(coeffs, ys) - xs) ** 2))
        if residual > MAX_RESIDUAL:
            return None
        # Ek kontrol: fit sonucu mantikli bir x araliginda mi?
        x_bot = float(np.polyval(coeffs, Y_EVAL))
        if not (0 <= x_bot <= IMG_W):
            return None
        return coeffs
    except Exception:
        return None


def poly_x(coeffs, y):
    return float(np.polyval(coeffs, y)) if coeffs is not None else None


def px_to_world(u, v):
    dv = float(v) - CY
    if dv <= 0:
        return None, None
    z = FOCAL_PX * CAM_H / dv
    x = (float(u) - CX) * CAM_H / dv
    return x, z


def curvature_px(coeffs, y_eval=Y_EVAL):
    if coeffs is None:
        return None
    A, B, _ = coeffs
    if abs(A) < 1e-10:
        return None
    R = (1.0 + (2.0*A*y_eval + B)**2)**1.5 / abs(2.0*A)
    return float(R) if R <= MAX_CURV_PX else None


def curvature_m(coeffs, y_eval=Y_EVAL):
    R_px = curvature_px(coeffs, y_eval)
    if R_px is None:
        return None
    dv = y_eval - CY
    if dv <= 0:
        return None
    scale = (FOCAL_PX * CAM_H / dv) / FOCAL_PX
    return R_px * scale


def curv_label(R_m):
    """
    Egrilik etiketi - metre cinsinden.
    Tipik degerler: sehir viraji ~20-80m, otoyol >500m
    """
    if R_m is None:
        return 'duz yol'
    if R_m > 80:
        return 'duz yol'
    if R_m > 25:
        return 'hafif viraj'
    if R_m > 8:
        return 'orta viraj'
    return 'sert viraj'


def lane_type(xs, ys):
    """Serit tipi: surekli / kesikli / belirsiz"""
    if xs is None or len(xs) < 30:
        return 'belirsiz'
    bot_mask = ys >= int(IMG_H * 0.4)
    ys_b = ys[bot_mask]
    if len(ys_b) < 20:
        return 'belirsiz'
    band = 10
    y_min, y_max = int(ys_b.min()), int(ys_b.max())
    dolu = bos = 0
    for y0 in range(y_min, y_max, band):
        if np.any((ys_b >= y0) & (ys_b < y0 + band)):
            dolu += 1
        else:
            bos += 1
    toplam = dolu + bos
    if toplam == 0:
        return 'belirsiz'
    bos_oran = bos / toplam
    if bos_oran < 0.20:
        return 'surekli'
    elif bos_oran < 0.55:
        return 'kesikli'
    return 'belirsiz'


def calc_yaw(left_c, right_c):
    c = left_c if left_c is not None else right_c
    if c is None:
        return None
    y_bot = Y_EVAL
    y_top = max(int(IMG_H * 0.35), y_bot - 350)
    xb = poly_x(c, y_bot)
    xt = poly_x(c, y_top)
    if xb is None or xt is None:
        return None
    return float(np.degrees(np.arctan2(xb - xt, float(y_bot - y_top))))


def clip_da_mask(da_mask, ll_mask):
    """
    Drivable area maskesini serit cizgisi sinirlarinda kirp.
    Park alani gibi yanlis boyanan bolgeleri azaltir.
    """
    # ll_mask'ten sol ve sag siniri bul
    ys, xs = np.where(ll_mask > 0)
    if len(xs) == 0:
        return da_mask

    # Alt %40'ta sol ve sag sinir
    bot = int(IMG_H * 0.6)
    bot_mask = ys >= bot
    xs_b = xs[bot_mask]
    if len(xs_b) < 10:
        return da_mask

    x_left  = max(0,     int(np.percentile(xs_b, 5))  - 30)
    x_right = min(IMG_W, int(np.percentile(xs_b, 95)) + 30)

    clipped = da_mask.copy()
    clipped[:, :x_left]  = 0
    clipped[:, x_right:] = 0
    return clipped


def draw_overlay(frame, da_mask, ll_mask):
    """
    Yesil  = surulebilir alan
    Kirmizi = surulemez bolge (alt %60)
    Mavi   = serit cizgileri
    """
    # da_mask'i serit sinirlarinda kirp
    da_clipped = clip_da_mask(da_mask, ll_mask)

    ov = np.zeros_like(frame, dtype=np.uint8)
    ov[da_clipped == 1] = (0, 180, 0)

    y_start  = int(IMG_H * 0.4)
    red_mask = np.zeros(da_mask.shape, dtype=bool)
    red_mask[y_start:] = (da_clipped[y_start:] == 0) & (ll_mask[y_start:] == 0)
    ov[red_mask] = (0, 0, 160)

    ov[ll_mask == 1] = (200, 60, 0)
    cv2.addWeighted(ov, 0.38, frame, 0.62, 0, frame)


def draw_lane_poly(frame, coeffs, color, y0=None, y1=None):
    if coeffs is None:
        return
    pts = []
    for y in range(y0 or int(IMG_H*0.35), y1 or IMG_H, 4):
        x = int(poly_x(coeffs, y))
        if 0 <= x < IMG_W:
            pts.append((x, y))
    if len(pts) > 2:
        cv2.polylines(frame, [np.array(pts)], False, color, 3, cv2.LINE_AA)


def draw_center_lines(frame, lane_cx_px, y_top=400):
    cv2.line(frame, (CX, y_top), (CX, IMG_H-1), (255,255,255), 1, cv2.LINE_AA)
    if lane_cx_px is not None:
        lc = int(lane_cx_px)
        cv2.line(frame, (lc, y_top), (lc, IMG_H-1), (0,220,220), 2, cv2.LINE_AA)
        y_mid = IMG_H - 55
        cv2.line(frame, (CX, y_mid), (lc, y_mid), (255,255,0), 1, cv2.LINE_AA)


def draw_arrow(frame, offset_m):
    if offset_m is None:
        return
    cy_row = IMG_H - 35
    col = ((0,200,0) if abs(offset_m) < 0.15 else
           (0,165,255) if abs(offset_m) < 0.40 else
           (0,0,220))
    length    = int(min(abs(offset_m) * 250, 130))
    direction = 1 if offset_m > 0 else -1
    tip = (CX + direction * length, cy_row)
    cv2.arrowedLine(frame, (CX, cy_row), tip, col, 3, tipLength=0.3)
    cv2.putText(frame, f"{offset_m:+.2f}m",
                (CX - 35, cy_row - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2, cv2.LINE_AA)


def draw_panel(frame, m, fps):
    pw = 335
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (pw, IMG_H), (0,0,0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    def put(txt, row, col=(220,220,220)):
        cv2.putText(frame, txt, (6, 20 + row*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

    def f(v, unit=''):
        if v is None: return '-'
        return f"{v:.2f}{unit}" if isinstance(v, float) else f"{v}{unit}"

    cl_lbl = curv_label(m['curv_left_m'])
    cr_lbl = curv_label(m['curv_right_m'])

    put(f"FPS: {fps:.1f}",                                                    0, (0,220,100))
    put(f"T: {m['timestamp']:.2f}s   F:{m['frame']}",                        1)
    put(f" 1. Sapma          : {f(m['lateral_offset_m'],'m')} / {f(m['lateral_offset_px'],'px')}", 2)
    put(f" 2. Merkez uzaklik : {f(m['lane_centre_dist_m'],'m')} / {f(m['lane_centre_dist_px'],'px')}", 3)
    put(f" 3. Yaw acisi      : {f(m['yaw_deg'],'deg')}",                     4)
    put(f" 4. Sol egrilik    : {f(m['curv_left_m'],'m')}  [{cl_lbl}]",       5)
    put(f" 5. Sag egrilik    : {f(m['curv_right_m'],'m')}  [{cr_lbl}]",      6)
    put(f" 6. Egrilik degisim: {f(m['curv_change'],'m')}",                   7)
    put(f" 7. Serit genisligi: {f(m['lane_width_m'],'m')} / {f(m['lane_width_px'],'px')}", 8)
    put(f" 8. Sol teker uzak : {f(m['dist_left_wheel_m'],'m')}",             9)
    put(f" 9. Sag teker uzak : {f(m['dist_right_wheel_m'],'m')}",           10)
    put(f"10. Sol serit tipi : {m['lane_type_left']}",                       11)
    put(f"11. Sag serit tipi : {m['lane_type_right']}",                      12)
    put('-'*46,                                                               13, (70,70,70))
    put(f"12. Zaman          : {m['timestamp']:.3f}s",                       14)
    put(f"13. Arac seridi    : {m['vehicle_lane']}",                         15)
    put(f"14. Komsu sol      : {m['adj_left_lane']}",                        16)
    put(f"15. Komsu sag      : {m['adj_right_lane']}",                       17)
    put(f"16. Algilama mes.  : {f(m['detect_dist_m'],'m')}",                18)
    rel = m['reliability']
    put(f"17. Guvenilirlik   : {rel}",
        19, (0,255,0) if rel == 'yuksek' else (0,0,255))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = str(OUT_DIR / 'output.mp4')
    csv_path   = str(OUT_DIR / 'metrics.csv')

    device = select_device('0')
    half   = device.type != 'cpu'
    model  = torch.jit.load(WEIGHTS).to(device)
    if half:
        model.half()
    model.eval()

    dummy = torch.zeros(1, 3, IMG_SZ, IMG_SZ).to(device)
    if half:
        dummy = dummy.half()
    print(f"GPU isiniyor ({WARMUP_N} iter)...")
    with torch.no_grad():
        for _ in range(WARMUP_N):
            model(dummy)
    torch.cuda.synchronize()
    print("Hazir!")

    dataset    = LoadImages(SOURCE, img_size=IMG_SZ, stride=32)
    vid_writer = None
    fps_src    = 30.0

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
                        video_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps_src, (w, h),
                    )

                # Inference
                img_t = torch.from_numpy(img).to(device)
                img_t = (img_t.half() if half else img_t.float()) / 255.0
                if img_t.ndimension() == 3:
                    img_t = img_t.unsqueeze(0)

                t1 = time_synchronized()
                [pred, anchor_grid], seg, ll = model(img_t)
                t2 = time_synchronized()

                da_mask = driving_area_mask(seg)
                ll_mask = lane_line_mask(ll)
                ll_mask[:int(IMG_H * 0.40)] = 0

                # Serit fit
                lxs, lys, rxs, rys = find_lane_components(ll_mask)
                left_c  = fit_poly(lxs, lys)
                right_c = fit_poly(rxs, rys)

                lx_px = poly_x(left_c,  Y_EVAL)
                rx_px = poly_x(right_c, Y_EVAL)

                if lx_px is not None and rx_px is not None:
                    lane_cx = (lx_px + rx_px) / 2.0
                elif lx_px is not None:
                    lane_cx = lx_px + 200.0
                elif rx_px is not None:
                    lane_cx = rx_px - 200.0
                else:
                    lane_cx = None

                # Sapma
                if lane_cx is not None:
                    off_px = float(CX - lane_cx)
                    dv     = Y_EVAL - CY
                    off_m  = off_px * CAM_H / dv if dv > 0 else None
                else:
                    off_px = off_m = None

                lat_px = sm['lateral_offset_px'].update(off_px)
                lat_m  = sm['lateral_offset_m'].update(off_m)
                lcd_px = sm['lane_centre_dist_px'].update(abs(off_px) if off_px is not None else None)
                lcd_m  = sm['lane_centre_dist_m'].update(abs(off_m)  if off_m  is not None else None)

                # Yaw
                yaw_s = sm['yaw_deg'].update(calc_yaw(left_c, right_c))

                # Egrilik
                cl_px = sm['curv_left_px'].update(curvature_px(left_c))
                cr_px = sm['curv_right_px'].update(curvature_px(right_c))
                cl_m  = sm['curv_left_m'].update(curvature_m(left_c))
                cr_m  = sm['curv_right_m'].update(curvature_m(right_c))

                # Egrilik degisimi
                avg_curv = ((cl_m + cr_m) / 2.0 if (cl_m and cr_m) else cl_m or cr_m)
                curv_chg = None
                if avg_curv is not None and prev_curv_m is not None:
                    dc = avg_curv - prev_curv_m
                    curv_chg = dc if abs(dc) < 50 else None
                prev_curv_m = avg_curv

                # Serit genisligi
                if lx_px is not None and rx_px is not None:
                    w_px = rx_px - lx_px
                    dv   = Y_EVAL - CY
                    w_m  = (w_px * CAM_H / dv) if (w_px > 0 and dv > 0) else None
                else:
                    w_px = w_m = None
                lw_px = sm['lane_width_px'].update(w_px)
                lw_m  = sm['lane_width_m'].update(w_m)

                # Teker uzakliklari
                dv  = Y_EVAL - CY
                spm = dv / CAM_H if dv > 0 else None
                dl_m = dr_m = None
                if spm:
                    lwx = CX - HALF_FRONT * spm
                    rwx = CX + HALF_FRONT * spm
                    if lx_px is not None:
                        dl_m = (lwx - lx_px) / spm
                    if rx_px is not None:
                        dr_m = (rx_px - rwx) / spm
                dl_m_s = sm['dist_left_wheel_m'].update(dl_m)
                dr_m_s = sm['dist_right_wheel_m'].update(dr_m)

                # Serit tipi
                lt_l = lane_type(lxs, lys)
                lt_r = lane_type(rxs, rys)

                # Algilama mesafesi
                dd_m = None
                for (xs_, ys_) in [(lxs, lys), (rxs, rys)]:
                    if xs_ is not None and len(ys_) > 0:
                        _, z = px_to_world(CX, int(np.min(ys_)))
                        if z is not None:
                            dd_m = max(dd_m or 0.0, z)
                dd_m_s = sm['detect_dist_m'].update(dd_m)

                # Arac seridi
                if lat_m is not None:
                    if lat_m < -LANE_THRESH:
                        v_lane = 'sol serit'
                    elif lat_m > LANE_THRESH:
                        v_lane = 'sag serit'
                    else:
                        v_lane = 'merkez'
                else:
                    v_lane = 'belirsiz'

                # Komsu seritler
                q     = IMG_W // 8
                adj_l = 'var' if (ll_mask[:, :q].sum() > 150) else 'yok'
                adj_r = 'var' if (ll_mask[:, IMG_W-q:].sum() > 150) else 'yok'

                # Guvenilirlik
                both_none  = (left_c is None and right_c is None)
                big_offset = (lat_m is not None and abs(lat_m) > 0.9)
                reliability = 'dusuk' if (both_none or big_offset) else 'yuksek'

                timestamp = frame_idx / fps_src

                metrics = {
                    'frame':               frame_idx,
                    'timestamp':           round(timestamp, 3),
                    'lateral_offset_px':   lat_px,
                    'lateral_offset_m':    lat_m,
                    'lane_centre_dist_px': lcd_px,
                    'lane_centre_dist_m':  lcd_m,
                    'yaw_deg':             yaw_s,
                    'curv_left_px':        cl_px,
                    'curv_left_m':         cl_m,
                    'curv_right_px':       cr_px,
                    'curv_right_m':        cr_m,
                    'curv_change':         curv_chg,
                    'lane_width_px':       lw_px,
                    'lane_width_m':        lw_m,
                    'dist_left_wheel_m':   dl_m_s,
                    'dist_right_wheel_m':  dr_m_s,
                    'lane_type_left':      lt_l,
                    'lane_type_right':     lt_r,
                    'vehicle_lane':        v_lane,
                    'adj_left_lane':       adj_l,
                    'adj_right_lane':      adj_r,
                    'detect_dist_m':       dd_m_s,
                    'reliability':         reliability,
                }

                # Gorsellestime
                frame = im0.copy()
                draw_overlay(frame, da_mask, ll_mask)
                draw_lane_poly(frame, left_c,  (255, 80,  0))
                draw_lane_poly(frame, right_c, (0,  80, 255))
                draw_center_lines(frame, lane_cx)

                fps_live = 1.0 / (time.time() - t_frame + 1e-6)
                draw_panel(frame, metrics, fps_live)
                draw_arrow(frame, lat_m)

                vid_writer.write(frame)

                row = {k: (f"{v:.4f}" if isinstance(v, float) else
                           (str(v) if v is not None else ''))
                       for k, v in metrics.items()}
                writer.writerow(row)

                frame_idx += 1
                if frame_idx % 200 == 0:
                    elapsed = time.time() - t_start
                    print(f"[{frame_idx:5d}] {elapsed:5.1f}s  "
                          f"lat={f'{lat_m:.2f}m' if lat_m is not None else 'N/A'}  "
                          f"yaw={f'{yaw_s:.1f}deg' if yaw_s is not None else 'N/A'}  "
                          f"R_L={f'{cl_m:.1f}m' if cl_m is not None else '-'}  "
                          f"lt_L={lt_l}  lt_R={lt_r}  "
                          f"rel={reliability}")

    if vid_writer:
        vid_writer.release()

    total = time.time() - t_start
    print(f"\nTamamlandi -- {frame_idx} frame, {total:.1f}s, ort {frame_idx/total:.1f} fps")
    print(f"  Video : {video_path}")
    print(f"  CSV   : {csv_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Lane Metrics - Bee1 / YOLOPv2
Kamera: ZED2  |  Yukseklik: 685mm  |  Odak uzunlugu: ~700px (1080p)
Arac  : Bee1  |  On iz: 886mm      |  Arka iz: 850mm  |  Dingil: 1860mm
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
TRACK_REAR   = 0.850
WHEELBASE    = 1.860
CAM_H        = 0.685
CAM_X_OFF    = 0.205
HALF_FRONT   = TRACK_FRONT / 2.0   # 0.443 m

IMG_W, IMG_H = 1280, 720
FOCAL_PX     = 700.0 * IMG_W / 1920.0   # ~466.7 px
CX           = IMG_W // 2
CY           = IMG_H // 2
Y_EVAL       = int(IMG_H * 0.85)

# ── Pipeline ayarlari ──────────────────────────────────────────────────────
SMOOTH_N     = 10
WARMUP_N     = 30
MIN_LANE_PX  = 8
MAX_RESIDUAL = 200
MAX_CURV_PX  = 8000

# Arac seridi esikleri — Bee1 genisligi 1060mm, serit ~3.5m
# Arac merkezi seridin 0.3m disina cikinca farkli serit sayilir
LANE_OFFSET_THRESH = 0.30   # m

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


# ── Smoother ───────────────────────────────────────────────────────────────
class Smoother:
    def __init__(self, n=SMOOTH_N):
        self.buf = deque(maxlen=n)

    def update(self, val):
        if val is not None and np.isfinite(float(val)):
            self.buf.append(float(val))
        return float(np.mean(self.buf)) if self.buf else None


# ── Serit yardimci fonksiyonlari ───────────────────────────────────────────
def find_lane_clusters(ll_mask):
    """
    Serit piksellerini sol / sag olarak ayir.
    Sol: 0 .. CX*1.3   Sag: CX*0.7 .. sona
    """
    ys, xs = np.where(ll_mask > 0)
    if len(xs) == 0:
        return None, None, None, None

    left_lim  = int(CX * 1.3)
    right_lim = int(CX * 0.7)

    lmask = xs < left_lim
    rmask = xs >= right_lim

    lxs = xs[lmask]; lys = ys[lmask]
    rxs = xs[rmask]; rys = ys[rmask]

    lxs = lxs if len(lxs) >= MIN_LANE_PX else None
    lys = lys if lxs is not None else None
    rxs = rxs if len(rxs) >= MIN_LANE_PX else None
    rys = rys if rxs is not None else None

    return lxs, lys, rxs, rys


def fit_poly(xs, ys):
    if xs is None or len(xs) < MIN_LANE_PX:
        return None
    try:
        coeffs   = np.polyfit(ys, xs, 2)
        residual = float(np.mean((np.polyval(coeffs, ys) - xs) ** 2))
        return coeffs if residual <= MAX_RESIDUAL else None
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
    Egrilik etiketi - metre cinsinden R degerine gore.
    Tipik degerler: duz otoyol > 500m, sehir viraji 20-100m
    """
    if R_m is None:
        return 'duz yol'
    if R_m > 150:
        return 'duz yol'
    if R_m > 50:
        return 'hafif viraj'
    if R_m > 15:
        return 'orta viraj'
    return 'sert viraj'


def lane_type(xs, ys):
    """
    Serit tipi tespiti: surekli / kesikli / belirsiz
    Mantik: serit piksellerini y ekseninde grupla,
    ardisik gruplar arasindaki bosluk oranina gore karar ver.
    """
    if xs is None or len(xs) < 30:
        return 'belirsiz'

    # Sadece alt %60'i kullan (daha guvenilir bolge)
    bot_y = int(IMG_H * 0.4)
    mask  = ys >= bot_y
    ys_b  = ys[mask]
    if len(ys_b) < 20:
        return 'belirsiz'

    # Her 10px'lik banti dolu mu bos mu say
    band  = 10
    y_min = int(ys_b.min())
    y_max = int(ys_b.max())
    dolu  = 0
    bos   = 0
    for y0 in range(y_min, y_max, band):
        if np.any((ys_b >= y0) & (ys_b < y0 + band)):
            dolu += 1
        else:
            bos  += 1

    toplam = dolu + bos
    if toplam == 0:
        return 'belirsiz'

    bos_oran = bos / toplam

    if bos_oran < 0.20:
        return 'surekli'
    elif bos_oran < 0.55:
        return 'kesikli'
    else:
        return 'belirsiz'


def calc_yaw(left_c, right_c):
    """
    Yaw acisi: serit yonune gore aracin acisal sapmasini hesapla.
    Sol serit tercih edilir, yoksa sag serit kullanilir.
    """
    c = left_c if left_c is not None else right_c
    if c is None:
        return None
    y_bot = Y_EVAL
    y_top = max(int(IMG_H * 0.35), y_bot - 350)
    xb = poly_x(c, y_bot)
    xt = poly_x(c, y_top)
    if xb is None or xt is None:
        return None
    # Serit yonu ile dikey eksen arasindaki aci
    angle = float(np.degrees(np.arctan2(xb - xt, float(y_bot - y_top))))
    return angle


# ── Cizim fonksiyonlari ────────────────────────────────────────────────────
def draw_overlay(frame, da_mask, ll_mask):
    """
    Yesil  = surulebilir alan (da_mask==1)
    Kirmizi = surulemez bolge (alt %60, da==0 ve ll==0)
    Mavi   = serit cizgileri (ll_mask==1)
    """
    ov = np.zeros_like(frame, dtype=np.uint8)
    ov[da_mask == 1] = (0, 180, 0)

    # Ters serit / park: sadece alt %60, drivable ve lane disinda
    y_start  = int(IMG_H * 0.4)
    red_mask = np.zeros(da_mask.shape, dtype=bool)
    red_mask[y_start:] = (da_mask[y_start:] == 0) & (ll_mask[y_start:] == 0)
    ov[red_mask] = (0, 0, 200)

    ov[ll_mask == 1] = (200, 60, 0)   # mavi serit cizgileri
    cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)


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


# ── Ana dongu ──────────────────────────────────────────────────────────────
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

                # ── Inference ─────────────────────────────────────────────
                img_t = torch.from_numpy(img).to(device)
                img_t = (img_t.half() if half else img_t.float()) / 255.0
                if img_t.ndimension() == 3:
                    img_t = img_t.unsqueeze(0)

                t1 = time_synchronized()
                [pred, anchor_grid], seg, ll = model(img_t)
                t2 = time_synchronized()

                da_mask = driving_area_mask(seg)
                ll_mask = lane_line_mask(ll)
                ll_mask[:int(IMG_H * 0.40)] = 0   # ufuk gurultusunu kaldir

                # ── Serit fit ─────────────────────────────────────────────
                lxs, lys, rxs, rys = find_lane_clusters(ll_mask)
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

                # ── Metrik 1 & 2: Sapma ───────────────────────────────────
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

                # ── Metrik 3: Yaw ─────────────────────────────────────────
                yaw_s = sm['yaw_deg'].update(calc_yaw(left_c, right_c))

                # ── Metrik 4 & 5: Egrilik ─────────────────────────────────
                cl_px = sm['curv_left_px'].update(curvature_px(left_c))
                cr_px = sm['curv_right_px'].update(curvature_px(right_c))
                cl_m  = sm['curv_left_m'].update(curvature_m(left_c))
                cr_m  = sm['curv_right_m'].update(curvature_m(right_c))

                # ── Metrik 6: Egrilik degisimi ────────────────────────────
                avg_curv = ((cl_m + cr_m) / 2.0 if (cl_m and cr_m)
                            else cl_m or cr_m)
                curv_chg = None
                if avg_curv is not None and prev_curv_m is not None:
                    dc = avg_curv - prev_curv_m
                    curv_chg = dc if abs(dc) < 50 else None
                prev_curv_m = avg_curv

                # ── Metrik 7: Serit genisligi ─────────────────────────────
                if lx_px is not None and rx_px is not None:
                    w_px = rx_px - lx_px
                    dv   = Y_EVAL - CY
                    w_m  = (w_px * CAM_H / dv) if (w_px > 0 and dv > 0) else None
                else:
                    w_px = w_m = None
                lw_px = sm['lane_width_px'].update(w_px)
                lw_m  = sm['lane_width_m'].update(w_m)

                # ── Metrik 8 & 9: Teker uzakliklari ──────────────────────
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

                # ── Metrik 10 & 11: Serit tipi ────────────────────────────
                lt_l = lane_type(lxs, lys)
                lt_r = lane_type(rxs, rys)

                # ── Metrik 16: Algilama mesafesi ──────────────────────────
                dd_m = None
                for (xs_, ys_) in [(lxs, lys), (rxs, rys)]:
                    if xs_ is not None and len(ys_) > 0:
                        _, z = px_to_world(CX, int(np.min(ys_)))
                        if z is not None:
                            dd_m = max(dd_m or 0.0, z)
                dd_m_s = sm['detect_dist_m'].update(dd_m)

                # ── Metrik 13: Arac seridi ────────────────────────────────
                # Genis esik: emniyet seridi vs normal serit ayrimi icin
                if lat_m is not None:
                    if lat_m < -LANE_OFFSET_THRESH:
                        v_lane = 'sol serit'
                    elif lat_m > LANE_OFFSET_THRESH:
                        v_lane = 'sag serit'
                    else:
                        v_lane = 'merkez'
                else:
                    v_lane = 'belirsiz'

                # ── Metrik 14 & 15: Komsu seritler ────────────────────────
                q     = IMG_W // 8
                adj_l = 'var' if (ll_mask[:, :q].sum() > 150) else 'yok'
                adj_r = 'var' if (ll_mask[:, IMG_W-q:].sum() > 150) else 'yok'

                # ── Metrik 17: Guvenilirlik ───────────────────────────────
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

                # ── Gorsellestime ─────────────────────────────────────────
                frame = im0.copy()
                draw_overlay(frame, da_mask, ll_mask)
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
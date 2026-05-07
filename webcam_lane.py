#!/usr/bin/env python3
"""
Webcam Lane - CLRNet + YOLOPv2
Gercek zamanli serit takibi ve metrik gosterimi
Q = cikis  |  S = screenshot  |  R = smoother sifirla
"""

import sys, time
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import deque

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.utils import driving_area_mask, letterbox
from utils.clrnet_infer import CLRNetPredictor

# ── Kamera ayari ──────────────────────────────────────────────────────────
CAM_ID   = 0          # 0 = dahili webcam, 1/2/3 = harici kamera
CAM_W    = 1280
CAM_H_PX = 720

# ── Bee1 / ZED2 sabitleri ──────────────────────────────────────────────────
TRACK_FRONT = 0.886
CAM_H       = 0.685
HALF_FRONT  = TRACK_FRONT / 2.0

IMG_W, IMG_H = 1280, 720
FOCAL_PX     = 700.0 * IMG_W / 1920.0
CX           = IMG_W // 2
CY           = IMG_H // 2
Y_EVAL       = int(IMG_H * 0.85)

# ── Model ayarlari ────────────────────────────────────────────────────────
WEIGHTS  = str(ROOT / 'data/weights/yolopv2.pt')
CLRNET   = str(ROOT / 'data/weights/culane_r18.pth')
IMG_SZ   = 640
WARMUP_N = 20

# ── Pipeline ayarlari ─────────────────────────────────────────────────────
SMOOTH_N      = 6        # webcam icin daha az smoothing (daha reaktif)
MIN_FIT_PTS   = 4
MAX_RESIDUAL  = 150
MAX_CURV_PX   = 8000
MAX_SLOPE_DEG = 65.0
MIN_LANE_W_PX = 180
MAX_LANE_W_PX = 950
LANE_THRESH   = 0.30

OUT_DIR = ROOT / 'runs/webcam'


class Smoother:
    def __init__(self, n=SMOOTH_N):
        self.buf = deque(maxlen=n)

    def update(self, val):
        if val is not None and np.isfinite(float(val)):
            self.buf.append(float(val))
        return float(np.mean(self.buf)) if self.buf else None

    def reset(self):
        self.buf.clear()


def _x_at(pts, y_ref):
    idx = np.argmin(np.abs(pts[:, 1] - y_ref))
    return float(pts[idx, 0])


def _validate_poly(c, side):
    if c is None:
        return None
    x_bot = np.polyval(c, Y_EVAL)
    x_top = np.polyval(c, int(IMG_H * 0.4))
    if not (0 <= x_bot <= IMG_W) or not (0 <= x_top <= IMG_W):
        return None
    if side == 'left'  and x_bot > CX * 1.4:
        return None
    if side == 'right' and x_bot < CX * 0.6:
        return None
    dy    = Y_EVAL - int(IMG_H * 0.4)
    angle = abs(np.degrees(np.arctan2(abs(x_bot - x_top), dy)))
    return c if angle <= MAX_SLOPE_DEG else None


def fit_poly(xs, ys, side):
    if xs is None or len(xs) < MIN_FIT_PTS:
        return None
    try:
        c   = np.polyfit(ys, xs, 2)
        res = float(np.mean((np.polyval(c, ys) - xs) ** 2))
        return _validate_poly(c, side) if res <= MAX_RESIDUAL else None
    except Exception:
        return None


def poly_x(c, y):
    return float(np.polyval(c, y)) if c is not None else None


def lanes_to_polys(lanes):
    left_c = right_c = lxs = lys = rxs = rys = None
    left_cands, right_cands = [], []
    for pts in lanes:
        if len(pts) < MIN_FIT_PTS:
            continue
        x_ev = _x_at(pts, Y_EVAL)
        xs, ys = pts[:, 0].astype(float), pts[:, 1].astype(float)
        (left_cands if x_ev < CX else right_cands).append(
            (abs(x_ev - CX), xs, ys))
    if left_cands:
        left_cands.sort(key=lambda t: t[0])
        _, lxs, lys = left_cands[0]
        left_c = fit_poly(lxs, lys, 'left')
    if right_cands:
        right_cands.sort(key=lambda t: t[0])
        _, rxs, rys = right_cands[0]
        right_c = fit_poly(rxs, rys, 'right')
    return left_c, right_c, lxs, lys, rxs, rys


def px_to_world(u, v):
    dv = float(v) - CY
    if dv <= 0:
        return None, None
    return (float(u) - CX) * CAM_H / dv, FOCAL_PX * CAM_H / dv


def curvature_m(c):
    if c is None:
        return None
    A, B, _ = c
    if abs(A) < 1e-10:
        return None
    R_px = (1.0 + (2*A*Y_EVAL + B)**2)**1.5 / abs(2*A)
    if R_px > MAX_CURV_PX:
        return None
    dv = Y_EVAL - CY
    return R_px * (FOCAL_PX * CAM_H / dv) / FOCAL_PX if dv > 0 else None


def curv_label(R_m):
    if R_m is None or R_m > 200: return 'duz yol'
    if R_m > 80:  return 'hafif viraj'
    if R_m > 30:  return 'orta viraj'
    return 'sert viraj'


def calc_yaw(lc, rc):
    c = lc if lc is not None else rc
    if c is None:
        return None
    xb = poly_x(c, Y_EVAL)
    xt = poly_x(c, max(int(IMG_H * 0.35), Y_EVAL - 350))
    if xb is None or xt is None:
        return None
    return float(np.degrees(np.arctan2(xb - xt, float(Y_EVAL - max(int(IMG_H*0.35), Y_EVAL-350)))))


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
        if np.any((ys_b >= y0) & (ys_b < y0 + band)):
            dolu += 1
        else:
            bos += 1
    total = dolu + bos
    if total == 0:
        return 'belirsiz'
    r = bos / total
    return 'surekli' if r < 0.25 else ('kesikli' if r < 0.60 else 'belirsiz')


# ── Cizim ─────────────────────────────────────────────────────────────────

def draw_da(frame, da_mask):
    ov = np.zeros_like(frame, dtype=np.uint8)
    ov[da_mask == 1] = (0, 180, 0)
    cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)


def draw_lanes(frame, lanes, confs, lc, rc):
    for i, pts in enumerate(lanes):
        x_ev  = _x_at(pts, Y_EVAL)
        color = (255, 100, 0) if x_ev < CX else (0, 100, 255)
        for j in range(len(pts) - 1):
            cv2.line(frame,
                     (int(pts[j,0]),   int(pts[j,1])),
                     (int(pts[j+1,0]), int(pts[j+1,1])),
                     color, 3, cv2.LINE_AA)
        if i < len(confs):
            mid = len(pts) // 2
            cv2.putText(frame, f"{confs[i]:.2f}",
                        (int(pts[mid,0])+4, int(pts[mid,1])-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    for c, color in [(lc, (0,230,255)), (rc, (255,230,0))]:
        if c is None:
            continue
        pts_p = [(int(poly_x(c, y)), y)
                 for y in range(int(IMG_H*0.4), IMG_H, 3)
                 if 0 <= poly_x(c, y) <= IMG_W]
        if len(pts_p) > 2:
            cv2.polylines(frame, [np.array(pts_p)], False, color, 2, cv2.LINE_AA)


def draw_center(frame, lane_cx):
    cv2.line(frame, (CX, 400), (CX, IMG_H-1), (255,255,255), 1, cv2.LINE_AA)
    if lane_cx is not None:
        lc = int(lane_cx)
        cv2.line(frame, (lc, 400), (lc, IMG_H-1), (0,220,220), 2, cv2.LINE_AA)
        cv2.line(frame, (CX, IMG_H-55), (lc, IMG_H-55), (255,255,0), 1, cv2.LINE_AA)


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
    cv2.rectangle(ov, (0,0), (pw, IMG_H), (0,0,0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    def put(txt, row, col=(220,220,220)):
        cv2.putText(frame, txt, (6, 20+row*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

    def f(v, unit=''):
        return '-' if v is None else (f"{v:.2f}{unit}" if isinstance(v, float) else f"{v}{unit}")

    cl = curv_label(m['cl_m'])
    cr = curv_label(m['cr_m'])

    put(f"FPS: {fps:.1f}  [CLRNet + YOLOPv2]",                           0, (0,220,100))
    put(f"T: {m['ts']:.2f}s",                                             1)
    put(f" 1. Sapma          : {f(m['lat_m'],'m')} / {f(m['lat_px'],'px')}", 2)
    put(f" 2. Merkez uzaklik : {f(m['lcd_m'],'m')} / {f(m['lcd_px'],'px')}", 3)
    put(f" 3. Yaw acisi      : {f(m['yaw'],'deg')}",                     4)
    put(f" 4. Sol egrilik    : {f(m['cl_m'],'m')}  [{cl}]",              5)
    put(f" 5. Sag egrilik    : {f(m['cr_m'],'m')}  [{cr}]",              6)
    put(f" 6. Egrilik degisim: {f(m['dc'],'m')}",                        7)
    put(f" 7. Serit genisligi: {f(m['lw_m'],'m')} / {f(m['lw_px'],'px')}", 8)
    put(f" 8. Sol teker uzak : {f(m['dl'],'m')}",                        9)
    put(f" 9. Sag teker uzak : {f(m['dr'],'m')}",                       10)
    put(f"10. Sol serit tipi : {m['lt_l']}",                             11)
    put(f"11. Sag serit tipi : {m['lt_r']}",                             12)
    put('-'*47,                                                           13, (70,70,70))
    put(f"12. Arac seridi    : {m['v_lane']}",                           14)
    put(f"13. Komsu sol/sag  : {m['adj_l']} / {m['adj_r']}",            15)
    put(f"14. Algilama mes.  : {f(m['dd'],'m')}",                       16)
    put(f"15. CLRNet serit   : {m['n_lanes']}  conf:{f(m['conf'],'')}", 17, (100,220,255))
    rel = m['rel']
    put(f"16. Guvenilirlik   : {rel}",
        18, (0,255,0) if rel == 'yuksek' else (0,0,255))
    put("Q=Cikis  S=Screenshot  R=Sifirla",
        19, (130,130,130))


# ── Ana dongu ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Modeller
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("UYARI: CUDA bulunamadi, CPU modunda calisiliyor. FPS dusuk olabilir.")
    half = device.type == 'cuda'
    yolo   = torch.jit.load(WEIGHTS).to(device)
    if half:
        yolo.half()
    yolo.eval()

    dummy = (torch.zeros(1,3,IMG_SZ,IMG_SZ).to(device).half()
             if half else torch.zeros(1,3,IMG_SZ,IMG_SZ).to(device))
    print(f"YoloPv2 isiniyor...")
    with torch.no_grad():
        for _ in range(WARMUP_N):
            yolo(dummy)
    if half:
        torch.cuda.synchronize()

    print("CLRNet yukleniyor...")
    clrnet = CLRNetPredictor(CLRNET, device=str(device))
    clrnet.predict(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8))
    print("Hazir! Kamera aciliyor...\n")

    # Kamera
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H_PX)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # latency azalt

    if not cap.isOpened():
        print(f"HATA: Kamera {CAM_ID} acilamadi!")
        return

    # Smootherlar
    sm = {k: Smoother() for k in [
        'lat_m','lat_px','lcd_m','lcd_px','yaw',
        'cl_m','cr_m','lw_m','lw_px','dl','dr','dd','conf'
    ]}

    prev_curv_m = None
    t_start     = time.time()
    fps_buf     = deque(maxlen=30)
    frame_idx   = 0

    cv2.namedWindow('Webcam Lane', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam Lane', IMG_W, IMG_H)

    print("Calisuyor... Q=Cikis  S=Screenshot  R=Sifirla")

    with torch.no_grad():
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Kamera okunamadi!")
                break

            # Goruntu boyutunu kontrol et ve resize et
            h_f, w_f = frame.shape[:2]
            if w_f != IMG_W or h_f != IMG_H:
                frame = cv2.resize(frame, (IMG_W, IMG_H))

            # ── YOLOPv2 → drivable area ──
            img = letterbox(frame, IMG_SZ, stride=32)[0]
            img = img[:,:,::-1].transpose(2,0,1)
            img = np.ascontiguousarray(img)
            img_t = torch.from_numpy(img).to(device)
            img_t = (img_t.half() if half else img_t.float()) / 255.0
            img_t = img_t.unsqueeze(0)
            [_,_], seg, _ = yolo(img_t)
            da_mask = driving_area_mask(seg)

            # ── CLRNet → seritler ──
            lanes, confs = clrnet.predict_with_conf(frame)
            max_conf = max(confs) if confs else 0.0

            lc, rc, lxs, lys, rxs, rys = lanes_to_polys(lanes)

            lx_px = poly_x(lc, Y_EVAL)
            rx_px = poly_x(rc, Y_EVAL)

            # Genislik kontrolu
            if lx_px is not None and rx_px is not None:
                w_check = rx_px - lx_px
                if not (MIN_LANE_W_PX <= w_check <= MAX_LANE_W_PX):
                    lc, lx_px = None, None

            lx_px = poly_x(lc, Y_EVAL)
            rx_px = poly_x(rc, Y_EVAL)

            lane_cx = ((lx_px + rx_px)/2.0 if lx_px and rx_px else
                       lx_px + 200.0        if lx_px is not None else
                       rx_px - 200.0        if rx_px is not None else None)

            # ── Metrikler ──
            dv    = Y_EVAL - CY
            off_px = float(CX - lane_cx) if lane_cx is not None else None
            off_m  = off_px * CAM_H / dv if (off_px is not None and dv > 0) else None

            lat_px = sm['lat_px'].update(off_px)
            lat_m  = sm['lat_m'].update(off_m)
            lcd_px = sm['lcd_px'].update(abs(off_px) if off_px is not None else None)
            lcd_m  = sm['lcd_m'].update(abs(off_m)  if off_m  is not None else None)
            yaw_s  = sm['yaw'].update(calc_yaw(lc, rc))
            cl_m   = sm['cl_m'].update(curvature_m(lc))
            cr_m   = sm['cr_m'].update(curvature_m(rc))

            avg_c   = ((cl_m+cr_m)/2.0 if (cl_m and cr_m) else cl_m or cr_m)
            curv_chg = None
            if avg_c is not None and prev_curv_m is not None:
                dc = avg_c - prev_curv_m
                curv_chg = dc if abs(dc) < 50 else None
            prev_curv_m = avg_c

            w_px = (rx_px - lx_px) if (lx_px and rx_px) else None
            w_m  = (w_px * CAM_H / dv) if (w_px and w_px > 0 and dv > 0) else None
            lw_px = sm['lw_px'].update(w_px)
            lw_m  = sm['lw_m'].update(w_m)

            spm = dv / CAM_H if dv > 0 else None
            dl_m = dr_m = None
            if spm:
                if lx_px is not None:
                    dl_m = (CX - HALF_FRONT*spm - lx_px) / spm
                if rx_px is not None:
                    dr_m = (rx_px - (CX + HALF_FRONT*spm)) / spm
            dl_s = sm['dl'].update(dl_m)
            dr_s = sm['dr'].update(dr_m)

            lt_l = lane_type(lxs, lys)
            lt_r = lane_type(rxs, rys)

            dd_m = None
            for pts in lanes:
                if len(pts) > 0:
                    _, z = px_to_world(CX, int(pts[:,1].min()))
                    if z is not None:
                        dd_m = max(dd_m or 0.0, z)
            dd_s   = sm['dd'].update(dd_m)
            conf_s = sm['conf'].update(max_conf)

            v_lane = ('sol serit' if lat_m is not None and lat_m < -LANE_THRESH else
                      'sag serit' if lat_m is not None and lat_m >  LANE_THRESH else
                      'merkez'    if lat_m is not None else 'belirsiz')

            has_fl = any(_x_at(p, Y_EVAL) < CX*0.30 for p in lanes)
            has_fr = any(_x_at(p, Y_EVAL) > IMG_W - CX*0.30 for p in lanes)

            rel = ('dusuk' if (lc is None and rc is None)
                   or (lat_m is not None and abs(lat_m) > 0.9)
                   else 'yuksek')

            metrics = dict(
                ts=time.time()-t_start, lat_m=lat_m, lat_px=lat_px,
                lcd_m=lcd_m, lcd_px=lcd_px, yaw=yaw_s,
                cl_m=cl_m, cr_m=cr_m, dc=curv_chg,
                lw_m=lw_m, lw_px=lw_px, dl=dl_s, dr=dr_s,
                lt_l=lt_l, lt_r=lt_r, v_lane=v_lane,
                adj_l='var' if has_fl else 'yok',
                adj_r='var' if has_fr else 'yok',
                dd=dd_s, n_lanes=len(lanes), conf=conf_s, rel=rel,
            )

            # ── Gorsel ──
            draw_da(frame, da_mask)
            draw_lanes(frame, lanes, confs, lc, rc)
            draw_center(frame, lane_cx)

            fps_buf.append(1.0 / (time.time()-t0+1e-9))
            fps_live = float(np.mean(fps_buf))
            draw_panel(frame, metrics, fps_live)
            draw_arrow(frame, lat_m)

            cv2.imshow('Webcam Lane', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = str(OUT_DIR / f'screenshot_{frame_idx:05d}.png')
                cv2.imwrite(fname, frame)
                print(f"Screenshot: {fname}")
            elif key == ord('r'):
                for s in sm.values():
                    s.reset()
                prev_curv_m = None
                print("Smoother sifirlandi.")

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTamamlandi. {frame_idx} frame islendi.")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
ros2_lane_node.py — ROS2 Lane Detection Node (YOLOPv2 + CLRNet)
================================================================

SUBSCRIBE
  /zed2/zed_node/left/image_rect_color  [sensor_msgs/Image]
      ZED2 sol kamera ham görüntüsü (BGR, rect, 1280x720 beklenir)

PUBLISH
  /perception/lane/state  [lane_msgs/LaneState  VEYA  std_msgs/String JSON]
      Şerit durum mesajı — alanlar:
        float32  lateral_offset_m   : araç merkezinden sağ (+) veya sol (–) kayma (m)
        float32  yaw_error_deg      : araç ile şerit yönü arasındaki açı (derece)
        float32  lane_width_m       : sol–sağ şerit genişliği (m)
        string   left_lane_type     : 'kesik' | 'dolu' | 'belirsiz'
        string   right_lane_type    : 'kesik' | 'dolu' | 'belirsiz'
        float32  confidence         : CLRNet ortalama güven skoru [0,1]
        builtin_interfaces/Time stamp

ÇALIŞMA MODU
  ROS2 + cv_bridge mevcut  → Node olarak çalışır, kameradan subscribe edilir
  ROS2 mevcut değil        → Webcam modunda (cv2.VideoCapture) çalışır
  GPU mevcut değil         → Otomatik CPU fallback, uyarı mesajıyla

BAĞIMLILIKLAR
  Python  : rclpy, cv_bridge, sensor_msgs  (ROS2 Humble/Iron)
  Mesaj   : lane_msgs/LaneState paketi  (yoksa std_msgs/String JSON fallback)
  Model   : data/weights/yolopv2.pt, data/weights/culane_r18.pth
  Utils   : utils/utils.py, utils/clrnet_infer.py
"""

# ── Standart kütüphaneler ─────────────────────────────────────────────────────
import sys
import time
import json
import traceback
from pathlib import Path
from collections import deque

# ── Bilimsel / görüntü ────────────────────────────────────────────────────────
import numpy as np
import cv2
import torch

# ── Proje kök dizinini path'e ekle ───────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.utils import (
    driving_area_mask,
    lane_line_mask,
    letterbox,
    select_device,
)
from utils.clrnet_infer import CLRNetPredictor

# ── ROS2 import denemesi ──────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("[UYARI] rclpy veya cv_bridge bulunamadi — webcam modu aktif.")

# ── Özel mesaj tipi denemesi; yoksa JSON string fallback ─────────────────────
HAS_LANE_MSG = False
if HAS_ROS2:
    try:
        from lane_msgs.msg import LaneState
        from builtin_interfaces.msg import Time as RosTime
        HAS_LANE_MSG = True
    except ImportError:
        from std_msgs.msg import String as _StrMsg
        print("[UYARI] lane_msgs/LaneState bulunamadi — std_msgs/String JSON ile yayinlanacak.")

# ── Kamera / model sabitleri ──────────────────────────────────────────────────
YOLO_WEIGHTS   = str(ROOT / 'data/weights/yolopv2.pt')
CLRNET_WEIGHTS = str(ROOT / 'data/weights/culane_r18.pth')
IMG_SZ         = 640      # YOLOPv2 giriş boyutu
WARMUP_FRAMES  = 10       # ısınma inferansı sayısı

# ZED2 / Bee1 araç kamera sabitleri
FRAME_W    = 1280
FRAME_H    = 720
FOCAL_PX   = 700.0 * FRAME_W / 1920.0   # focal length (piksel)
CX         = FRAME_W // 2
CY         = FRAME_H // 2
Y_EVAL     = int(FRAME_H * 0.85)        # metrik ölçüm satırı
CAM_HEIGHT = 0.685                       # kamera yerden yüksekliği (m)

# Lane fit / smoothing ayarları
SMOOTH_N      = 6
MIN_FIT_PTS   = 4
MAX_RESIDUAL  = 150.0
MAX_SLOPE_DEG = 65.0
MIN_LANE_W_PX = 180
MAX_LANE_W_PX = 950

# ── Yardımcı sınıf: üstel ortalama ───────────────────────────────────────────

class Smoother:
    def __init__(self, n=SMOOTH_N):
        self.buf = deque(maxlen=n)

    def update(self, val):
        if val is not None and np.isfinite(float(val)):
            self.buf.append(float(val))
        return float(np.mean(self.buf)) if self.buf else None

    def reset(self):
        self.buf.clear()


# ── Metrik hesaplama yardımcıları ─────────────────────────────────────────────

def _x_at(pts, y_ref):
    idx = np.argmin(np.abs(pts[:, 1] - y_ref))
    return float(pts[idx, 0])


def _validate_poly(c, side):
    if c is None:
        return None
    x_bot = np.polyval(c, Y_EVAL)
    x_top = np.polyval(c, int(FRAME_H * 0.40))
    if not (0 <= x_bot <= FRAME_W) or not (0 <= x_top <= FRAME_W):
        return None
    if side == 'left'  and x_bot > CX * 1.4:
        return None
    if side == 'right' and x_bot < CX * 0.6:
        return None
    dy    = Y_EVAL - int(FRAME_H * 0.40)
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


def lanes_to_polys(lanes):
    left_c = right_c = lxs = lys = rxs = rys = None
    left_cands, right_cands = [], []
    for pts in lanes:
        if len(pts) < MIN_FIT_PTS:
            continue
        x_ev = _x_at(pts, Y_EVAL)
        xs = pts[:, 0].astype(float)
        ys = pts[:, 1].astype(float)
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


def calc_yaw_deg(lc, rc):
    c = lc if lc is not None else rc
    if c is None:
        return 0.0
    y_top = max(int(FRAME_H * 0.35), Y_EVAL - 350)
    xb = float(np.polyval(c, Y_EVAL))
    xt = float(np.polyval(c, y_top))
    return float(np.degrees(np.arctan2(xb - xt, float(Y_EVAL - y_top))))


def lane_width_m(lc, rc):
    """Sol ve sağ polinomdan fiziksel şerit genişliği (m)."""
    if lc is None or rc is None:
        return None
    xl = float(np.polyval(lc, Y_EVAL))
    xr = float(np.polyval(rc, Y_EVAL))
    w_px = xr - xl
    if not (MIN_LANE_W_PX <= w_px <= MAX_LANE_W_PX):
        return None
    dv = Y_EVAL - CY
    if dv <= 0:
        return None
    return w_px * (FOCAL_PX * CAM_HEIGHT / dv) / FOCAL_PX


def lateral_offset_m(lc, rc):
    """Araç merkezinden şerit merkezine yanal sapma (m, sağ = pozitif)."""
    xl = float(np.polyval(lc, Y_EVAL)) if lc is not None else None
    xr = float(np.polyval(rc, Y_EVAL)) if rc is not None else None
    if xl is not None and xr is not None:
        lane_cx = (xl + xr) / 2.0
    elif xl is not None:
        lane_cx = xl + MIN_LANE_W_PX / 2.0
    elif xr is not None:
        lane_cx = xr - MIN_LANE_W_PX / 2.0
    else:
        return None
    offset_px = CX - lane_cx       # sola kayma pozitif → ters çevir
    dv = Y_EVAL - CY
    if dv <= 0:
        return None
    return -offset_px * (FOCAL_PX * CAM_HEIGHT / dv) / FOCAL_PX


def classify_lane_type(xs, ys):
    """Nokta yoğunluğuna göre 'kesik' / 'dolu' / 'belirsiz'."""
    if xs is None or len(xs) < 8:
        return 'belirsiz'
    ys_b = ys[ys >= int(FRAME_H * 0.40)]
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
    ratio = dolu / total
    if ratio >= 0.75:
        return 'dolu'
    if ratio <= 0.40:
        return 'kesik'
    return 'belirsiz'


# ── YOLOPv2 çıkarım sarmalayıcı ──────────────────────────────────────────────

class YOLOPv2Inference:
    """YOLOPv2 TorchScript modelini yükler ve drivable area maskesi üretir."""

    def __init__(self, weights: str, img_sz: int = 640):
        if not torch.cuda.is_available():
            print("[UYARI] CUDA bulunamadi — YOLOPv2 CPU modunda calisacak (yavash).")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.half   = self.device.type != 'cpu'
        self.img_sz = img_sz

        print(f"[YOLOPv2] Model yukleniyor: {weights}")
        self.model = torch.jit.load(weights, map_location=self.device)
        if self.half:
            self.model.half()
        self.model.eval()

        # Isınma
        dummy = torch.zeros(1, 3, img_sz, img_sz, device=self.device)
        if self.half:
            dummy = dummy.half()
        with torch.no_grad():
            for _ in range(WARMUP_FRAMES):
                self.model(dummy)
        print("[YOLOPv2] Hazir.")

    def infer(self, bgr_frame: np.ndarray):
        """
        Returns:
            da_mask : (H, W) uint8  — sürülebilir alan maskesi (0/1)
            ll_mask : (H, W) uint8  — şerit çizgisi maskesi   (0/1)
        """
        img, _, _ = letterbox(bgr_frame, self.img_sz, stride=32, auto=True)
        img = img.transpose(2, 0, 1)                   # HWC → CHW
        img = np.ascontiguousarray(img)
        t   = torch.from_numpy(img).to(self.device)
        t   = t.half() if self.half else t.float()
        t  /= 255.0
        if t.ndimension() == 3:
            t = t.unsqueeze(0)

        with torch.no_grad():
            [_, _], seg, ll = self.model(t)

        return driving_area_mask(seg), lane_line_mask(ll)


# ── Ana işlem fonksiyonu: frame → metrikleri hesapla ─────────────────────────

def compute_lane_metrics(bgr_frame, yolo_infer, clrnet_pred, smoothers):
    """
    Tek bir frame için tüm şerit metriklerini hesaplar.

    Returns dict:
        lateral_offset_m, yaw_error_deg, lane_width_m,
        left_lane_type, right_lane_type, confidence
    """
    # YOLOPv2 — drivable area maskesi (şu an görselleştirme / gelecek kullanım)
    try:
        da_mask, ll_mask = yolo_infer.infer(bgr_frame)
    except Exception as e:
        print(f"[YOLOPv2] Inference hatasi: {e}")
        da_mask = ll_mask = None

    # CLRNet — şerit noktaları + güven
    try:
        lanes, confs = clrnet_pred.predict_with_conf(bgr_frame)
    except Exception as e:
        print(f"[CLRNet] Inference hatasi: {e}")
        lanes, confs = [], []

    # Polinom fit
    lc, rc, lxs, lys, rxs, rys = lanes_to_polys(lanes)

    # Metrikler
    lat   = lateral_offset_m(lc, rc)
    yaw   = calc_yaw_deg(lc, rc)
    lw    = lane_width_m(lc, rc)
    conf  = float(np.mean(confs)) if confs else 0.0
    lt_l  = classify_lane_type(lxs, lys) if lxs is not None else 'belirsiz'
    lt_r  = classify_lane_type(rxs, rys) if rxs is not None else 'belirsiz'

    # Yumuşatma
    lat  = smoothers['lat'].update(lat)
    yaw  = smoothers['yaw'].update(yaw)
    lw   = smoothers['lw'].update(lw)
    conf = smoothers['conf'].update(conf)

    return {
        'lateral_offset_m': lat  if lat  is not None else 0.0,
        'yaw_error_deg':    yaw  if yaw  is not None else 0.0,
        'lane_width_m':     lw   if lw   is not None else 0.0,
        'left_lane_type':   lt_l,
        'right_lane_type':  lt_r,
        'confidence':       conf if conf is not None else 0.0,
    }


# ── ROS2 NODE ─────────────────────────────────────────────────────────────────

if HAS_ROS2:
    class LaneDetectionNode(Node):
        """
        ROS2 düğümü: ZED2 görüntüsünü alır, YOLOPv2 + CLRNet çalıştırır,
        şerit durumunu /perception/lane/state üzerinde yayınlar.
        """

        def __init__(self):
            super().__init__('lane_detection_node')
            self.get_logger().info('lane_detection_node baslatiliyor...')

            self.bridge = CvBridge()

            # Model yükleme
            try:
                self.yolo = YOLOPv2Inference(YOLO_WEIGHTS, IMG_SZ)
            except Exception as e:
                self.get_logger().error(f'YOLOPv2 yuklenemedi: {e}')
                raise

            try:
                clrnet_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.clrnet = CLRNetPredictor(CLRNET_WEIGHTS, device=clrnet_device)
                self.get_logger().info('[CLRNet] Hazir.')
            except Exception as e:
                self.get_logger().error(f'CLRNet yuklenemedi: {e}')
                raise

            # Smoother'lar
            self.sm = {k: Smoother() for k in ['lat', 'yaw', 'lw', 'conf']}

            # Subscriber
            self.sub = self.create_subscription(
                Image,
                '/zed2/zed_node/left/image_rect_color',
                self._image_callback,
                10,
            )

            # Publisher — özel mesaj mı yoksa JSON string mi?
            if HAS_LANE_MSG:
                from lane_msgs.msg import LaneState
                self.pub = self.create_publisher(LaneState, '/perception/lane/state', 10)
                self._publish = self._publish_lane_msg
                self.get_logger().info('Yayin: lane_msgs/LaneState → /perception/lane/state')
            else:
                from std_msgs.msg import String
                self.pub = self.create_publisher(String, '/perception/lane/state', 10)
                self._publish = self._publish_json
                self.get_logger().warn(
                    'lane_msgs paketi bulunamadi — std_msgs/String JSON yayinlanacak.')

            self.get_logger().info('lane_detection_node hazir. Goruntu bekleniyor...')

        # ── Callback ──────────────────────────────────────────────────────────

        def _image_callback(self, msg: Image):
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f'cv_bridge donusum hatasi: {e}')
                return

            try:
                metrics = compute_lane_metrics(bgr, self.yolo, self.clrnet, self.sm)
            except Exception as e:
                self.get_logger().error(f'Metrik hesaplama hatasi: {e}')
                traceback.print_exc()
                return

            self._publish(metrics, msg.header.stamp)

        # ── Yayın yardımcıları ────────────────────────────────────────────────

        def _publish_lane_msg(self, m, stamp):
            from lane_msgs.msg import LaneState
            out = LaneState()
            out.lateral_offset_m  = float(m['lateral_offset_m'])
            out.yaw_error_deg     = float(m['yaw_error_deg'])
            out.lane_width_m      = float(m['lane_width_m'])
            out.left_lane_type    = m['left_lane_type']
            out.right_lane_type   = m['right_lane_type']
            out.confidence        = float(m['confidence'])
            out.stamp             = stamp
            self.pub.publish(out)

        def _publish_json(self, m, stamp):
            from std_msgs.msg import String
            payload = {
                'lateral_offset_m': m['lateral_offset_m'],
                'yaw_error_deg':    m['yaw_error_deg'],
                'lane_width_m':     m['lane_width_m'],
                'left_lane_type':   m['left_lane_type'],
                'right_lane_type':  m['right_lane_type'],
                'confidence':       m['confidence'],
                'stamp_sec':        stamp.sec,
                'stamp_nanosec':    stamp.nanosec,
            }
            msg = String()
            msg.data = json.dumps(payload)
            self.pub.publish(msg)


# ── Standalone webcam modu ────────────────────────────────────────────────────

def run_webcam():
    """ROS2 yokken webcam üzerinden çalışan bağımsız mod."""
    print("[Standalone] Webcam modu basliyor...")

    if not torch.cuda.is_available():
        print("[UYARI] GPU bulunamadi — CPU modunda calisiliyor (dusuk FPS beklenir).")

    yolo   = YOLOPv2Inference(YOLO_WEIGHTS, IMG_SZ)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clrnet = CLRNetPredictor(CLRNET_WEIGHTS, device=device)
    sm     = {k: Smoother() for k in ['lat', 'yaw', 'lw', 'conf']}

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[HATA] Webcam acilamadi!")
        return

    print("[Standalone] Hazir. Q = cikis")
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        metrics = compute_lane_metrics(frame, yolo, clrnet, sm)

        # FPS
        now   = time.time()
        fps   = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now

        # HUD
        lat  = metrics['lateral_offset_m']
        yaw  = metrics['yaw_error_deg']
        lw   = metrics['lane_width_m']
        conf = metrics['confidence']
        lt_l = metrics['left_lane_type']
        lt_r = metrics['right_lane_type']

        def put(txt, row, color=(220, 220, 220)):
            cv2.putText(frame, txt, (8, 22 + row * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        put(f"FPS: {fps:.1f}", 0, (0, 220, 80))
        put(f"Sapma      : {lat:+.3f} m",  1)
        put(f"Yaw hatasi : {yaw:+.2f} deg", 2)
        put(f"Serit gen. : {lw:.2f} m",    3)
        put(f"Tip  sol/sag: {lt_l} / {lt_r}", 4)
        put(f"CLRNet conf : {conf:.2f}",   5)
        put("Q = Cikis", 6, (130, 130, 130))

        cv2.imshow('Lane Detection (Standalone)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Giriş noktası ─────────────────────────────────────────────────────────────

def main():
    if HAS_ROS2:
        rclpy.init()
        node = LaneDetectionNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_webcam()


if __name__ == '__main__':
    main()

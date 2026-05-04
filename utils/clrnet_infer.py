"""
CLRNet inference wrapper — no mmcv or compiled CUDA extensions required.

Injects mock modules for mmcv.cnn and clrnet.ops before importing CLRNet,
so the model code loads cleanly on Windows with stock PyTorch.
"""
import os
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn as nn


# ── ConvModule stub (replaces mmcv.cnn.ConvModule) ───────────────────────────

_UNSET = object()  # sentinel to distinguish "not passed" from "passed as None"


class ConvModule(nn.Module):
    """
    Minimal drop-in for mmcv.cnn.ConvModule.
    Submodules are named 'conv' and 'bn' to match mmcv checkpoint keys.
    act_cfg=_UNSET (default) → ReLU; act_cfg=None (explicit) → no activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=None,
                 conv_cfg=None, norm_cfg=None,
                 act_cfg=_UNSET, inplace=True, **kwargs):
        super().__init__()
        if act_cfg is _UNSET:
            act_cfg = {'type': 'ReLU'}   # mmcv default
        if bias is None:
            bias = (norm_cfg is None)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = None
        if norm_cfg is not None:
            if norm_cfg.get('type', 'BN') == 'BN':
                self.bn = nn.BatchNorm2d(out_channels)
        self.activate = None
        if act_cfg is not None:
            t = act_cfg.get('type', 'ReLU')
            if t == 'ReLU':
                self.activate = nn.ReLU(inplace=inplace)
            elif t == 'LeakyReLU':
                self.activate = nn.LeakyReLU(
                    act_cfg.get('negative_slope', 0.01), inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


# ── Lane NMS (pure Python, matches CLRNet's CUDA kernel logic) ───────────────

def lane_nms(predictions, scores, overlap, top_k):
    """
    Lane-level NMS replacing the compiled CUDA extension.

    predictions: (N, 5+72) — cols: [bg, fg, start_y, start_x, length*n_strips, x0..x71]
                              x coords already in pixel scale (0..img_w-1)
    scores:      (N,)  confidence scores
    overlap:     float threshold (pixels per strip)
    top_k:       int   max lanes to keep
    Returns (keep_tensor, num_keep, None) matching C++ API.
    """
    N_OFFSETS = 72
    N_STRIPS  = N_OFFSETS - 1

    order = torch.argsort(scores, descending=True)
    n = len(order)
    suppressed = [False] * n
    keep = []

    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(order[i].item())
        if len(keep) >= top_k:
            break
        a = predictions[order[i]]
        start_a = int(a[2].item() * N_STRIPS + 0.5)
        end_a   = int(start_a + a[4].item() - 1 + 0.5)

        for j in range(i + 1, n):
            if suppressed[j]:
                continue
            b = predictions[order[j]]
            start_b = int(b[2].item() * N_STRIPS + 0.5)
            end_b   = int(start_b + b[4].item() - 1 + 0.5)

            s = max(start_a, start_b)
            e = min(min(end_a, end_b), N_OFFSETS - 1)
            if e < s:
                continue

            dist = (a[5 + s:5 + e + 1] - b[5 + s:5 + e + 1]).abs().sum().item()
            if dist < overlap * (e - s + 1):
                suppressed[j] = True

    keep_t = torch.tensor(keep, dtype=torch.long,
                          device=predictions.device)
    return keep_t, len(keep), None


# ── Module injection (must happen before any clrnet import) ──────────────────

def _inject_mocks(clrnet_src_dir):
    # mmcv.cnn stub
    mmcv_cnn = types.ModuleType('mmcv.cnn')
    mmcv_cnn.ConvModule = ConvModule
    mmcv = types.ModuleType('mmcv')
    mmcv.cnn = mmcv_cnn
    def _jit_passthrough(coderize=False):
        def decorator(fn):
            return fn
        return decorator

    mmcv.jit = _jit_passthrough

    # mmcv.runner.auto_fp16 is a passthrough decorator
    def _auto_fp16(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) is False else (
            decorator(args[0]) if args else decorator)

    runner_mod = types.ModuleType('mmcv.runner')
    runner_mod.auto_fp16 = _auto_fp16
    sys.modules['mmcv.runner'] = runner_mod
    mmcv.runner = runner_mod

    for sub in ('parallel', 'image', 'utils', 'ops'):
        m = types.ModuleType(f'mmcv.{sub}')
        setattr(mmcv, sub, m)
        sys.modules[f'mmcv.{sub}'] = m
    sys.modules['mmcv']     = mmcv
    sys.modules['mmcv.cnn'] = mmcv_cnn

    # clrnet.ops stub (avoids compiling CUDA NMS extension)
    ops_mod = types.ModuleType('clrnet.ops')
    ops_mod.nms     = lane_nms
    ops_mod.__all__ = ['nms']
    sys.modules['clrnet.ops']      = ops_mod
    sys.modules['clrnet.ops.nms']  = types.ModuleType('clrnet.ops.nms')

    # Add source dir to path
    if clrnet_src_dir not in sys.path:
        sys.path.insert(0, clrnet_src_dir)


# ── Predictor ─────────────────────────────────────────────────────────────────

class CLRNetPredictor:
    """
    Runs culane_r18 CLRNet lane detection from a .pth checkpoint.

    Example::
        predictor = CLRNetPredictor('data/weights/culane_r18.pth')
        lanes = predictor.predict(bgr_frame)
        # lanes: list of np.ndarray (N, 2) — pixel [x, y] in original frame
    """

    # CULane training dimensions
    IMG_W      = 800
    IMG_H      = 320
    ORI_W      = 1640
    ORI_H      = 590
    CUT_HEIGHT = 270

    def __init__(self, weight_path: str, device: str = 'cuda'):
        src_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'clrnet_src'))
        _inject_mocks(src_dir)

        # Import CLRNet model code (registries populated as side-effect)
        import clrnet.models.backbones  # noqa
        import clrnet.models.necks      # noqa
        import clrnet.models.heads      # noqa
        from clrnet.models.registry import build_net

        # Config.fromfile uses NamedTemporaryFile which is broken on Windows.
        # Load the Python config directly with importlib instead.
        import importlib.util
        from clrnet.utils.config import Config, ConfigDict
        cfg_path = os.path.join(src_dir,
                                'configs/clrnet/clr_resnet18_culane.py')
        spec = importlib.util.spec_from_file_location('_clr_cfg', cfg_path)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        cfg_dict = {k: v for k, v in _mod.__dict__.items()
                    if not k.startswith('__')}
        cfg = Config(cfg_dict, filename=cfg_path)
        self.cfg = cfg

        self.device = torch.device(device
                                   if torch.cuda.is_available() else 'cpu')

        model = build_net(cfg)
        state = torch.load(weight_path, map_location='cpu')
        net_state = state['net']
        # Strip DataParallel 'module.' prefix if present
        if all(k.startswith('module.') for k in net_state):
            net_state = {k[len('module.'):]: v for k, v in net_state.items()}
        model.load_state_dict(net_state, strict=False)
        model.eval().to(self.device)
        self.model = model

    # ── pre/postprocess ──────────────────────────────────────────────────────

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(bgr, (self.ORI_W, self.ORI_H))
        img = img[self.CUT_HEIGHT:, :]           # crop top sky region
        img = cv2.resize(img, (self.IMG_W, self.IMG_H))
        img = img.astype(np.float32) / 255.0     # CULane normalization: /255
        img = img.transpose(2, 0, 1)             # HWC → CHW
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    def predict(self, bgr_frame: np.ndarray) -> list:
        """
        Args:
            bgr_frame: (H, W, 3) uint8 BGR image
        Returns:
            list of np.ndarray (N, 2) int32 — [x, y] pixel coords in bgr_frame space
        """
        lanes, _ = self._predict_internal(bgr_frame)
        return lanes

    def predict_with_conf(self, bgr_frame: np.ndarray):
        """
        Like predict() but also returns per-lane confidence scores.
        Returns:
            lanes: list of np.ndarray (N, 2) int32
            confs: list of float  (softmax fg confidence for each lane)
        """
        return self._predict_internal(bgr_frame)

    def _predict_internal(self, bgr_frame: np.ndarray):
        h_orig, w_orig = bgr_frame.shape[:2]
        tensor = self._preprocess(bgr_frame)

        with torch.no_grad():
            output = self.model(tensor)

        lanes_raw = self.model.heads.get_lanes(output)

        lanes, confs = [], []
        softmax = torch.nn.Softmax(dim=0)
        for lane in lanes_raw[0]:
            pts = lane.to_array(self.cfg)
            if len(pts) == 0:
                continue
            pts[:, 0] = pts[:, 0] / self.ORI_W * w_orig
            pts[:, 1] = pts[:, 1] / self.ORI_H * h_orig
            lanes.append(pts.astype(np.int32))
            # metadata['conf'] is the raw fg logit; convert via softmax pair
            raw_conf = float(lane.metadata.get('conf', 0.0))
            confs.append(float(torch.sigmoid(torch.tensor(raw_conf)).item()))
        return lanes, confs

    def draw(self, bgr_frame: np.ndarray,
             lanes: list,
             color: tuple = (0, 255, 0),
             thickness: int = 3) -> np.ndarray:
        """Overlay detected lanes onto a copy of the frame."""
        out = bgr_frame.copy()
        for pts in lanes:
            for i in range(len(pts) - 1):
                cv2.line(out,
                         tuple(pts[i]),
                         tuple(pts[i + 1]),
                         color, thickness)
        return out

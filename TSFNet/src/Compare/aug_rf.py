from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    import albumentations as A  # type: ignore
except Exception:
    A = None

# Boxes are COCO format [x, y, w, h] in pixel coords (float)
Box = List[float]


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _boxes_clip_and_filter(boxes: List[Box], W: int, H: int, min_size: int = 2) -> List[Box]:
    out: List[Box] = []
    for x, y, w, h in boxes:
        x = float(np.clip(x, 0, W - 1))
        y = float(np.clip(y, 0, H - 1))
        w = float(max(0.0, min(w, W - x)))
        h = float(max(0.0, min(h, H - y)))
        if w >= min_size and h >= min_size:
            out.append([x, y, w, h])
    return out


def _boxes_translate(boxes: List[Box], dx: float, dy: float) -> List[Box]:
    return [[x + dx, y + dy, w, h] for x, y, w, h in boxes]


def _boxes_scale(boxes: List[Box], sx: float, sy: float) -> List[Box]:
    return [[x * sx, y * sy, w * sx, h * sy] for x, y, w, h in boxes]


def _boxes_shift_after_pad_crop(boxes: List[Box], pad_left: float, pad_top: float) -> List[Box]:
    return [[x + pad_left, y + pad_top, w, h] for x, y, w, h in boxes]


@dataclass
class RFAugConfig:
    img_size: int = 512
    # probabilities（强度分支按随机分流控制）
    p_awgn: float = 0.6
    p_colored: float = 0.3
    p_narrowband: float = 0.3
    p_chirp: float = 0.2
    p_gamma: float = 0.5
    p_clahe: float = 0.3
    p_time_freq_mask: float = 0.4
    # 几何（老实现的平移/缩放兜底）
    p_translate: float = 0.5
    p_scale: float = 0.3
    # 新增：强度分流与轻度去噪
    p_denoise: float = 0.2  # 训练中低概率去噪
    # Albumentations 几何
    use_albu: bool = True
    rotate_limit: float = 5.0  # 轻微旋转上限（度）
    # parameters
    snr_db_range: Tuple[float, float] = (0.0, 15.0)
    sinr_db_range: Tuple[float, float] = (0.0, 10.0)
    colored_beta: float = 1.0  # 1/f^beta
    nb_lines_range: Tuple[int, int] = (1, 4)
    nb_line_width: Tuple[int, int] = (1, 4)
    chirp_lines: Tuple[int, int] = (0, 2)
    gamma_range: Tuple[float, float] = (0.8, 1.2)
    time_mask_max: float = 0.10  # fraction of width
    freq_mask_max: float = 0.08  # fraction of height
    masks_per_dim: Tuple[int, int] = (0, 2)
    # geometric（老实现兜底）
    translate_x_frac: float = 0.05
    translate_y_frac: float = 0.03
    scale_range: Tuple[float, float] = (0.9, 1.1)


class RFAugmentor:
    def __init__(self, cfg: RFAugConfig, *, interference_pool: Optional[List[np.ndarray]] = None):
        self.cfg = cfg
        self.interference_pool: List[np.ndarray] = interference_pool or []

    def set_interference_pool(self, pool: List[np.ndarray]):
        self.interference_pool = pool or []

    # ---------- Non-geometric ----------
    def _add_awgn(self, imgf: np.ndarray) -> np.ndarray:
        # imgf in [0,1]
        snr_db = random.uniform(*self.cfg.snr_db_range)
        sig_power = np.mean(imgf ** 2) + 1e-8
        snr = 10 ** (snr_db / 10.0)
        noise_power = sig_power / snr
        noise = np.random.normal(0.0, math.sqrt(noise_power), imgf.shape).astype(np.float32)
        return _clip01(imgf + noise)

    def _add_colored_noise(self, imgf: np.ndarray) -> np.ndarray:
        # simple 1/f^beta along frequency axis
        H, W = imgf.shape[:2]
        beta = self.cfg.colored_beta
        f = np.fft.rfftfreq(H)
        w = 1.0 / (np.maximum(f, 1e-4) ** beta)
        w = w / w.max()
        noise = np.random.normal(0.0, 1.0, (H, W)).astype(np.float32)
        Nf = np.fft.rfft(noise, axis=0)
        Nf *= w[:, None]
        colored = np.fft.irfft(Nf, n=H, axis=0).astype(np.float32)
        colored = (colored - colored.min()) / (colored.ptp() + 1e-8)
        alpha = random.uniform(0.1, 0.4)
        return _clip01((1 - alpha) * imgf + alpha * colored)

    def _add_narrowband(self, imgf: np.ndarray) -> np.ndarray:
        H, W = imgf.shape[:2]
        n = random.randint(*self.cfg.nb_lines_range)
        out = imgf.copy()
        for _ in range(n):
            vertical = bool(random.getrandbits(1))
            width = random.randint(*self.cfg.nb_line_width)
            amp = random.uniform(0.3, 0.8)
            if vertical:
                x = random.randint(0, W - 1)
                x1 = max(0, x - width // 2)
                x2 = min(W, x1 + width)
                out[:, x1:x2] = _clip01(out[:, x1:x2] * (1 - amp) + amp)
            else:
                y = random.randint(0, H - 1)
                y1 = max(0, y - width // 2)
                y2 = min(H, y1 + width)
                out[y1:y2, :] = _clip01(out[y1:y2, :] * (1 - amp) + amp)
        return out

    def _add_chirp(self, imgf: np.ndarray) -> np.ndarray:
        H, W = imgf.shape[:2]
        k = random.randint(*self.cfg.chirp_lines)
        out = imgf.copy()
        for _ in range(k):
            x0 = random.randint(0, W - 1)
            y0 = random.randint(0, H - 1)
            x1 = random.randint(0, W - 1)
            y1 = random.randint(0, H - 1)
            thickness = random.randint(1, 3)
            amp = random.uniform(0.3, 0.7)
            cv2.line(out, (x0, y0), (x1, y1), float(amp), thickness=thickness)
        return _clip01(out)

    def _gamma(self, imgf: np.ndarray) -> np.ndarray:
        g = random.uniform(*self.cfg.gamma_range)
        return _clip01(imgf ** g)

    def _clahe(self, imgf: np.ndarray) -> np.ndarray:
        gray8 = np.clip(imgf * 255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray8).astype(np.float32) / 255.0
        return eq

    def _time_freq_mask(self, imgf: np.ndarray) -> np.ndarray:
        H, W = imgf.shape[:2]
        out = imgf.copy()
        n_time = random.randint(*self.cfg.masks_per_dim)
        n_freq = random.randint(*self.cfg.masks_per_dim)
        for _ in range(n_time):
            w = int(W * random.uniform(0.0, self.cfg.time_mask_max))
            if w <= 0:
                continue
            x = random.randint(0, max(0, W - w))
            out[:, x:x + w] *= random.uniform(0.0, 0.5)
        for _ in range(n_freq):
            h = int(H * random.uniform(0.0, self.cfg.freq_mask_max))
            if h <= 0:
                continue
            y = random.randint(0, max(0, H - h))
            out[y:y + h, :] *= random.uniform(0.0, 0.5)
        return out

    # 真实干扰混合（可选）
    def _mix_interference(self, imgf: np.ndarray) -> np.ndarray:
        if not self.interference_pool:
            return imgf
        H, W = imgf.shape[:2]
        interference = random.choice(self.interference_pool)
        if interference.shape != imgf.shape:
            interference = cv2.resize(interference.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        # 随机频率轴平移
        shift_ratio = random.uniform(-0.4, 0.4)
        shift_pix = int(round(shift_ratio * H))
        interference = np.roll(interference, shift_pix, axis=0)
        # 线性功率域混合（按 SINR）
        P_sig = float(np.mean(imgf ** 2) + 1e-8)
        P_int = float(np.mean(interference ** 2) + 1e-8)
        sinr = random.uniform(*self.cfg.sinr_db_range)
        alpha = P_sig / (P_int * (10 ** (sinr / 10.0)))
        mixed = imgf + alpha * interference
        # 轻噪重标定
        P_noise = P_sig / (10 ** ((sinr + 5.0) / 10.0))
        noise = np.random.normal(0.0, math.sqrt(P_noise), imgf.shape).astype(np.float32)
        mixed = mixed + noise
        # 归一化
        mmin, mmax = float(mixed.min()), float(mixed.max())
        if mmax - mmin > 1e-8:
            mixed = (mixed - mmin) / (mmax - mmin)
        else:
            mixed = np.clip(mixed, 0.0, 1.0)
        return mixed.astype(np.float32)

    # 轻度去噪（median/bilateral 随机）
    def _denoise_light(self, imgf: np.ndarray) -> np.ndarray:
        img8 = np.clip(imgf * 255.0, 0, 255).astype(np.uint8)
        if bool(random.getrandbits(1)):
            out8 = cv2.medianBlur(img8, 3)
        else:
            out8 = cv2.bilateralFilter(img8, d=5, sigmaColor=15, sigmaSpace=7)
        return out8.astype(np.float32) / 255.0

    # ---------- Geometric (with boxes) ----------
    def _translate(self, img: np.ndarray, boxes: List[Box]) -> Tuple[np.ndarray, List[Box]]:
        H, W = img.shape[:2]
        dx = int(round(W * random.uniform(-self.cfg.translate_x_frac, self.cfg.translate_x_frac)))
        dy = int(round(H * random.uniform(-self.cfg.translate_y_frac, self.cfg.translate_y_frac)))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img2 = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        boxes2 = _boxes_translate(boxes, dx, dy)
        boxes2 = _boxes_clip_and_filter(boxes2, W, H)
        return img2, boxes2

    def _scale(self, img: np.ndarray, boxes: List[Box]) -> Tuple[np.ndarray, List[Box]]:
        H, W = img.shape[:2]
        s = random.uniform(*self.cfg.scale_range)
        newW, newH = max(1, int(round(W * s))), max(1, int(round(H * s)))
        img_s = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
        # pad/crop to original size centered
        canvas = np.zeros_like(img)
        x_off = (W - newW) // 2
        y_off = (H - newH) // 2
        x0 = max(0, x_off)
        y0 = max(0, y_off)
        x1 = min(W, x_off + newW)
        y1 = min(H, y_off + newH)
        sx0 = max(0, -x_off)
        sy0 = max(0, -y_off)
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)
        canvas[y0:y1, x0:x1] = img_s[sy0:sy1, sx0:sx1]
        img2 = canvas
        boxes2 = _boxes_scale(boxes, s, s)
        boxes2 = _boxes_shift_after_pad_crop(boxes2, x_off, y_off)
        boxes2 = _boxes_clip_and_filter(boxes2, W, H)
        return img2, boxes2

    # ---------- Public API ----------
    def __call__(self, img: np.ndarray, boxes: List[Box], *, apply_color_map: bool = False) -> Tuple[np.ndarray, List[Box]]:
        """Apply augmentations per spec.
        - img: uint8 HxW or HxWx3
        - boxes: list of [x,y,w,h] float (COCO)
        Returns augmented img (uint8 3ch if apply_color_map else uint8 gray) and boxes.
        """
        H, W = img.shape[:2] if img.ndim == 2 else (img.shape[0], img.shape[1])
        # work in gray [0,1]
        gray = _to_gray(img).astype(np.float32) / 255.0

        # 第一阶段：强度变换（随机分支）
        r = random.random()
        if r < 0.6:
            # 加噪/混合（优先真实干扰；否则 AWGN）
            if self.interference_pool and random.random() < 0.5:
                gray = self._mix_interference(gray)
            else:
                gray = self._add_awgn(gray)
        elif r < 0.8:
            # 轻度去噪
            if random.random() < self.cfg.p_denoise:
                gray = self._denoise_light(gray)
        else:
            # 保持原样
            pass

        # 可选：附加的非几何扰动（弱化，不改变分支语义）
        if random.random() < self.cfg.p_gamma:
            gray = self._gamma(gray)
        if random.random() < self.cfg.p_clahe:
            gray = self._clahe(gray)
        if random.random() < self.cfg.p_time_freq_mask:
            gray = self._time_freq_mask(gray)

        # 第二阶段：几何变换（Albumentations 优先；无则使用平移/缩放兜底）
        out_gray_u8 = (gray * 255.0).astype(np.uint8)
        new_boxes = boxes[:] if boxes else []
        if self.cfg.use_albu and A is not None:
            # 将 scale_range -> scale_limit（相对 1 的偏差）
            smin, smax = self.cfg.scale_range
            scale_limit = max(abs(smin - 1.0), abs(smax - 1.0))
            T = A.Compose([
                A.ShiftScaleRotate(
                    shift_limit_x=self.cfg.translate_x_frac,
                    shift_limit_y=self.cfg.translate_y_frac,
                    scale_limit=scale_limit,
                    rotate_limit=self.cfg.rotate_limit,
                    border_mode=cv2.BORDER_REFLECT_101,
                    interpolation=cv2.INTER_LINEAR,
                    p=0.9,
                ),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], min_visibility=0.0))
            labels = [1] * len(new_boxes)
            aug = T(image=out_gray_u8, bboxes=new_boxes, labels=labels)
            out_gray_u8 = aug['image']
            new_boxes = list(aug['bboxes'])
            new_boxes = _boxes_clip_and_filter(new_boxes, W, H)
        else:
            # 兜底：随机平移/缩放
            if new_boxes:
                if random.random() < self.cfg.p_translate:
                    out_gray_u8, new_boxes = self._translate(out_gray_u8, new_boxes)
                if random.random() < self.cfg.p_scale:
                    out_gray_u8, new_boxes = self._scale(out_gray_u8, new_boxes)
            else:
                if random.random() < self.cfg.p_translate:
                    out_gray_u8, _ = self._translate(out_gray_u8, [])
                if random.random() < self.cfg.p_scale:
                    out_gray_u8, _ = self._scale(out_gray_u8, [])

        # 输出
        if apply_color_map:
            out_img = cv2.applyColorMap(out_gray_u8, cv2.COLORMAP_JET)
        else:
            out_img = _ensure_3ch(out_gray_u8)
        return out_img, new_boxes

    # 抽取核心流水线：返回增强后的灰度 uint8 与 boxes
    def _pipeline_gray(self, img: np.ndarray, boxes: List[Box]) -> Tuple[np.ndarray, List[Box]]:
        H, W = img.shape[:2] if img.ndim == 2 else (img.shape[0], img.shape[1])
        gray = _to_gray(img).astype(np.float32) / 255.0
        # 强度分流
        r = random.random()
        if r < 0.6:
            if self.interference_pool and random.random() < 0.5:
                gray = self._mix_interference(gray)
            else:
                gray = self._add_awgn(gray)
        elif r < 0.8:
            if random.random() < self.cfg.p_denoise:
                gray = self._denoise_light(gray)
        # 细微强度扰动
        if random.random() < self.cfg.p_gamma:
            gray = self._gamma(gray)
        if random.random() < self.cfg.p_clahe:
            gray = self._clahe(gray)
        if random.random() < self.cfg.p_time_freq_mask:
            gray = self._time_freq_mask(gray)
        out_gray_u8 = (gray * 255.0).astype(np.uint8)
        new_boxes = boxes[:] if boxes else []
        # 几何：albumentations 优先
        if self.cfg.use_albu and A is not None:
            smin, smax = self.cfg.scale_range
            scale_limit = max(abs(smin - 1.0), abs(smax - 1.0))
            T = A.Compose([
                A.ShiftScaleRotate(
                    shift_limit_x=self.cfg.translate_x_frac,
                    shift_limit_y=self.cfg.translate_y_frac,
                    scale_limit=scale_limit,
                    rotate_limit=self.cfg.rotate_limit,
                    border_mode=cv2.BORDER_REFLECT_101,
                    interpolation=cv2.INTER_LINEAR,
                    p=0.9,
                ),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], min_visibility=0.0))
            labels = [1] * len(new_boxes)
            aug = T(image=out_gray_u8, bboxes=new_boxes, labels=labels)
            out_gray_u8 = aug['image']
            new_boxes = list(aug['bboxes'])
            new_boxes = _boxes_clip_and_filter(new_boxes, W, H)
        else:
            if new_boxes:
                if random.random() < self.cfg.p_translate:
                    out_gray_u8, new_boxes = self._translate(out_gray_u8, new_boxes)
                if random.random() < self.cfg.p_scale:
                    out_gray_u8, new_boxes = self._scale(out_gray_u8, new_boxes)
            else:
                if random.random() < self.cfg.p_translate:
                    out_gray_u8, _ = self._translate(out_gray_u8, [])
                if random.random() < self.cfg.p_scale:
                    out_gray_u8, _ = self._scale(out_gray_u8, [])
        return out_gray_u8, new_boxes

    # 新增：直接产出双流（I_tex, I_pos, boxes_aug）
    # tex_mode: 'colormap'（伪彩）或 'coord'（原始+坐标通道）
    def augment_to_streams(self, img: np.ndarray, boxes: List[Box], *, tex_mode: str = 'colormap') -> Tuple[np.ndarray, np.ndarray, List[Box]]:
        out_gray_u8, new_boxes = self._pipeline_gray(img, boxes)
        gray01 = out_gray_u8.astype(np.float32) / 255.0
        # 位置流：edge + corner + gray
        edge = _extract_edge_gray01(gray01)
        corner = _extract_corner_gray01(gray01)
        I_pos = np.stack([edge, corner, gray01], axis=0).astype(np.float32)  # (3,H,W)
        # 纹理流
        if tex_mode == 'coord':
            I_tex = _add_coord_channels_gray01(gray01)  # (3,H,W)
        else:
            cm = cv2.applyColorMap(out_gray_u8, cv2.COLORMAP_JET)
            I_tex = (cm.astype(np.float32) / 255.0).transpose(2, 0, 1).astype(np.float32)
        return I_tex, I_pos, new_boxes


# 新增：位置/纹理双流所需的特征构造
def _extract_edge_gray01(gray01: np.ndarray) -> np.ndarray:
    # gray01: float [0,1], HxW
    g = gray01.astype(np.float32, copy=False)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    mmin, mmax = float(mag.min()), float(mag.max())
    if mmax - mmin > 1e-8:
        mag = (mag - mmin) / (mmax - mmin)
    else:
        mag = np.zeros_like(mag, dtype=np.float32)
    return mag.astype(np.float32)


def _extract_corner_gray01(gray01: np.ndarray) -> np.ndarray:
    g = gray01.astype(np.float32, copy=False)
    H, W = g.shape
    small = cv2.resize(g, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    corner_s = cv2.cornerHarris(small, blockSize=2, ksize=3, k=0.04)
    corner_s = np.abs(corner_s, dtype=np.float32)
    corner = cv2.resize(corner_s, (W, H), interpolation=cv2.INTER_LINEAR)
    cmin, cmax = float(corner.min()), float(corner.max())
    if cmax - cmin > 1e-8:
        corner = (corner - cmin) / (cmax - cmin)
    else:
        corner = np.zeros_like(corner, dtype=np.float32)
    return corner.astype(np.float32)


def _add_coord_channels_gray01(gray01: np.ndarray) -> np.ndarray:
    H, W = gray01.shape
    t_coord = np.linspace(0, 1, W, dtype=np.float32).reshape(1, -1).repeat(H, axis=0)
    f_coord = np.linspace(0, 1, H, dtype=np.float32).reshape(-1, 1).repeat(W, axis=1)
    chw = np.stack([gray01, t_coord, f_coord], axis=0)
    return chw.astype(np.float32)

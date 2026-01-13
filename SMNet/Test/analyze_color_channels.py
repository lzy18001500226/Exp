"""测试现有谱图的灰度映射、伪彩组合与归一化策略。

运行示例：
    python analyze_color_channels.py --npz-root D:/Exp/SMNet/FCSData/1 --limit 4

脚本只读数据，会把统计信息打印到终端，并在当前目录输出对比可视化 PNG。
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

import cv2


def iter_npz(root: Path) -> Iterable[Path]:
    files = sorted(
        fp for fp in root.rglob("*.npz") if fp.is_file()
    )
    for fp in files:
        yield fp


def select_sample_paths(
    root: Path,
    limit: int,
    sample_mode: str,
    sample_stride: int,
    sample_per_class: int,
    random_seed: int | None,
) -> list[Path]:
    files = list(iter_npz(root))
    if not files:
        return []

    stride = max(1, sample_stride)
    per_class = max(0, sample_per_class)
    mode = sample_mode.lower()
    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    if per_class > 0:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for fp in files:
            grouped[fp.parent.name].append(fp)
        selected: list[Path] = []
        for cls in sorted(grouped):
            group = grouped[cls]
            if mode == "random":
                rng.shuffle(group)
            elif mode == "stride":
                group = group[::stride]
            selected.extend(group[:per_class])
        files = selected
        if mode == "random":
            rng.shuffle(files)
    else:
        if mode == "random":
            rng.shuffle(files)
        elif mode == "stride":
            files = files[::stride]

    if limit > 0:
        files = files[:limit]

    return files


def load_sample(fp: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(fp, allow_pickle=True) as bundle:
        available_keys = set(bundle.files)

        image: np.ndarray | None = None
        if "image" in available_keys:
            image = bundle["image"].astype(np.float32)

        spec: np.ndarray | None = None
        if "spec_512" in available_keys:
            spec = bundle["spec_512"].astype(np.float32)
        elif image is not None:
            spec = image.mean(axis=-1)
        elif "spec_raw" in available_keys:
            spec = np.asarray(bundle["spec_raw"], dtype=np.float32)

        if spec is None:
            raise KeyError("Sample missing both image and spec data")

        if spec.ndim == 3 and spec.shape[-1] == 1:
            spec = spec[..., 0]

        if image is None:
            # Fall back to a 3-channel visualization derived from the spectrum.
            spec_for_image = spec
            if spec_for_image.ndim == 2:
                spec_for_image = np.repeat(spec_for_image[..., None], 3, axis=-1)
            elif spec_for_image.ndim == 3 and spec_for_image.shape[-1] != 3:
                spec_for_image = np.repeat(spec_for_image[..., [0]], 3, axis=-1)
            image = spec_for_image.astype(np.float32)

        bboxes = bundle.get("bboxes")
        if bboxes is None:
            # Try to derive bounding boxes from available masks (FCS preferred, fallback to VTS).
            mask_source: np.ndarray | None = None
            for mask_key in ("fcs_mask_512", "vts_mask_512"):
                if mask_key not in available_keys:
                    continue
                candidate = np.asarray(bundle[mask_key])
                candidate = np.squeeze(candidate)
                if candidate.ndim == 3:
                    # Collapse channel dimension if present by taking the union.
                    candidate = candidate.max(axis=-1)
                if candidate.ndim != 2:
                    continue
                if np.count_nonzero(candidate) == 0:
                    # Try next mask source when current one carries no signal.
                    continue
                mask_source = candidate.astype(np.float32)
                break

            derived_boxes: list[tuple[float, float, float, float, float]] = []
            if mask_source is not None:
                mask = np.squeeze(mask_source).astype(np.float32)
                if mask.ndim == 2:
                    mask_norm = mask - mask.min()
                    if mask_norm.max() > 1e-6:
                        mask_norm = mask_norm / mask_norm.max()
                        mask_bin = (mask_norm >= 0.5).astype(np.uint8)
                        if mask_bin.any():
                            # Prefer connected component stats over contour areas to handle thin masks.
                            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
                            for label_idx in range(1, num_labels):
                                x, y, w, h, area = stats[label_idx]
                                if area < 4:
                                    continue
                                if w <= 0 or h <= 0:
                                    continue
                                derived_boxes.append((float(x), float(y), float(x + w), float(y + h), 1.0))
                            if not derived_boxes:
                                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for cnt in contours:
                                    x, y, w, h = cv2.boundingRect(cnt)
                                    if w <= 0 or h <= 0:
                                        continue
                                    derived_boxes.append((float(x), float(y), float(x + w), float(y + h), 1.0))
            if derived_boxes:
                bboxes = np.asarray(derived_boxes, dtype=np.float32)
            else:
                bboxes = np.zeros((0, 5), dtype=np.float32)
        else:
            bboxes = bboxes.astype(np.float32)
    return image, spec, bboxes


def percentile_normalize(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip by percentile and re-scale to [0, 1]."""
    clipped = np.asarray(arr, dtype=np.float32)
    lo = np.percentile(clipped, low)
    hi = np.percentile(clipped, high)
    if hi - lo < 1e-6:
        if clipped.max() > 1e-6:
            norm = clipped / clipped.max()
        else:
            norm = np.zeros_like(clipped, dtype=np.float32)
        return norm
    clipped = np.clip(clipped, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    return norm.astype(np.float32)


def _ensure_kernel(sz: int) -> int:
    sz = int(max(1, sz))
    if sz % 2 == 0:
        sz += 1
    return sz


EDGE_METHOD_CHOICES: tuple[str, ...] = (
    "sobel",
    "log-hp",
    "log-hp-raw",
    "laplacian",
    "window-std",
    "tophat",
    "canny",
    "gabor",
    "local-entropy",
)


TEXTURE_METHOD_CHOICES: tuple[str, ...] = (
    "none",
    "log-hp",
    "laplacian",
    "window-std",
    "tophat",
    "canny",
    "gabor",
    "local-entropy",
)


def _normalize_and_threshold(
    magnitude: np.ndarray,
    lambda_low: float,
    lambda_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = magnitude.astype(np.float32, copy=False)
    max_val = float(magnitude.max())
    if max_val <= 1e-6:
        zeros = np.zeros_like(magnitude, dtype=np.float32)
        return zeros, zeros

    lambda_low_val = lambda_low * max_val if lambda_low < 1.0 else lambda_low
    lambda_high_val = lambda_high * max_val if lambda_high < 1.0 else lambda_high
    lambda_low_val = max(0.0, lambda_low_val)
    lambda_high_val = max(lambda_low_val + 1e-6, lambda_high_val)

    edge_raw = (magnitude / max_val).astype(np.float32, copy=False)
    edge_clean = np.zeros_like(edge_raw, dtype=np.float32)
    mask = magnitude >= lambda_low_val
    if lambda_high_val > lambda_low_val:
        edge_clean[mask] = np.clip(
            (magnitude[mask] - lambda_low_val) / (lambda_high_val - lambda_low_val),
            0.0,
            1.0,
        )
    else:
        edge_clean[mask] = 1.0

    return edge_raw, edge_clean


def make_bbox_mask(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if boxes.size == 0:
        return mask
    for x1, y1, x2, y2, *_ in boxes:
        x_min = max(0, min(width, int(math.floor(x1))))
        x_max = max(0, min(width, int(math.ceil(x2))))
        y_min = max(0, min(height, int(math.floor(y1))))
        y_max = max(0, min(height, int(math.ceil(y2))))
        if x_max <= x_min or y_max <= y_min:
            continue
        mask[y_min:y_max, x_min:x_max] = True
    return mask


def _safe_stats(arr: np.ndarray) -> tuple[float, float, float, float]:
    if arr.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(arr.mean())
    std = float(arr.std())
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))
    return mean, std, p90, p99


def compute_feature_stats(
    arr: np.ndarray,
    bbox_mask: np.ndarray,
    activation_mask: np.ndarray | None = None,
) -> dict[str, float]:
    bbox_mask = bbox_mask.astype(bool, copy=False)
    total_mean, total_std, p90, p99 = _safe_stats(arr)
    inside_mean, inside_std, inside_p90, inside_p99 = _safe_stats(arr[bbox_mask])
    outside_mask = ~bbox_mask
    outside_mean, outside_std, outside_p90, outside_p99 = _safe_stats(arr[outside_mask])
    contrast = inside_mean - outside_mean
    snr = contrast / (outside_std + 1e-6) if not math.isnan(contrast) else math.nan

    stats: dict[str, float] = {
        "total_mean": total_mean,
        "total_std": total_std,
        "p90": p90,
        "p99": p99,
        "inside_mean": inside_mean,
        "inside_std": inside_std,
        "inside_p90": inside_p90,
        "inside_p99": inside_p99,
        "outside_mean": outside_mean,
        "outside_std": outside_std,
        "outside_p90": outside_p90,
        "outside_p99": outside_p99,
        "contrast": contrast,
        "snr": snr,
    }

    if activation_mask is not None:
        activation_mask = activation_mask.astype(bool, copy=False)
        total_act = float(activation_mask.mean()) if activation_mask.size else 0.0
        inside_act = float(activation_mask[bbox_mask].mean()) if bbox_mask.any() else 0.0
        outside_act = float(activation_mask[outside_mask].mean()) if outside_mask.any() else 0.0
        stats.update(
            {
                "activation_total": total_act,
                "activation_inside": inside_act,
                "activation_outside": outside_act,
            }
        )

    return stats


def print_feature_stats(
    name: str,
    arr: np.ndarray,
    bbox_mask: np.ndarray,
    activation_mask: np.ndarray | None = None,
) -> None:
    stats = compute_feature_stats(arr, bbox_mask, activation_mask)
    total_mean = stats["total_mean"]
    total_std = stats["total_std"]
    p90 = stats["p90"]
    p99 = stats["p99"]
    inside_mean = stats["inside_mean"]
    inside_std = stats["inside_std"]
    outside_mean = stats["outside_mean"]
    outside_std = stats["outside_std"]
    contrast = stats["contrast"]
    snr = stats["snr"]
    print(
        f"    {name}: mean={total_mean:.4f} std={total_std:.4f} p90={p90:.4f} p99={p99:.4f} | "
        f"inside mean={inside_mean:.4f} std={inside_std:.4f} | outside mean={outside_mean:.4f} std={outside_std:.4f} | "
        f"contrast={contrast:.4f} snr={snr:.2f}"
    )
    if activation_mask is not None:
        total_act = stats.get("activation_total", math.nan)
        inside_act = stats.get("activation_inside", math.nan)
        outside_act = stats.get("activation_outside", math.nan)
        print(
            f"        activation ratio: total={total_act:.4f}, inside={inside_act:.4f}, outside={outside_act:.4f}"
        )


def compute_edges_sobel(
    gray_uint8: np.ndarray,
    delta: int,
    lambda_low: float,
    lambda_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Edge extraction matching Eq.(8)-(11) in the paper.

    Parameters interpret λ values as either absolute magnitudes (>=1)
    or relative ratios to the maximum gradient magnitude (<1).
    """
    gray = gray_uint8.astype(np.float32)
    kernel_t = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_f = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    g_t = cv2.filter2D(gray, cv2.CV_32F, kernel_t, borderType=cv2.BORDER_REFLECT)
    g_f = cv2.filter2D(gray, cv2.CV_32F, kernel_f, borderType=cv2.BORDER_REFLECT)
    mag = cv2.magnitude(g_t, g_f)
    if mag.size == 0:
        return np.zeros_like(gray), np.zeros_like(gray)

    max_mag = float(mag.max())
    if max_mag <= 1e-6:
        return np.zeros_like(gray), np.zeros_like(gray)

    lambda_low_val = lambda_low * max_mag if lambda_low < 1.0 else lambda_low
    lambda_high_val = lambda_high * max_mag if lambda_high < 1.0 else lambda_high
    lambda_low_val = max(0.0, lambda_low_val)
    lambda_high_val = max(lambda_low_val + 1e-6, lambda_high_val)

    kernel = np.ones((_ensure_kernel(2 * delta + 1),) * 2, dtype=np.uint8)
    local_max = cv2.dilate(mag, kernel)
    hat_g = np.where((mag >= lambda_low_val) & (mag >= local_max - 1e-6), mag, 0.0)

    s_edge = np.zeros_like(hat_g, dtype=np.float32)
    s_edge[hat_g >= lambda_high_val] = 1.0
    mid_mask = (hat_g >= lambda_low_val) & (hat_g < lambda_high_val)
    s_edge[mid_mask] = 0.5

    mag_norm = np.zeros_like(hat_g, dtype=np.float32)
    max_hat = hat_g.max()
    if max_hat > 1e-6:
        mag_norm = hat_g / max_hat

    return mag_norm.astype(np.float32), s_edge.astype(np.float32)


def compute_corners_harris(
    gray_uint8: np.ndarray,
    gaussian_radius: int,
    gamma: float,
    eta_ratio: float,
    min_coverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corner extraction according to Eq.(12)-(13)."""
    gray = gray_uint8.astype(np.float32)
    kernel_t = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_f = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    g_t = cv2.filter2D(gray, cv2.CV_32F, kernel_t, borderType=cv2.BORDER_REFLECT)
    g_f = cv2.filter2D(gray, cv2.CV_32F, kernel_f, borderType=cv2.BORDER_REFLECT)

    size = _ensure_kernel(2 * gaussian_radius + 1)
    if size < 3:
        size = 3
    a = cv2.GaussianBlur(g_t * g_t, (size, size), 0)
    c = cv2.GaussianBlur(g_f * g_f, (size, size), 0)
    b = cv2.GaussianBlur(g_t * g_f, (size, size), 0)

    r = a * c - b * b - gamma * (a + c)
    r = np.maximum(r, 0.0)
    max_r = r.max()
    if max_r <= 1e-6:
        zeros = np.zeros_like(r, dtype=np.float32)
        return zeros, zeros, zeros

    r_norm = r / max_r
    eta = eta_ratio * max_r
    mask_raw = (r >= eta).astype(np.uint8)
    if min_coverage > 0 and mask_raw.mean() < min_coverage:
        required_percentile = max(0.0, 100.0 * (1.0 - min_coverage))
        thr = np.percentile(r, required_percentile)
        mask_raw = (r >= thr).astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.dilate(mask_clean, kernel)
    mask_clean = cv2.medianBlur(mask_clean, 3)
    if min_coverage > 0 and mask_clean.mean() < min_coverage:
        mask_clean = mask_raw

    return r_norm.astype(np.float32), mask_raw.astype(np.float32), mask_clean.astype(np.float32)


def compute_edges_log_highpass(
    channel_float: np.ndarray,
    gaussian_radius: int,
    lambda_low: float,
    lambda_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = max(1, int(gaussian_radius))
    size = _ensure_kernel(2 * radius + 1)
    blurred = cv2.GaussianBlur(channel_float.astype(np.float32, copy=False), (size, size), 0)
    highpass = channel_float.astype(np.float32, copy=False) - blurred
    magnitude = np.abs(highpass)

    magnitude = magnitude.astype(np.float32, copy=False)
    max_val = float(magnitude.max())
    if max_val <= 1e-6:
        zeros = np.zeros_like(magnitude, dtype=np.float32)
        return zeros, zeros

    lambda_low_val = lambda_low * max_val if lambda_low < 1.0 else lambda_low
    lambda_high_val = lambda_high * max_val if lambda_high < 1.0 else lambda_high
    lambda_low_val = max(0.0, lambda_low_val)
    lambda_high_val = max(lambda_low_val + 1e-6, lambda_high_val)

    scaled = np.clip((magnitude - lambda_low_val) / (lambda_high_val - lambda_low_val), 0.0, None)
    max_scaled = float(scaled.max())
    if max_scaled > 1e-6:
        scaled = scaled / max_scaled

    texture = scaled.astype(np.float32, copy=False)
    return texture, texture.copy()


def compute_edges_laplacian(
    channel_float: np.ndarray,
    radius: int,
    lambda_low: float,
    lambda_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    ksize = _ensure_kernel(2 * max(1, int(radius)) + 1)
    lap = cv2.Laplacian(channel_float.astype(np.float32, copy=False), cv2.CV_32F, ksize=ksize, borderType=cv2.BORDER_REFLECT)
    magnitude = np.abs(lap)
    return _normalize_and_threshold(magnitude, lambda_low, lambda_high)


def compute_edges_window_std(
    channel_float: np.ndarray,
    radius: int,
    lambda_low: float,
    lambda_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    size = _ensure_kernel(2 * max(1, int(radius)) + 1)
    channel = channel_float.astype(np.float32, copy=False)
    mean = cv2.blur(channel, (size, size), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.blur(channel * channel, (size, size), borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(variance).astype(np.float32, copy=False)
    return _normalize_and_threshold(std, lambda_low, lambda_high)


def compute_edges_tophat(
    channel_float: np.ndarray,
    radius: int,
    lambda_low: float,
    lambda_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    size = _ensure_kernel(2 * max(1, int(radius)) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    channel = channel_float.astype(np.float32, copy=False)
    tophat = cv2.morphologyEx(channel, cv2.MORPH_TOPHAT, kernel)
    magnitude = np.abs(tophat)
    return _normalize_and_threshold(magnitude, lambda_low, lambda_high)


def log_rescale(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, None).astype(np.float32)
    return np.log1p(arr)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    if arr.max() > 1e-6:
        arr = arr / arr.max()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def summarize(arr: np.ndarray, name: str) -> None:
    print(f"  {name}: dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, std={arr.std():.4f}")


def _compute_local_entropy_map(channel_float: np.ndarray, radius: int, levels: int = 16) -> np.ndarray:
    size = _ensure_kernel(2 * max(1, int(radius)) + 1)
    if size < 3:
        size = 3
    channel = np.clip(channel_float.astype(np.float32, copy=False), 0.0, 1.0)
    labels = np.clip((channel * (levels - 1)).astype(np.int32), 0, levels - 1)
    entropy = np.zeros_like(channel, dtype=np.float32)
    window_area = float(size * size)
    if window_area <= 0:
        return entropy
    for value in range(levels):
        mask = (labels == value).astype(np.float32)
        if mask.any():
            counts = cv2.boxFilter(mask, ddepth=-1, ksize=(size, size), normalize=False, borderType=cv2.BORDER_REFLECT)
            probs = counts / window_area
            valid = probs > 1e-6
            entropy[valid] -= probs[valid] * np.log(probs[valid] + 1e-12)
    max_entropy = math.log(levels) if levels > 1 else 1.0
    if max_entropy > 0:
        entropy = entropy / max_entropy
    return np.clip(entropy, 0.0, 1.0).astype(np.float32)


def build_sample_cache(
    spec_float: np.ndarray,
    bboxes: np.ndarray,
    clip_low: float,
    clip_high: float,
) -> dict[str, object]:
    gray_raw = np.clip(spec_float.astype(np.float32), 0.0, None)
    gray_norm = percentile_normalize(gray_raw, clip_low, clip_high)
    gray_u8 = to_uint8(gray_raw)
    gray_norm_u8 = to_uint8(gray_norm)

    log_gray = log_rescale(gray_raw)
    log_gray_norm = percentile_normalize(log_gray, clip_low, clip_high)
    log_gray_u8 = to_uint8(log_gray_norm)

    bbox_mask = make_bbox_mask(bboxes, gray_norm.shape[0], gray_norm.shape[1])

    return {
        "gray_raw": gray_raw,
        "gray_norm": gray_norm,
        "gray_u8": gray_u8,
        "gray_norm_u8": gray_norm_u8,
        "log_gray": log_gray,
        "log_gray_norm": log_gray_norm,
        "log_gray_u8": log_gray_u8,
        "bbox_mask": bbox_mask,
        "method_cache": {},
    }


def _method_cache(cache: dict[str, object]) -> dict[tuple, dict[str, object]]:
    method_cache = cache.get("method_cache")
    if not isinstance(method_cache, dict):
        method_cache = {}
        cache["method_cache"] = method_cache
    return method_cache  # type: ignore[return-value]


def _get_sobel_data(
    cache: dict[str, object],
    use_log: bool,
    delta: int,
) -> dict[str, object]:
    key = ("sobel", bool(use_log), int(delta))
    method_cache = _method_cache(cache)
    entry = method_cache.get(key)
    if entry is not None:
        return entry

    base_u8 = cache["log_gray_u8"] if use_log else cache["gray_u8"]
    gray = np.asarray(base_u8, dtype=np.float32)
    kernel_t = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_f = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    g_t = cv2.filter2D(gray, cv2.CV_32F, kernel_t, borderType=cv2.BORDER_REFLECT)
    g_f = cv2.filter2D(gray, cv2.CV_32F, kernel_f, borderType=cv2.BORDER_REFLECT)
    mag = cv2.magnitude(g_t, g_f)
    max_mag = float(mag.max())

    kernel_size = _ensure_kernel(2 * max(1, int(delta)) + 1)
    dilate_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_max = cv2.dilate(mag, dilate_kernel)
    hat_base = np.where(mag >= local_max - 1e-6, mag, 0.0).astype(np.float32)

    entry = {"hat_base": hat_base, "max_mag": max_mag}
    method_cache[key] = entry
    return entry


def _compute_sobel_from_cache(
    cache: dict[str, object],
    use_log: bool,
    delta: int,
    lambda_low: float,
    lambda_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = _get_sobel_data(cache, use_log, delta)
    hat_base = np.asarray(data["hat_base"], dtype=np.float32)
    max_mag = float(data["max_mag"])
    if max_mag <= 1e-6:
        zeros = np.zeros_like(hat_base, dtype=np.float32)
        return zeros, zeros

    lambda_low_val = lambda_low * max_mag if lambda_low < 1.0 else lambda_low
    lambda_high_val = lambda_high * max_mag if lambda_high < 1.0 else lambda_high
    lambda_low_val = max(0.0, lambda_low_val)
    lambda_high_val = max(lambda_low_val + 1e-6, lambda_high_val)

    hat = np.where(hat_base >= lambda_low_val, hat_base, 0.0).astype(np.float32)
    max_hat = float(hat.max())
    if max_hat > 1e-6:
        edge_raw = hat / max_hat
    else:
        edge_raw = np.zeros_like(hat, dtype=np.float32)

    edge_clean = np.zeros_like(hat_base, dtype=np.float32)
    edge_clean[hat_base >= lambda_high_val] = 1.0
    mid_mask = (hat_base >= lambda_low_val) & (hat_base < lambda_high_val)
    edge_clean[mid_mask] = 0.5

    return edge_raw.astype(np.float32), edge_clean.astype(np.float32)


def _get_magnitude_map(
    cache: dict[str, object],
    method: str,
    use_log: bool,
    radius: int,
) -> dict[str, object]:
    radius = max(1, int(radius))
    key = (method, bool(use_log), radius)
    method_cache = _method_cache(cache)
    entry = method_cache.get(key)
    if entry is not None:
        return entry

    def _select_source(raw: bool = False) -> np.ndarray:
        if raw:
            return np.asarray(cache["log_gray"] if use_log else cache["gray_raw"], dtype=np.float32)
        return np.asarray(cache["log_gray_norm"] if use_log else cache["gray_norm"], dtype=np.float32)

    if method == "log-hp":
        source = _select_source(raw=True)
        size = _ensure_kernel(2 * radius + 1)
        blurred = cv2.GaussianBlur(source, (size, size), 0)
        magnitude = np.abs(source - blurred).astype(np.float32)
        entry = {"magnitude": magnitude, "max_val": float(magnitude.max())}
    elif method == "laplacian":
        source = _select_source()
        ksize = _ensure_kernel(2 * radius + 1)
        lap = cv2.Laplacian(source, cv2.CV_32F, ksize=ksize, borderType=cv2.BORDER_REFLECT)
        entry = {"magnitude": np.abs(lap).astype(np.float32)}
    elif method == "window-std":
        source = _select_source()
        size = _ensure_kernel(2 * radius + 1)
        mean = cv2.blur(source, (size, size), borderType=cv2.BORDER_REFLECT)
        mean_sq = cv2.blur(source * source, (size, size), borderType=cv2.BORDER_REFLECT)
        variance = np.maximum(mean_sq - mean * mean, 0.0)
        entry = {"magnitude": np.sqrt(variance).astype(np.float32)}
    elif method == "tophat":
        source = _select_source()
        size = _ensure_kernel(2 * radius + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        tophat = cv2.morphologyEx(source, cv2.MORPH_TOPHAT, kernel)
        entry = {"magnitude": np.abs(tophat).astype(np.float32)}
    elif method == "gabor":
        source = _select_source()
        ksize = _ensure_kernel(2 * radius + 1)
        if ksize < 3:
            ksize = 3
        sigma = max(1.0, 0.5 * ksize)
        lambd = max(4.0, float(ksize))
        gamma = 0.5
        psi = 0.0
        orientations = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
        energy_accumulator: np.ndarray | None = None
        count = 0
        for theta in orientations:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            kernel = kernel.astype(np.float32)
            kernel -= float(kernel.mean())
            norm = float(np.linalg.norm(kernel))
            if norm > 1e-6:
                kernel /= norm
            filtered = cv2.filter2D(source, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            magnitude_map = np.abs(filtered).astype(np.float32)
            if energy_accumulator is None:
                energy_accumulator = magnitude_map * magnitude_map
            else:
                energy_accumulator += magnitude_map * magnitude_map
            count += 1
        if energy_accumulator is None or count == 0:
            magnitude = np.zeros_like(source, dtype=np.float32)
        else:
            magnitude = np.sqrt(energy_accumulator / count).astype(np.float32)
        entry = {"magnitude": magnitude}
    elif method == "local-entropy":
        source = _select_source()
        entropy = _compute_local_entropy_map(source, radius)
        entry = {"magnitude": entropy.astype(np.float32)}
    else:
        raise ValueError(f"Unsupported magnitude method: {method}")

    method_cache[key] = entry
    return entry


def _compute_log_hp_norm(
    magnitude: np.ndarray,
    max_val: float,
    lambda_low: float,
    lambda_high: float,
) -> np.ndarray:
    if max_val <= 1e-6:
        return np.zeros_like(magnitude, dtype=np.float32)

    lambda_low_val = lambda_low * max_val if lambda_low < 1.0 else lambda_low
    lambda_high_val = lambda_high * max_val if lambda_high < 1.0 else lambda_high
    lambda_low_val = max(0.0, lambda_low_val)
    lambda_high_val = max(lambda_low_val + 1e-6, lambda_high_val)
    scaled = np.clip((magnitude - lambda_low_val) / (lambda_high_val - lambda_low_val), 0.0, None)
    max_scaled = float(scaled.max())
    if max_scaled > 1e-6:
        scaled = scaled / max_scaled
    return scaled.astype(np.float32)


def _compute_canny_from_cache(
    cache: dict[str, object],
    use_log: bool,
    radius: int,
    lambda_low: float,
    lambda_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    key = (
        "canny",
        bool(use_log),
        int(radius),
        float(lambda_low),
        float(lambda_high),
    )
    method_cache = _method_cache(cache)
    entry = method_cache.get(key)
    if entry is None:
        base = cache["log_gray_u8"] if use_log else cache["gray_u8"]
        base_u8 = np.asarray(base, dtype=np.uint8)
        blur_size = _ensure_kernel(2 * max(1, int(radius)) + 1) if radius > 1 else 0
        if blur_size >= 3:
            blurred = cv2.GaussianBlur(base_u8, (blur_size, blur_size), 0)
        else:
            blurred = base_u8
        aperture = _ensure_kernel(2 * max(1, int(radius)) + 1)
        if aperture not in (3, 5, 7):
            aperture = 3 if aperture < 5 else 7
        low_thresh = lambda_low * 255.0 if lambda_low < 1.0 else lambda_low
        high_thresh = lambda_high * 255.0 if lambda_high < 1.0 else lambda_high
        low_thresh = max(0.0, min(255.0, float(low_thresh)))
        high_thresh = max(low_thresh + 1.0, min(255.0, float(high_thresh)))
        low_int = int(round(low_thresh))
        high_int = int(round(high_thresh))
        edges = cv2.Canny(blurred, low_int, high_int, apertureSize=aperture, L2gradient=True)
        edge_float = edges.astype(np.float32) / 255.0
        entry = {"edge": edge_float, "input": base_u8}
        method_cache[key] = entry
    return (
        np.asarray(entry["edge"], dtype=np.float32),
        np.asarray(entry["input"], dtype=np.uint8),
    )


def _compute_edge_maps_cached(
    cache: dict[str, object],
    method: str,
    use_log: bool,
    radius: int,
    lambda_low: float,
    lambda_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


    method_norm = method.lower()
    if method_norm == "sobel":
        edge_raw, edge_clean = _compute_sobel_from_cache(cache, use_log, radius, lambda_low, lambda_high)
        edge_input_u8 = cache["log_gray_u8"] if use_log else cache["gray_u8"]
        return edge_raw, edge_clean, np.asarray(edge_input_u8, dtype=np.uint8)

    if method_norm == "canny":
        edge_raw, edge_input_u8 = _compute_canny_from_cache(cache, use_log, radius, lambda_low, lambda_high)
        edge_clean = edge_raw.copy()
        edge_input_u8 = np.asarray(edge_input_u8, dtype=np.uint8)
        return edge_raw.astype(np.float32), edge_clean.astype(np.float32), edge_input_u8

    if method_norm in {"log-hp", "log-hp-raw"}:
        data = _get_magnitude_map(cache, "log-hp", use_log, radius)
        magnitude = np.asarray(data["magnitude"], dtype=np.float32)
        max_val = float(data.get("max_val", float(magnitude.max())))
        if method_norm == "log-hp-raw":
            if max_val > 1e-6:
                edge_raw = (magnitude / max_val).astype(np.float32)
            else:
                edge_raw = np.zeros_like(magnitude, dtype=np.float32)
            edge_clean = edge_raw.copy()
        else:
            edge_raw = _compute_log_hp_norm(magnitude, max_val, lambda_low, lambda_high)
            edge_clean = edge_raw.copy()
        edge_input_u8 = to_uint8(edge_raw)
        return edge_raw, edge_clean, edge_input_u8

    if method_norm in {"laplacian", "window-std", "tophat", "gabor", "local-entropy"}:
        data = _get_magnitude_map(cache, method_norm, use_log, radius)
        magnitude = np.asarray(data["magnitude"], dtype=np.float32)
        edge_raw, edge_clean = _normalize_and_threshold(magnitude, lambda_low, lambda_high)
        edge_input_u8 = to_uint8(edge_raw)
        return edge_raw, edge_clean, edge_input_u8

    raise ValueError(f"Unsupported edge method: {method}")


def prepare_features(
    spec_float: np.ndarray | None,
    bboxes: np.ndarray | None,
    clip_low: float,
    clip_high: float,
    edge_delta: int,
    edge_lambda_low: float,
    edge_lambda_high: float,
    edge_threshold: float,
    edge_max_coverage: float,
    corner_mask_threshold: float,
    corner_min_coverage: float,
    corner_gaussian_radius: int,
    corner_gamma: float,
    corner_eta_ratio: float,
    edge_use_log: bool,
    edge_method: str,
    texture_method: str,
    texture_use_log: bool,
    texture_lambda_low: float,
    texture_lambda_high: float,
    texture_threshold: float,
    texture_max_coverage: float,
    texture_radius: int,
    precomputed: dict[str, object] | None = None,
) -> dict[str, np.ndarray | float]:
    cache: dict[str, object]
    if precomputed is None:
        if spec_float is None or bboxes is None:
            raise ValueError("spec_float and bboxes must be provided when no cache is supplied.")
        cache = build_sample_cache(spec_float, bboxes, clip_low, clip_high)
    else:
        cache = precomputed

    gray_raw = np.asarray(cache["gray_raw"], dtype=np.float32)
    gray_norm = np.asarray(cache["gray_norm"], dtype=np.float32)
    gray_u8 = np.asarray(cache["gray_u8"], dtype=np.uint8)
    gray_norm_u8 = np.asarray(cache["gray_norm_u8"], dtype=np.uint8)
    log_gray_norm = np.asarray(cache["log_gray_norm"], dtype=np.float32)
    log_gray_u8 = np.asarray(cache["log_gray_u8"], dtype=np.uint8)

    method = (edge_method or "sobel").lower()
    if method not in EDGE_METHOD_CHOICES:
        method = "sobel"

    edge_raw, edge_clean, edge_input_u8 = _compute_edge_maps_cached(
        cache,
        method,
        edge_use_log,
        edge_delta,
        edge_lambda_low,
        edge_lambda_high,
    )

    texture_method_norm = (texture_method or "none").lower()
    if texture_method_norm not in TEXTURE_METHOD_CHOICES:
        texture_method_norm = "none"

    texture_map = np.zeros_like(gray_norm, dtype=np.float32)
    texture_threshold_used = float(texture_threshold)
    texture_activation = np.zeros_like(gray_norm, dtype=bool)
    texture_input_u8 = np.zeros_like(gray_norm_u8)

    if texture_method_norm != "none":
        tex_radius = max(1, int(texture_radius))
        tex_lambda_low = max(0.0, float(texture_lambda_low))
        tex_lambda_high = max(tex_lambda_low + 1e-6, float(texture_lambda_high))

        tex_raw, _tex_clean, _tex_input = _compute_edge_maps_cached(
            cache,
            texture_method_norm,
            texture_use_log,
            tex_radius,
            tex_lambda_low,
            tex_lambda_high,
        )

        texture_map = np.clip(tex_raw, 0.0, 1.0)
        texture_activation = texture_map >= texture_threshold_used
        if texture_max_coverage > 0 and float(texture_activation.mean()) > texture_max_coverage:
            cutoff = 100.0 * (1.0 - texture_max_coverage)
            adaptive_thr = float(np.percentile(texture_map, cutoff))
            texture_threshold_used = max(texture_threshold_used, adaptive_thr)
            texture_activation = texture_map >= texture_threshold_used
        texture_input_u8 = to_uint8(texture_map)
    corner_resp, corner_mask_raw, corner_mask_clean = compute_corners_harris(
        gray_u8,
        gaussian_radius=corner_gaussian_radius,
        gamma=corner_gamma,
        eta_ratio=corner_eta_ratio,
        min_coverage=corner_min_coverage,
    )

    bbox_mask = np.asarray(cache["bbox_mask"], dtype=bool)

    threshold_used = edge_threshold
    edge_activation = edge_clean >= threshold_used
    if edge_max_coverage > 0 and float(edge_activation.mean()) > edge_max_coverage:
        cutoff = 100.0 * (1.0 - edge_max_coverage)
        adaptive_thr = float(np.percentile(edge_clean, cutoff))
        threshold_used = max(threshold_used, adaptive_thr)
        edge_activation = edge_clean >= threshold_used

    corner_activation_raw = corner_mask_raw >= corner_mask_threshold
    corner_activation = corner_mask_clean >= corner_mask_threshold

    if texture_method_norm != "none":
        pseudo_rgb = np.stack([log_gray_norm, texture_map, corner_mask_clean], axis=-1)
    else:
        pseudo_rgb = np.stack([log_gray_norm, edge_clean, corner_mask_clean], axis=-1)
    pseudo_rgb_u8 = to_uint8(pseudo_rgb)

    return {
        "gray_raw": gray_raw,
        "gray_norm": gray_norm,
        "gray_u8": gray_u8,
        "gray_norm_u8": gray_norm_u8,
        "log_gray_norm": log_gray_norm,
        "log_gray_u8": log_gray_u8,
        "edge_input_u8": edge_input_u8,
        "edge_raw": edge_raw,
        "edge_clean": edge_clean,
        "edge_activation": edge_activation,
        "edge_threshold_used": float(threshold_used),
        "corner_resp": corner_resp,
        "corner_mask_raw": corner_mask_raw,
        "corner_mask_clean": corner_mask_clean,
        "corner_activation_raw": corner_activation_raw,
        "corner_activation": corner_activation,
        "pseudo_rgb_u8": pseudo_rgb_u8,
        "bbox_mask": bbox_mask,
        "edge_method": method,
        "texture_map": texture_map,
        "texture_activation": texture_activation,
        "texture_threshold_used": float(texture_threshold_used),
        "texture_method": texture_method_norm,
        "texture_input_u8": texture_input_u8,
    }


def visualize(
    sample_id: str,
    image: np.ndarray,
    spec_float: np.ndarray,
    bboxes: np.ndarray,
    out_dir: Path,
    clip_low: float,
    clip_high: float,
    edge_delta: int,
    edge_lambda_low: float,
    edge_lambda_high: float,
    edge_threshold: float,
    corner_mask_threshold: float,
    edge_max_coverage: float,
    corner_min_coverage: float,
    corner_gaussian_radius: int,
    corner_gamma: float,
    corner_eta_ratio: float,
    create_plot: bool = True,
    edge_use_log: bool = False,
    edge_method: str = "sobel",
    texture_method: str = "none",
    texture_use_log: bool = False,
    texture_lambda_low: float = 0.0,
    texture_lambda_high: float = 1.0,
    texture_threshold: float = 0.5,
    texture_max_coverage: float = 0.1,
    texture_radius: int = 3,
) -> None:
    features = prepare_features(
        spec_float,
        bboxes,
        clip_low,
        clip_high,
        edge_delta,
        edge_lambda_low,
        edge_lambda_high,
        edge_threshold,
        edge_max_coverage,
        corner_mask_threshold,
        corner_min_coverage,
        corner_gaussian_radius,
        corner_gamma,
        corner_eta_ratio,
        edge_use_log,
        edge_method,
        texture_method,
        texture_use_log,
        texture_lambda_low,
        texture_lambda_high,
        texture_threshold,
        texture_max_coverage,
        texture_radius,
    )

    gray_norm = features["gray_norm"]
    gray_norm_u8 = features["gray_norm_u8"]
    gray_u8 = features["gray_u8"]
    log_gray_u8 = features["log_gray_u8"]
    log_gray_norm = features["log_gray_norm"]
    edge_raw = features["edge_raw"]
    edge_clean = features["edge_clean"]
    edge_activation = features["edge_activation"]
    threshold_used = features["edge_threshold_used"]
    corner_resp = features["corner_resp"]
    corner_mask_raw = features["corner_mask_raw"]
    corner_mask_clean = features["corner_mask_clean"]
    corner_activation_raw = features["corner_activation_raw"]
    corner_activation = features["corner_activation"]
    bbox_mask = features["bbox_mask"]
    pseudo_rgb_u8 = features["pseudo_rgb_u8"]
    edge_input_u8 = features["edge_input_u8"]
    method_used = str(features.get("edge_method", edge_method or "sobel")).lower()
    texture_map = features["texture_map"]
    texture_activation = features["texture_activation"]
    texture_threshold_used = float(features.get("texture_threshold_used", texture_threshold))
    texture_method_used = str(features.get("texture_method", texture_method or "none")).lower()
    texture_input_u8 = features["texture_input_u8"]

    print(f"    bbox pixel ratio: {float(bbox_mask.mean()):.4f}")
    print_feature_stats("gray_norm", gray_norm, bbox_mask)
    print_feature_stats("log_gray_norm", log_gray_norm, bbox_mask)
    print_feature_stats("edge_raw", edge_raw, bbox_mask)
    print_feature_stats("edge_clean", edge_clean, bbox_mask, edge_activation)
    edge_cov = float(edge_activation.mean())
    print(
        f"        edge threshold used: {threshold_used:.4f} (requested {edge_threshold:.4f}); "
        f"coverage={edge_cov:.4f}, target_max={edge_max_coverage:.4f}"
    )
    if texture_method_used != "none":
        print_feature_stats("texture_map", texture_map, bbox_mask, texture_activation)
        texture_cov = float(texture_activation.mean())
        print(
            f"        texture threshold used: {texture_threshold_used:.4f} (requested {texture_threshold:.4f}); "
            f"coverage={texture_cov:.4f}, target_max={texture_max_coverage:.4f}"
        )
    print_feature_stats("corner_resp", corner_resp, bbox_mask)
    print_feature_stats("corner_mask_raw", corner_mask_raw, bbox_mask, corner_activation_raw)
    print_feature_stats("corner_mask_clean", corner_mask_clean, bbox_mask, corner_activation)
    corner_cov = float(corner_activation.mean())
    corner_cov_raw = float(corner_activation_raw.mean())
    print(
        f"        corner coverage raw={corner_cov_raw:.4f}, clean={corner_cov:.4f}, "
        f"enforced_min={corner_min_coverage:.4f}, activation threshold={corner_mask_threshold:.4f}"
    )

    if create_plot:
        edge_activation_img = edge_activation.astype(np.float32)
        texture_activation_img = texture_activation.astype(np.float32)

        fig, ax = plt.subplots(2, 8, figsize=(24, 6))
        ax = ax.flatten()

        ax[0].imshow(image)
        ax[0].set_title("original image")
        ax[1].imshow(gray_u8, cmap="gray")
        ax[1].set_title("gray raw")
        ax[2].imshow(gray_norm_u8, cmap="gray")
        ax[2].set_title("gray norm")
        ax[3].imshow(log_gray_u8, cmap="gray")
        ax[3].set_title("log gray norm")
        ax[4].imshow(gray_norm, cmap="inferno")
        ax[4].set_title("gray norm heat")
        title_suffix_parts: list[str] = []
        if edge_use_log and "log" not in method_used:
            title_suffix_parts.append("log")
        if method_used != "sobel":
            title_suffix_parts.append(method_used)
        title_suffix = f" ({'+'.join(title_suffix_parts)})" if title_suffix_parts else ""
        ax[5].imshow(edge_input_u8, cmap="gray")
        ax[5].set_title(f"edge input{title_suffix}")
        ax[6].imshow(edge_raw, cmap="magma")
        ax[6].set_title("edge raw")
        ax[7].imshow(edge_clean, cmap="magma")
        ax[7].set_title("edge clean")
        ax[8].imshow(edge_activation_img, cmap="binary")
        ax[8].set_title("edge activation")
        title_tex: list[str] = []
        if texture_use_log and "log" not in texture_method_used:
            title_tex.append("log")
        if texture_method_used != "none":
            title_tex.append(texture_method_used)
        texture_suffix = f" ({'+'.join(title_tex)})" if title_tex else ""
        ax[9].imshow(texture_input_u8, cmap="gray")
        ax[9].set_title(f"texture input{texture_suffix}")
        ax[10].imshow(texture_map, cmap="cividis")
        ax[10].set_title("texture map")
        ax[11].imshow(texture_activation_img, cmap="binary")
        ax[11].set_title("texture activation")
        ax[12].imshow(pseudo_rgb_u8)
        ax[12].set_title("pseudo rgb")
        ax[13].imshow(corner_resp, cmap="viridis")
        ax[13].set_title("corner resp")
        ax[14].imshow(corner_mask_raw, cmap="plasma")
        ax[14].set_title("corner mask raw")
        ax[15].imshow(corner_mask_clean, cmap="plasma")
        ax[15].set_title("corner mask clean")

        for a in ax:
            a.axis("off")

        out_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{sample_id}_compare.png", dpi=150)
        plt.close(fig)


def _unique_preserve_order(values: Iterable[float]) -> list[float]:
    seen: set[float] = set()
    ordered: list[float] = []
    for val in values:
        v = float(val)
        if math.isnan(v):
            continue
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0 or np.isnan(arr).all():
        return math.nan
    return float(np.nanmean(arr))


def _nanstd(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0 or np.isnan(arr).all():
        return math.nan
    return float(np.nanstd(arr))


def _nanmin(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0 or np.isnan(arr).all():
        return math.nan
    return float(np.nanmin(arr))


def _nanmax(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0 or np.isnan(arr).all():
        return math.nan
    return float(np.nanmax(arr))


def aggregate_sample_metrics(metrics: list[dict[str, float]]) -> dict[str, float | int]:
    agg: dict[str, float | int] = {"sample_count": len(metrics)}
    if not metrics:
        return agg

    fields = [
        "bbox_ratio",
        "edge_contrast",
        "edge_snr",
        "edge_activation_total",
        "edge_activation_inside",
        "edge_activation_outside",
        "edge_threshold_used",
        "texture_contrast",
        "texture_snr",
        "texture_activation_total",
        "texture_activation_inside",
        "texture_activation_outside",
        "texture_threshold_used",
        "corner_resp_contrast",
        "corner_resp_snr",
        "corner_clean_contrast",
        "corner_clean_snr",
        "corner_clean_activation_total",
        "corner_clean_activation_inside",
        "corner_clean_activation_outside",
        "corner_raw_activation_total",
        "corner_raw_activation_inside",
        "corner_raw_activation_outside",
    ]

    for field in fields:
        values = [row.get(field, math.nan) for row in metrics]
        agg[f"{field}_mean"] = _nanmean(values)
        agg[f"{field}_std"] = _nanstd(values)
        if field in {"edge_threshold_used", "texture_threshold_used"}:
            agg[f"{field}_min"] = _nanmin(values)
            agg[f"{field}_max"] = _nanmax(values)

    return agg


def _fmt_float(value: float, precision: int = 4) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


def run_parameter_sweep(
    samples: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    edge_lambda_low_vals = _unique_preserve_order(
        args.sweep_edge_lambda_low if args.sweep_edge_lambda_low else [args.edge_lambda_low]
    )
    edge_lambda_high_vals = _unique_preserve_order(
        args.sweep_edge_lambda_high if args.sweep_edge_lambda_high else [args.edge_lambda_high]
    )
    edge_threshold_vals = _unique_preserve_order(
        args.sweep_edge_threshold if args.sweep_edge_threshold else [args.edge_threshold]
    )
    edge_max_cov_vals = _unique_preserve_order(
        args.sweep_edge_max_coverage if args.sweep_edge_max_coverage else [args.edge_max_coverage]
    )
    texture_lambda_low_vals = _unique_preserve_order(
        args.sweep_texture_lambda_low if args.sweep_texture_lambda_low else [args.texture_lambda_low]
    )
    texture_lambda_high_vals = _unique_preserve_order(
        args.sweep_texture_lambda_high if args.sweep_texture_lambda_high else [args.texture_lambda_high]
    )
    texture_threshold_vals = _unique_preserve_order(
        args.sweep_texture_threshold if args.sweep_texture_threshold else [args.texture_threshold]
    )
    texture_max_cov_vals = _unique_preserve_order(
        args.sweep_texture_max_coverage if args.sweep_texture_max_coverage else [args.texture_max_coverage]
    )
    corner_eta_vals = _unique_preserve_order(
        args.sweep_corner_eta_ratio if args.sweep_corner_eta_ratio else [args.corner_eta_ratio]
    )
    corner_mask_thr_vals = _unique_preserve_order(
        args.sweep_corner_mask_threshold if args.sweep_corner_mask_threshold else [args.corner_mask_threshold]
    )
    corner_min_cov_vals = _unique_preserve_order(
        args.sweep_corner_min_coverage if args.sweep_corner_min_coverage else [args.corner_min_coverage]
    )

    edge_methods = [m.lower() for m in (args.sweep_edge_method or [args.edge_method])]
    edge_methods = [m for m in edge_methods if m in EDGE_METHOD_CHOICES]

    texture_methods = [m.lower() for m in (args.sweep_texture_method or [args.texture_method])]
    texture_methods = [m for m in texture_methods if m in TEXTURE_METHOD_CHOICES]
    if not texture_methods:
        texture_methods = ["none"]

    if not edge_methods:
        print("No valid edge methods specified for sweep.")
        return

    total_combos = (
        len(edge_methods)
        * len(texture_methods)
        * len(edge_lambda_low_vals)
        * len(edge_lambda_high_vals)
        * len(edge_threshold_vals)
        * len(edge_max_cov_vals)
        * len(texture_lambda_low_vals)
        * len(texture_lambda_high_vals)
        * len(texture_threshold_vals)
        * len(texture_max_cov_vals)
        * len(corner_eta_vals)
        * len(corner_mask_thr_vals)
        * len(corner_min_cov_vals)
    )
    if total_combos == 0:
        print("No sweep combinations to evaluate.")
        return

    all_combos = list(
        product(
            edge_methods,
            texture_methods,
            edge_lambda_low_vals,
            edge_lambda_high_vals,
            edge_threshold_vals,
            edge_max_cov_vals,
            texture_lambda_low_vals,
            texture_lambda_high_vals,
            texture_threshold_vals,
            texture_max_cov_vals,
            corner_eta_vals,
            corner_mask_thr_vals,
            corner_min_cov_vals,
        )
    )

    if args.sweep_randomize:
        sweep_seed = args.sweep_seed if args.sweep_seed is not None else args.random_seed
        rng = random.Random(sweep_seed) if sweep_seed is not None else random.Random()
        rng.shuffle(all_combos)

    if args.sweep_max_combos and args.sweep_max_combos > 0 and len(all_combos) > args.sweep_max_combos:
        all_combos = all_combos[: args.sweep_max_combos]

    eval_total = len(all_combos)
    if eval_total == 0:
        print("No sweep combinations to evaluate after applying limits.")
        return

    print(
        "Running sweep across "
        f"edge_methods={edge_methods}, texture_methods={texture_methods}, "
        f"edge λ_L={edge_lambda_low_vals}, edge λ_H={edge_lambda_high_vals}, edge_thr={edge_threshold_vals}, "
        f"edge_cov={edge_max_cov_vals}, texture λ_L={texture_lambda_low_vals}, texture λ_H={texture_lambda_high_vals}, "
        f"texture_thr={texture_threshold_vals}, texture_cov={texture_max_cov_vals}, corner_eta={corner_eta_vals}, "
        f"corner_thr={corner_mask_thr_vals}, corner_cov={corner_min_cov_vals} | total={total_combos}, evaluating={eval_total}"
    )

    sample_caches = [
        build_sample_cache(spec, bboxes, args.clip_low, args.clip_high) for _, _, spec, bboxes in samples
    ]

    results: list[dict[str, float]] = []
    for combo_index, (
        method_name,
        texture_method_name,
        lam_low,
        lam_high,
        edge_thr,
        edge_cov,
        tex_lam_low,
        tex_lam_high,
        tex_thr,
        tex_cov,
        corner_eta,
        corner_thr,
        corner_cov,
    ) in enumerate(all_combos, start=1):
        if lam_high < lam_low + 1e-6:
            print(
                f"[{combo_index}/{eval_total}] Skip λ_L={lam_low:.4g}, λ_H={lam_high:.4g} because upper threshold < lower threshold."
            )
            continue

        lam_low_clamped = max(0.0, lam_low)
        lam_high_clamped = max(lam_low_clamped + 1e-6, lam_high)
        edge_thr_clamped = float(min(max(edge_thr, 0.0), 1.0))
        edge_cov_clamped = float(min(max(edge_cov, 0.0), 1.0))
        tex_lam_low_clamped = max(0.0, tex_lam_low)
        tex_lam_high_clamped = max(tex_lam_low_clamped + 1e-6, tex_lam_high)
        tex_thr_clamped = float(min(max(tex_thr, 0.0), 1.0))
        tex_cov_clamped = float(min(max(tex_cov, 0.0), 1.0))
        corner_thr_clamped = float(min(max(corner_thr, 0.0), 1.0))
        corner_cov_clamped = float(min(max(corner_cov, 0.0), 1.0))
        corner_eta_clamped = max(0.0, corner_eta)

        print(
            f"[{combo_index}/{eval_total}] edge_method={method_name}, texture_method={texture_method_name}, "
            f"edge λ_L={lam_low_clamped:.4g}, edge λ_H={lam_high_clamped:.4g}, edge_thr={edge_thr_clamped:.3f}, "
            f"edge_cov={edge_cov_clamped:.3f}, texture λ_L={tex_lam_low_clamped:.4g}, texture λ_H={tex_lam_high_clamped:.4g}, "
            f"texture_thr={tex_thr_clamped:.3f}, texture_cov={tex_cov_clamped:.3f}, corner_eta={corner_eta_clamped:.3f}, "
            f"corner_thr={corner_thr_clamped:.3f}, corner_cov={corner_cov_clamped:.3f}"
        )

        sample_metrics: list[dict[str, float]] = []
        for cache in sample_caches:
            features = prepare_features(
                None,
                None,
                args.clip_low,
                args.clip_high,
                args.edge_delta,
                lam_low_clamped,
                lam_high_clamped,
                edge_thr_clamped,
                edge_cov_clamped,
                corner_thr_clamped,
                corner_cov_clamped,
                args.corner_gaussian_radius,
                args.corner_gamma,
                corner_eta_clamped,
                args.edge_use_log,
                method_name,
                texture_method_name,
                args.texture_use_log,
                tex_lam_low_clamped,
                tex_lam_high_clamped,
                tex_thr_clamped,
                tex_cov_clamped,
                args.texture_radius,
                precomputed=cache,
            )

            bbox_mask = features["bbox_mask"]
            edge_stats = compute_feature_stats(features["edge_clean"], bbox_mask, features["edge_activation"])
            texture_stats = compute_feature_stats(
                features["texture_map"], bbox_mask, features["texture_activation"]
            )
            corner_clean_stats = compute_feature_stats(
                features["corner_mask_clean"], bbox_mask, features["corner_activation"]
            )
            corner_raw_stats = compute_feature_stats(
                features["corner_mask_raw"], bbox_mask, features["corner_activation_raw"]
            )
            corner_resp_stats = compute_feature_stats(features["corner_resp"], bbox_mask)

            metrics_row = {
                "bbox_ratio": float(bbox_mask.mean()) if bbox_mask.size else math.nan,
                "edge_contrast": edge_stats["contrast"],
                "edge_snr": edge_stats["snr"],
                "edge_activation_total": edge_stats.get("activation_total", math.nan),
                "edge_activation_inside": edge_stats.get("activation_inside", math.nan),
                "edge_activation_outside": edge_stats.get("activation_outside", math.nan),
                "edge_threshold_used": float(features["edge_threshold_used"]),
                "texture_contrast": texture_stats["contrast"],
                "texture_snr": texture_stats["snr"],
                "texture_activation_total": texture_stats.get("activation_total", math.nan),
                "texture_activation_inside": texture_stats.get("activation_inside", math.nan),
                "texture_activation_outside": texture_stats.get("activation_outside", math.nan),
                "texture_threshold_used": float(features["texture_threshold_used"]),
                "corner_resp_contrast": corner_resp_stats["contrast"],
                "corner_resp_snr": corner_resp_stats["snr"],
                "corner_clean_contrast": corner_clean_stats["contrast"],
                "corner_clean_snr": corner_clean_stats["snr"],
                "corner_clean_activation_total": corner_clean_stats.get("activation_total", math.nan),
                "corner_clean_activation_inside": corner_clean_stats.get("activation_inside", math.nan),
                "corner_clean_activation_outside": corner_clean_stats.get("activation_outside", math.nan),
                "corner_raw_activation_total": corner_raw_stats.get("activation_total", math.nan),
                "corner_raw_activation_inside": corner_raw_stats.get("activation_inside", math.nan),
                "corner_raw_activation_outside": corner_raw_stats.get("activation_outside", math.nan),
            }
            sample_metrics.append(metrics_row)

        agg = aggregate_sample_metrics(sample_metrics)
        agg.update(
            {
                "edge_lambda_low": lam_low_clamped,
                "edge_lambda_high": lam_high_clamped,
                "edge_threshold": edge_thr_clamped,
                "edge_max_coverage": edge_cov_clamped,
                "corner_eta_ratio": corner_eta_clamped,
                "corner_mask_threshold": corner_thr_clamped,
                "corner_min_coverage": corner_cov_clamped,
                "edge_use_log": int(args.edge_use_log),
                "edge_method": method_name,
                "texture_method": texture_method_name,
                "texture_lambda_low": tex_lam_low_clamped,
                "texture_lambda_high": tex_lam_high_clamped,
                "texture_threshold": tex_thr_clamped,
                "texture_max_coverage": tex_cov_clamped,
                "texture_use_log": int(args.texture_use_log),
            }
        )
        results.append(agg)

        print(
            "    edge SNR mean="
            f"{_fmt_float(agg.get('edge_snr_mean'), 3)}, coverage total={_fmt_float(agg.get('edge_activation_total_mean'), 3)}, "
            f"inside={_fmt_float(agg.get('edge_activation_inside_mean'), 3)}"
        )
        print(
            "    texture SNR mean="
            f"{_fmt_float(agg.get('texture_snr_mean'), 3)}, coverage total={_fmt_float(agg.get('texture_activation_total_mean'), 3)}, "
            f"inside={_fmt_float(agg.get('texture_activation_inside_mean'), 3)}"
        )
        print(
            "    corner SNR mean="
            f"{_fmt_float(agg.get('corner_clean_snr_mean'), 3)}, coverage total={_fmt_float(agg.get('corner_clean_activation_total_mean'), 3)}, "
            f"inside={_fmt_float(agg.get('corner_clean_activation_inside_mean'), 3)}"
        )

    if not results:
        print("No valid sweep results computed.")
        return []

    sort_metric_map = {
        "edge_snr": "edge_snr_mean",
        "texture_snr": "texture_snr_mean",
        "corner_snr": "corner_clean_snr_mean",
        "corner_contrast": "corner_clean_contrast_mean",
    }
    sort_key_name = sort_metric_map.get(args.sweep_sort, "corner_clean_snr_mean")
    results_sorted = sorted(
        results,
        key=lambda row: row.get(sort_key_name, float("-inf")),
        reverse=True,
    )

    top_k = max(1, args.sweep_top_k)
    print(f"\nSweep summary (sorted by {sort_key_name}, top {top_k}):")
    for entry in results_sorted[:top_k]:
        edge_act = _fmt_float(entry.get("edge_activation_total_mean"), 3)
        texture_act = _fmt_float(entry.get("texture_activation_total_mean"), 3)
        corner_act = _fmt_float(entry.get("corner_clean_activation_total_mean"), 3)
        texture_lambda_low = entry.get("texture_lambda_low", math.nan)
        texture_lambda_high = entry.get("texture_lambda_high", math.nan)
        texture_thr = entry.get("texture_threshold", math.nan)
        texture_cov = entry.get("texture_max_coverage", math.nan)
        print(
            f"  edge_method={entry.get('edge_method', 'sobel')} texture_method={entry.get('texture_method', 'none')} "
            f"edge λ_L={entry['edge_lambda_low']:.4g}, edge λ_H={entry['edge_lambda_high']:.4g}, edge_thr={entry['edge_threshold']:.3f}, edge_cov={entry['edge_max_coverage']:.3f}, "
            f"texture λ_L={texture_lambda_low:.4g}, texture λ_H={texture_lambda_high:.4g}, texture_thr={texture_thr:.3f}, texture_cov={texture_cov:.3f}, "
            f"corner_eta={entry['corner_eta_ratio']:.3f}, corner_thr={entry['corner_mask_threshold']:.3f}, corner_cov={entry['corner_min_coverage']:.3f} | "
            f"edge_snr={_fmt_float(entry.get('edge_snr_mean'), 3)} texture_snr={_fmt_float(entry.get('texture_snr_mean'), 3)} corner_snr={_fmt_float(entry.get('corner_clean_snr_mean'), 3)} "
            f"edge_act={edge_act} texture_act={texture_act} corner_act={corner_act}"
        )

    if args.sweep_output:
        output_path = Path(args.sweep_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "edge_lambda_low",
            "edge_lambda_high",
            "edge_threshold",
            "edge_max_coverage",
            "edge_use_log",
            "edge_method",
            "texture_lambda_low",
            "texture_lambda_high",
            "texture_threshold",
            "texture_max_coverage",
            "texture_use_log",
            "texture_method",
            "corner_eta_ratio",
            "corner_mask_threshold",
            "corner_min_coverage",
            "sample_count",
            "edge_threshold_used_mean",
            "edge_threshold_used_std",
            "edge_threshold_used_min",
            "edge_threshold_used_max",
            "edge_contrast_mean",
            "edge_contrast_std",
            "edge_snr_mean",
            "edge_snr_std",
            "edge_activation_total_mean",
            "edge_activation_total_std",
            "edge_activation_inside_mean",
            "edge_activation_inside_std",
            "edge_activation_outside_mean",
            "edge_activation_outside_std",
            "texture_threshold_used_mean",
            "texture_threshold_used_std",
            "texture_threshold_used_min",
            "texture_threshold_used_max",
            "texture_contrast_mean",
            "texture_contrast_std",
            "texture_snr_mean",
            "texture_snr_std",
            "texture_activation_total_mean",
            "texture_activation_total_std",
            "texture_activation_inside_mean",
            "texture_activation_inside_std",
            "texture_activation_outside_mean",
            "texture_activation_outside_std",
            "corner_resp_contrast_mean",
            "corner_resp_contrast_std",
            "corner_resp_snr_mean",
            "corner_resp_snr_std",
            "corner_clean_contrast_mean",
            "corner_clean_contrast_std",
            "corner_clean_snr_mean",
            "corner_clean_snr_std",
            "corner_clean_activation_total_mean",
            "corner_clean_activation_total_std",
            "corner_clean_activation_inside_mean",
            "corner_clean_activation_inside_std",
            "corner_clean_activation_outside_mean",
            "corner_clean_activation_outside_std",
            "corner_raw_activation_total_mean",
            "corner_raw_activation_total_std",
            "corner_raw_activation_inside_mean",
            "corner_raw_activation_inside_std",
            "corner_raw_activation_outside_mean",
            "corner_raw_activation_outside_std",
            "bbox_ratio_mean",
            "bbox_ratio_std",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_sorted:
                writer.writerow({key: row.get(key, math.nan) for key in fieldnames})

        print(f"Sweep results saved to {output_path}")

    return results_sorted


def _process_root(
    root: Path,
    args: argparse.Namespace,
    label: str | None = None,
) -> list[dict[str, float | int]]:
    prefix = f"[{label}] " if label else ""

    selected_paths = select_sample_paths(
        root,
        args.limit,
        args.sample_mode,
        args.sample_stride,
        args.sample_per_class,
        args.random_seed,
    )

    sample_entries: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []
    for fp in selected_paths:
        image, spec, bboxes = load_sample(fp)
        sample_entries.append((fp, image, spec, bboxes))

    if sample_entries:
        unique_classes = {fp.parent.name for fp, *_ in sample_entries}
        print(
            f"{prefix}Selected {len(sample_entries)} samples across {len(unique_classes)} classes (mode={args.sample_mode}, per_class={args.sample_per_class})."
        )
    else:
        print(f"{prefix}No samples processed. Check --npz-root path or ensure files contain image key.")
        return []

    sweep_requested = any(
        getattr(args, name)
        for name in (
            "sweep_edge_lambda_low",
            "sweep_edge_lambda_high",
            "sweep_edge_threshold",
            "sweep_edge_max_coverage",
            "sweep_edge_method",
            "sweep_texture_lambda_low",
            "sweep_texture_lambda_high",
            "sweep_texture_threshold",
            "sweep_texture_max_coverage",
            "sweep_texture_method",
            "sweep_corner_eta_ratio",
            "sweep_corner_mask_threshold",
            "sweep_corner_min_coverage",
        )
    )
    if args.sweep_output and not sweep_requested:
        sweep_requested = True

    if sweep_requested:
        results = run_parameter_sweep(sample_entries, args)
        return results or []

    out_dir = Path("analysis_output")
    if label:
        out_dir = out_dir / label
    for fp, image, spec, bboxes in sample_entries:
        print(f"{prefix}sample: {fp.name}")
        summarize(image, "image")
        summarize(spec, "spec_512")
        spec_norm = percentile_normalize(np.clip(spec.astype(np.float32), 0.0, None), args.clip_low, args.clip_high)
        summarize(spec_norm, f"spec_norm[{args.clip_low},{args.clip_high}]")

        visualize(
            fp.stem,
            np.clip(image, 0, 255).astype(np.uint8),
            spec,
            bboxes,
            out_dir,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            edge_delta=args.edge_delta,
            edge_lambda_low=args.edge_lambda_low,
            edge_lambda_high=args.edge_lambda_high,
            edge_threshold=args.edge_threshold,
            corner_mask_threshold=args.corner_mask_threshold,
            edge_max_coverage=args.edge_max_coverage,
            corner_min_coverage=args.corner_min_coverage,
            corner_gaussian_radius=args.corner_gaussian_radius,
            corner_gamma=args.corner_gamma,
            corner_eta_ratio=args.corner_eta_ratio,
            create_plot=not args.no_plot,
            edge_use_log=args.edge_use_log,
            edge_method=args.edge_method,
            texture_method=args.texture_method,
            texture_use_log=args.texture_use_log,
            texture_lambda_low=args.texture_lambda_low,
            texture_lambda_high=args.texture_lambda_high,
            texture_threshold=args.texture_threshold,
            texture_max_coverage=args.texture_max_coverage,
            texture_radius=args.texture_radius,
        )

    return []

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-root", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=3, help="Number of samples to inspect after sampling (0 for all).")
    ap.add_argument(
        "--sample-mode",
        choices=["head", "random", "stride"],
        default="head",
        help="How to pick samples: head=sorted order, random=shuffle with optional seed, stride=take every Nth file.",
    )
    ap.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Stride used when --sample-mode=stride (>=1).",
    )
    ap.add_argument(
        "--sample-per-class",
        type=int,
        default=0,
        help="Limit samples per class folder (0 disables per-class limit).",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling when --sample-mode=random.",
    )
    ap.add_argument("--clip-low", type=float, default=1.0, help="Lower percentile for normalization.")
    ap.add_argument("--clip-high", type=float, default=99.5, help="Upper percentile for normalization.")
    ap.add_argument(
        "--edge-delta",
        type=int,
        default=3,
        help="Neighborhood/scale radius δ (used for Sobel NMS, high-pass blur, Laplacian kernel, or std window).",
    )
    ap.add_argument(
        "--edge-lambda-low",
        type=float,
        default=0.2,
        help="Lower threshold λ_L. <1 interpreted as ratio of max gradient; otherwise absolute.",
    )
    ap.add_argument(
        "--edge-lambda-high",
        type=float,
        default=0.4,
        help="Upper threshold λ_H. <1 interpreted as ratio of max gradient; otherwise absolute.",
    )
    ap.add_argument("--edge-threshold", type=float, default=0.5, help="Activation threshold for edge map (0-1).")
    ap.add_argument("--edge-max-coverage", type=float, default=0.1, help="Maximum allowed edge activation ratio (0-1).")
    ap.add_argument("--edge-use-log", action="store_true", help="Use log-normalized channel as edge detector input.")
    ap.add_argument(
        "--edge-method",
        choices=EDGE_METHOD_CHOICES,
        default="sobel",
        help="Edge extraction method: sobel, log-hp, log-hp-raw, laplacian, window-std, tophat, canny, gabor, or local-entropy.",
    )
    ap.add_argument(
        "--texture-method",
        choices=TEXTURE_METHOD_CHOICES,
        default="none",
        help="Texture descriptor: none, log-hp, laplacian, window-std, tophat, canny, gabor, or local-entropy.",
    )
    ap.add_argument(
        "--texture-use-log",
        action="store_true",
        help="Use log-normalized channel as texture input.",
    )
    ap.add_argument(
        "--texture-lambda-low",
        type=float,
        default=0.0,
        help="Texture lower threshold λ_L (<1 treated as ratio).",
    )
    ap.add_argument(
        "--texture-lambda-high",
        type=float,
        default=1.0,
        help="Texture upper threshold λ_H (<1 treated as ratio).",
    )
    ap.add_argument(
        "--texture-threshold",
        type=float,
        default=0.5,
        help="Activation threshold for texture map (0-1).",
    )
    ap.add_argument(
        "--texture-max-coverage",
        type=float,
        default=0.1,
        help="Maximum allowed texture activation ratio (0-1).",
    )
    ap.add_argument(
        "--texture-radius",
        type=int,
        default=3,
        help="Neighborhood radius for texture filters (log-hp blur, std window, tophat structuring element, Canny pre-blur/aperture, Gabor kernel, or local entropy window).",
    )
    ap.add_argument("--corner-gaussian-radius", type=int, default=2, help="Gaussian window radius |W| for corner response.")
    ap.add_argument("--corner-gamma", type=float, default=0.05, help="Harris-like γ parameter.")
    ap.add_argument("--corner-eta-ratio", type=float, default=0.01, help="η ratio relative to max corner response.")
    ap.add_argument("--corner-mask-threshold", type=float, default=0.5, help="Threshold for corner activation mask (0-1).")
    ap.add_argument("--corner-min-coverage", type=float, default=0.01, help="Minimum coverage enforced for corner mask (0-1).")
    ap.add_argument("--no-plot", action="store_true", help="Skip saving visualization figures.")
    ap.add_argument("--sweep-edge-lambda-low", type=float, nargs="+", help="λ_L values to sweep (ratios or absolutes).")
    ap.add_argument("--sweep-edge-lambda-high", type=float, nargs="+", help="λ_H values to sweep (ratios or absolutes).")
    ap.add_argument("--sweep-edge-threshold", type=float, nargs="+", help="Edge activation thresholds to sweep (0-1).")
    ap.add_argument(
        "--sweep-edge-max-coverage",
        type=float,
        nargs="+",
        help="Edge activation coverage caps to sweep (0-1).",
    )
    ap.add_argument(
        "--sweep-edge-method",
        choices=EDGE_METHOD_CHOICES,
        nargs="+",
        help="Edge extraction methods to sweep (defaults to the selected --edge-method).",
    )
    ap.add_argument("--sweep-texture-lambda-low", type=float, nargs="+", help="Texture λ_L sweep values.")
    ap.add_argument("--sweep-texture-lambda-high", type=float, nargs="+", help="Texture λ_H sweep values.")
    ap.add_argument("--sweep-texture-threshold", type=float, nargs="+", help="Texture activation thresholds (0-1).")
    ap.add_argument(
        "--sweep-texture-max-coverage",
        type=float,
        nargs="+",
        help="Texture activation coverage caps (0-1).",
    )
    ap.add_argument(
        "--sweep-texture-method",
        choices=TEXTURE_METHOD_CHOICES,
        nargs="+",
        help="Texture descriptor methods to sweep (defaults to the selected --texture-method).",
    )
    ap.add_argument("--sweep-corner-eta-ratio", type=float, nargs="+", help="Corner η ratio sweep values.")
    ap.add_argument(
        "--sweep-corner-mask-threshold",
        type=float,
        nargs="+",
        help="Corner mask activation thresholds to sweep (0-1).",
    )
    ap.add_argument(
        "--sweep-corner-min-coverage",
        type=float,
        nargs="+",
        help="Corner minimum coverage constraints to sweep (0-1).",
    )
    ap.add_argument(
        "--sweep-max-combos",
        type=int,
        default=0,
        help="Maximum number of sweep combinations to evaluate (0 for no limit).",
    )
    ap.add_argument(
        "--sweep-randomize",
        action="store_true",
        help="Shuffle sweep combinations before applying --sweep-max-combos.",
    )
    ap.add_argument(
        "--sweep-seed",
        type=int,
        default=None,
        help="Random seed used when --sweep-randomize is set (falls back to --random-seed).",
    )
    ap.add_argument(
        "--batch-subdirs",
        action="store_true",
        help="Run the analysis separately for each immediate subdirectory under --npz-root.",
    )
    ap.add_argument(
        "--batch-output-dir",
        type=Path,
        help="Directory for per-group sweep CSV exports when --batch-subdirs is enabled.",
    )
    ap.add_argument(
        "--batch-summary",
        type=Path,
        help="Optional CSV path to store the top sweep entry for each group when using --batch-subdirs.",
    )
    ap.add_argument("--sweep-output", type=Path, help="Optional CSV path for storing sweep results.")
    ap.add_argument(
        "--sweep-sort",
        choices=["edge_snr", "texture_snr", "corner_snr", "corner_contrast"],
        default="corner_snr",
        help="Metric used to sort sweep summary output.",
    )
    ap.add_argument("--sweep-top-k", type=int, default=10, help="Show top-K sweep configurations (default 10).")
    args = ap.parse_args()

    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")

    if args.clip_low >= args.clip_high:
        raise SystemExit("--clip-low 必须小于 --clip-high")

    args.edge_threshold = float(min(max(args.edge_threshold, 0.0), 1.0))
    args.edge_max_coverage = float(min(max(args.edge_max_coverage, 0.0), 1.0))
    args.corner_mask_threshold = float(min(max(args.corner_mask_threshold, 0.0), 1.0))
    args.corner_min_coverage = float(min(max(args.corner_min_coverage, 0.0), 1.0))
    args.sample_mode = (args.sample_mode or "head").lower()
    args.sample_stride = max(1, int(args.sample_stride))
    args.sample_per_class = max(0, int(args.sample_per_class))
    args.edge_delta = max(1, args.edge_delta)
    args.edge_lambda_low = max(0.0, args.edge_lambda_low)
    args.edge_lambda_high = max(args.edge_lambda_low + 1e-6, args.edge_lambda_high)
    args.corner_gaussian_radius = max(1, args.corner_gaussian_radius)
    args.corner_gamma = max(0.0, args.corner_gamma)
    args.corner_eta_ratio = max(0.0, args.corner_eta_ratio)
    args.texture_threshold = float(min(max(args.texture_threshold, 0.0), 1.0))
    args.texture_max_coverage = float(min(max(args.texture_max_coverage, 0.0), 1.0))
    args.texture_lambda_low = max(0.0, args.texture_lambda_low)
    args.texture_lambda_high = max(args.texture_lambda_low + 1e-6, args.texture_lambda_high)
    args.texture_radius = max(1, int(args.texture_radius))
    args.sweep_top_k = max(1, args.sweep_top_k)
    args.sweep_max_combos = max(0, int(args.sweep_max_combos))
    args.edge_method = (args.edge_method or "sobel").lower()
    if args.sweep_edge_method:
        args.sweep_edge_method = [m.lower() for m in args.sweep_edge_method]
    args.texture_method = (args.texture_method or "none").lower()
    if args.texture_method not in TEXTURE_METHOD_CHOICES:
        args.texture_method = "none"
    if args.sweep_texture_method:
        args.sweep_texture_method = [m.lower() for m in args.sweep_texture_method]

    if args.batch_subdirs:
        root_dir = args.npz_root
        if not root_dir.is_dir():
            raise SystemExit("--npz-root 必须是文件夹，当使用 --batch-subdirs 时。")

        subdirs = sorted(p for p in root_dir.iterdir() if p.is_dir())
        if not subdirs:
            print("未在 --npz-root 下找到子文件夹，无法执行 batch 分析。")
            return

        summary_rows: list[dict[str, float | int]] = []
        batch_output_dir = args.batch_output_dir
        if batch_output_dir:
            batch_output_dir.mkdir(parents=True, exist_ok=True)

        for subdir in subdirs:
            group_label = subdir.name
            print(f"\n=== Group {group_label} ===")
            group_args = argparse.Namespace(**vars(args))
            group_args.npz_root = subdir

            if batch_output_dir:
                group_args.sweep_output = batch_output_dir / f"{group_label}.csv"
            elif args.sweep_output:
                base = Path(args.sweep_output)
                group_args.sweep_output = base.with_name(f"{base.stem}_{group_label}{base.suffix}")

            results = _process_root(subdir, group_args, label=group_label)
            if results:
                top_entry = dict(results[0])
                top_entry["group"] = group_label
                summary_rows.append(top_entry)

        if args.batch_summary and summary_rows:
            summary_path = Path(args.batch_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            ordered_keys: list[str] = []
            for row in summary_rows:
                for key in row.keys():
                    if key not in ordered_keys:
                        ordered_keys.append(key)
            if "group" in ordered_keys:
                ordered_keys.remove("group")
            fieldnames = ["group"] + ordered_keys

            with summary_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow({key: row.get(key, math.nan) for key in fieldnames})

            print(f"Batch summary saved to {summary_path}")
        return

    _process_root(args.npz_root, args)


if __name__ == "__main__":
    main()
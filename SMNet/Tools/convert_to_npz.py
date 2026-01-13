"""Convert 512×512 single-channel spectrogram NPZ files into detection-ready bundles.

Each output NPZ stores:
    - `image`: 512×512×3 uint8 pseudo-color preview (log, edge, corner)
  - `mask`: 512×512 uint8 label mask (all zeros when no shapes)
  - `bboxes`: N×5 array `[x1, y1, x2, y2, class]`
  - `label_json`: path to the source JSON (empty string if missing)
  - `class_id`: integer class id copied from directory name
  - `source_npz`: original NPZ path (string)
    - `golden_triplet`: 3×512×512 float32 stack `[log_gray_norm, gray_norm, corner_mask_clean]` bounded to [0, 1]

The script accepts command-line arguments to choose class ranges, stem ranges, PNG
fallback masks, and writes a health report with per-class stats.
"""

from __future__ import annotations

import argparse
import math
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import cv2


DEFAULT_SOURCE_ROOT = Path("D:/Exp/Data-512NPZ")
DEFAULT_LABEL_ROOT = Path("D:/Exp/FCSLabel")
DEFAULT_OUTPUT_ROOT = Path("D:/Exp/SMNet/FCSData")


@dataclass
class SampleResult:
    class_id: int
    stem: str
    written: bool
    used_png_fallback: bool
    boxes: int
    label_missing: bool
    golden_min: float = 0.0
    golden_max: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class FeatureParams:
    clip_low: float = 1.0
    clip_high: float = 99.5
    edge_delta: int = 3
    edge_lambda_low: float = 0.2
    edge_lambda_high: float = 0.4
    edge_use_log: bool = True
    corner_gaussian_radius: int = 2
    corner_gamma: float = 0.05
    corner_eta_ratio: float = 0.01
    corner_min_coverage: float = 0.01


FEATURE_PARAMS = FeatureParams()


def percentile_normalize(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    lo = float(np.percentile(data, low))
    hi = float(np.percentile(data, high))
    if hi - lo < 1e-6:
        if data.max() > 1e-6:
            return (data / data.max()).astype(np.float32)
        return np.zeros_like(data, dtype=np.float32)
    clipped = np.clip(data, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def _ensure_kernel(sz: int) -> int:
    sz = int(max(1, sz))
    if sz % 2 == 0:
        sz += 1
    return sz


def log_rescale(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, None).astype(np.float32)
    return np.log1p(arr)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    data = data - float(data.min())
    max_val = float(data.max())
    if max_val > 1e-6:
        data = data / max_val
    return np.clip(data * 255.0, 0.0, 255.0).astype(np.uint8)


def ensure_unit_interval(name: str, arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    min_val = float(data.min())
    max_val = float(data.max())
    if min_val < -1e-5 or max_val > 1.0 + 1e-5:
        raise ValueError(f"{name} must be within [0, 1], got range [{min_val:.4f}, {max_val:.4f}]")
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def compute_edges_sobel(gray_uint8: np.ndarray, delta: int, lambda_low: float, lambda_high: float) -> Tuple[np.ndarray, np.ndarray]:
    gray = gray_uint8.astype(np.float32)
    kernel_t = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_f = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    g_t = cv2.filter2D(gray, cv2.CV_32F, kernel_t, borderType=cv2.BORDER_REFLECT)
    g_f = cv2.filter2D(gray, cv2.CV_32F, kernel_f, borderType=cv2.BORDER_REFLECT)
    mag = cv2.magnitude(g_t, g_f)
    if mag.size == 0:
        zeros = np.zeros_like(gray, dtype=np.float32)
        return zeros, zeros

    max_mag = float(mag.max())
    if max_mag <= 1e-6:
        zeros = np.zeros_like(gray, dtype=np.float32)
        return zeros, zeros

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

    hat_max = float(hat_g.max())
    edge_raw = (hat_g / hat_max).astype(np.float32) if hat_max > 1e-6 else np.zeros_like(hat_g, dtype=np.float32)
    return edge_raw.astype(np.float32), s_edge.astype(np.float32)


def compute_corners_harris(
    gray_uint8: np.ndarray,
    gaussian_radius: int,
    gamma: float,
    eta_ratio: float,
    min_coverage: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    max_r = float(r.max())
    if max_r <= 1e-6:
        zeros = np.zeros_like(r, dtype=np.float32)
        return zeros, zeros, zeros

    r_norm = (r / max_r).astype(np.float32)
    eta = eta_ratio * max_r
    mask_raw = (r >= eta).astype(np.uint8)
    if min_coverage > 0 and mask_raw.mean() < min_coverage:
        required_percentile = max(0.0, 100.0 * (1.0 - min_coverage))
        thr = float(np.percentile(r, required_percentile))
        mask_raw = (r >= thr).astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.dilate(mask_clean, kernel)
    mask_clean = cv2.medianBlur(mask_clean, 3)
    if min_coverage > 0 and mask_clean.mean() < min_coverage:
        mask_clean = mask_raw

    return r_norm, mask_raw.astype(np.float32), mask_clean.astype(np.float32)


def compute_feature_maps(spec: np.ndarray, params: FeatureParams) -> Dict[str, np.ndarray | object]:
    """Generate the feature stack used by SMNet training."""
    gray_raw = np.clip(spec.astype(np.float32), 0.0, None)
    gray_norm = percentile_normalize(gray_raw, params.clip_low, params.clip_high)
    gray_u8 = to_uint8(gray_raw)

    log_gray = log_rescale(gray_raw)
    log_gray_norm = percentile_normalize(log_gray, params.clip_low, params.clip_high)

    edge_input = log_gray_norm if params.edge_use_log else gray_norm
    edge_raw, edge_clean = compute_edges_sobel(to_uint8(edge_input), params.edge_delta, params.edge_lambda_low, params.edge_lambda_high)
    edge_clean = np.clip(edge_clean, 0.0, 1.0).astype(np.float32)

    _corner_resp, corner_mask_raw, corner_mask_clean = compute_corners_harris(
        gray_u8,
        gaussian_radius=params.corner_gaussian_radius,
        gamma=params.corner_gamma,
        eta_ratio=params.corner_eta_ratio,
        min_coverage=params.corner_min_coverage,
    )
    corner_mask_clean = np.clip(corner_mask_clean, 0.0, 1.0).astype(np.float32)

    log_gray_norm = np.clip(log_gray_norm, 0.0, 1.0).astype(np.float32)
    pseudo_rgb = np.stack([log_gray_norm, edge_clean, corner_mask_clean], axis=-1).astype(np.float32)
    pseudo_rgb_u8 = to_uint8(pseudo_rgb)

    feature_stack = np.concatenate(
        [
            pseudo_rgb.transpose(2, 0, 1).astype(np.float32),
            gray_norm[None, ...].astype(np.float32),
            corner_mask_clean[None, ...],
        ],
        axis=0,
    )

    feature_names = np.array(
        ["pseudo_rgb_r", "pseudo_rgb_g", "pseudo_rgb_b", "gray_norm", "corner_mask"],
        dtype="<U16",
    )

    return {
        "gray_norm": gray_norm.astype(np.float32),
        "gray_norm_u8": to_uint8(gray_norm),
        "log_gray_norm": log_gray_norm,
        "edge_clean": edge_clean,
        "corner_mask_raw": corner_mask_raw.astype(np.float32),
        "corner_mask_clean": corner_mask_clean,
        "pseudo_rgb": pseudo_rgb,
        "pseudo_rgb_u8": pseudo_rgb_u8,
        "feature_stack": feature_stack,
        "feature_names": feature_names,
        "params_json": json.dumps(asdict(params)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="Root directory containing class-id subdirectories with NPZ files.")
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT, help="Root directory containing label JSON/PNG organised by class.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Destination root for converted NPZ files.")
    parser.add_argument("--classes", type=int, nargs="*", help="Optional list of class ids to convert. Default converts every numeric subdirectory.")
    parser.add_argument("--stem-min", type=int, default=None, help="Minimum leading numeric stem to include (inclusive).")
    parser.add_argument("--stem-max", type=int, default=None, help="Maximum leading numeric stem to include (inclusive).")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap on number of files converted per class after filtering.")
    parser.add_argument("--label-suffix", type=str, default=".json", help="Suffix appended to the sample stem when locating JSON labels.")
    parser.add_argument("--mask-suffix", type=str, default=".png", help="Suffix appended to the sample stem when loading PNG masks.")
    parser.add_argument("--use-png-mask-fallback", action="store_true", help="If set, load PNG masks when JSON shapes are missing or empty.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output NPZ files instead of skipping them.")
    parser.add_argument("--health-json-name", type=str, default="npz_health_report.json", help="Filename for the generated health report under the output root.")
    parser.add_argument("--print-every", type=int, default=100, help="Progress log interval measured in samples.")
    return parser.parse_args()


def iter_source_npz(
    source_root: Path,
    classes: Iterable[int] | None,
    stem_min: int | None,
    stem_max: int | None,
    max_per_class: int | None,
) -> Iterator[Tuple[int, Path]]:
    selected = {int(c) for c in classes} if classes else None
    for class_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        if selected is not None and class_id not in selected:
            continue
        emitted = 0
        for npz_path in sorted(class_dir.glob("*.npz")):
            stem = npz_path.stem
            lead = extract_leading_int(stem)
            if stem_min is not None and (lead is None or lead < stem_min):
                continue
            if stem_max is not None and (lead is None or lead > stem_max):
                continue
            yield class_id, npz_path
            emitted += 1
            if max_per_class is not None and emitted >= max_per_class:
                break


def extract_leading_int(stem: str) -> int | None:
    digits: List[str] = []
    for ch in stem:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else None


def load_source_spec(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as bundle:
        if "spec_512" not in bundle:
            raise KeyError(f"spec_512 missing in {npz_path}")
        spec = bundle["spec_512"].astype(np.float32)
    if spec.ndim == 3:
        if spec.shape[0] == 1:
            spec = spec[0]
        elif spec.shape[-1] == 1:
            spec = spec[..., 0]
    if spec.shape != (512, 512):
        raise ValueError(f"spec_512 must be 512x512 but got {spec.shape} in {npz_path}")
    return spec


def convert_spec_to_image(spec: np.ndarray) -> np.ndarray:
    scaled = np.clip(spec * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.repeat(scaled[..., None], 3, axis=2)


def parse_label_value(raw: object, default: int) -> int:
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, str):
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            try:
                return int(digits)
            except ValueError:
                pass
    return default


def _ensure_bounds(value: float, lower: int, upper: int) -> int:
    return int(min(max(value, lower), upper))


def mask_from_shapes(shapes: List[dict], default_class: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask_img = None
    draw = None

    for shp in shapes:
        pts = shp.get("points")
        if not (isinstance(pts, list) and len(pts) >= 2):
            continue
        cls_val = max(0, parse_label_value(shp.get("label"), default_class))
        shape_type = (shp.get("shape_type") or "rectangle").lower()
        if shape_type == "rectangle":
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            x_min = math.floor(min(x0, x1))
            x_max = math.ceil(max(x0, x1))
            y_min = math.floor(min(y0, y1))
            y_max = math.ceil(max(y0, y1))
            x_min = _ensure_bounds(x_min, 0, width - 1)
            x_max = _ensure_bounds(x_max, 0, width)
            y_min = _ensure_bounds(y_min, 0, height - 1)
            y_max = _ensure_bounds(y_max, 0, height)
            if x_max <= x_min:
                x_max = min(width, x_min + 1)
            if y_max <= y_min:
                y_max = min(height, y_min + 1)
            mask[y_min:y_max, x_min:x_max] = int(cls_val)
        else:
            if draw is None:
                mask_img = Image.fromarray(mask)
                draw = ImageDraw.Draw(mask_img)
            poly = [(float(x), float(y)) for x, y in pts]
            draw.polygon(poly, fill=int(cls_val))

    if mask_img is not None:
        mask = np.asarray(mask_img, dtype=np.uint8)
    return mask


def boxes_from_shapes(shapes: List[dict], default_class: int, width: int, height: int) -> np.ndarray:
    boxes: List[List[float]] = []
    for shp in shapes:
        pts = shp.get("points")
        if not (isinstance(pts, list) and len(pts) >= 2):
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_min = max(0.0, min(float(width), x_min))
        x_max = max(0.0, min(float(width), x_max))
        y_min = max(0.0, min(float(height), y_min))
        y_max = max(0.0, min(float(height), y_max))
        if x_max <= x_min:
            x_max = min(float(width), x_min + 1.0)
        if y_max <= y_min:
            y_max = min(float(height), y_min + 1.0)
        if x_max <= x_min or y_max <= y_min:
            continue
        cls_val = parse_label_value(shp.get("label"), default_class)
        boxes.append([x_min, y_min, x_max, y_max, float(cls_val)])
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def load_png_mask(mask_path: Path) -> np.ndarray:
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def boxes_from_mask(mask: np.ndarray, class_id: int) -> np.ndarray:
    indices = np.nonzero(mask)
    if not indices[0].size:
        return np.zeros((0, 5), dtype=np.float32)
    ys, xs = indices
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    if x_max <= x_min or y_max <= y_min:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray([[float(x_min), float(y_min), float(x_max + 1), float(y_max + 1), float(class_id)]], dtype=np.float32)


def process_single(
    class_id: int,
    source_npz: Path,
    label_root: Path,
    output_root: Path,
    label_suffix: str,
    mask_suffix: str,
    use_png_fallback: bool,
    overwrite: bool,
) -> SampleResult:
    stem = source_npz.stem
    class_dir_name = f"{class_id}"
    out_dir = output_root / class_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.npz"
    if out_path.exists() and not overwrite:
        return SampleResult(class_id, stem, written=False, used_png_fallback=False, boxes=0, label_missing=False)

    label_path = label_root / class_dir_name / f"{stem}{label_suffix}"
    mask_path = label_root / class_dir_name / f"{stem}{mask_suffix}"
    label_exists = label_path.exists()
    try:
        spec = load_source_spec(source_npz)
        feature_maps = compute_feature_maps(spec, FEATURE_PARAMS)
        image = feature_maps["pseudo_rgb_u8"]
        shapes: List[dict] = []
        if label_exists:
            data = json.loads(label_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("shapes"), list):
                shapes = data["shapes"]
        boxes = boxes_from_shapes(shapes, class_id, image.shape[1], image.shape[0]) if shapes else np.zeros((0, 5), dtype=np.float32)
        mask = mask_from_shapes(shapes, class_id, image.shape[0], image.shape[1]) if shapes else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        used_png = False
        if use_png_fallback and (not shapes or not boxes.size) and mask_path.exists():
            mask = load_png_mask(mask_path)
            used_png = True
            if not boxes.size:
                boxes = boxes_from_mask(mask, class_id)
        log_gray_norm = ensure_unit_interval("log_gray_norm", feature_maps["log_gray_norm"])
        gray_norm = ensure_unit_interval("gray_norm", feature_maps["gray_norm"])
        corner_mask_clean = ensure_unit_interval("corner_mask_clean", feature_maps["corner_mask_clean"])
        golden_triplet = np.stack([log_gray_norm, gray_norm, corner_mask_clean], axis=0).astype(np.float32)
        golden_triplet_u8 = np.clip(golden_triplet.transpose(1, 2, 0) * 255.0, 0.0, 255.0).astype(np.uint8)
        golden_min = float(golden_triplet.min())
        golden_max = float(golden_triplet.max())
        label_json_value = label_path.read_text(encoding="utf-8") if label_exists else ""
        np.savez_compressed(
            out_path,
            image=image,
            mask=mask,
            bboxes=boxes,
            label_json=np.array(label_json_value),
            class_id=np.int32(class_id),
            source_npz=np.array(str(source_npz)),
            features=feature_maps["feature_stack"],
            feature_names=feature_maps["feature_names"],
            feature_params=np.array(feature_maps["params_json"]),
            pseudo_rgb=feature_maps["pseudo_rgb"],
            pseudo_rgb_u8=feature_maps["pseudo_rgb_u8"],
            gray_norm=gray_norm,
            corner_mask_clean=corner_mask_clean,
            golden_triplet=golden_triplet,
            golden_triplet_u8=golden_triplet_u8,
        )
        return SampleResult(
            class_id=class_id,
            stem=stem,
            written=True,
            used_png_fallback=used_png,
            boxes=int(boxes.shape[0]),
            label_missing=not label_exists,
            golden_min=golden_min,
            golden_max=golden_max,
        )
    except Exception as exc:  # noqa: BLE001
        return SampleResult(
            class_id=class_id,
            stem=stem,
            written=False,
            used_png_fallback=False,
            boxes=0,
            label_missing=not label_exists,
            error=str(exc),
        )


def summarize(
    results: List[SampleResult],
    output_root: Path,
    health_filename: str,
    source_root: Path,
    label_root: Path,
) -> None:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"source": 0, "converted": 0, "with_boxes": 0, "boxes": 0})
    golden_min = float("inf")
    golden_max = float("-inf")
    report: Dict[str, object] = {
        "source_root": str(source_root.resolve()),
        "label_root": str(label_root.resolve()),
        "output_root": str(output_root.resolve()),
        "total_samples": len(results),
        "converted": 0,
        "skipped_existing": 0,
        "missing_label_json": 0,
        "empty_boxes": 0,
        "png_fallback_used": 0,
        "errors": [],
        "per_class": {},
        "golden_triplet_range": {"min": None, "max": None},
    }
    for r in results:
        key = str(r.class_id)
        cls_stats = stats[key]
        cls_stats["source"] += 1
        if r.written:
            report["converted"] += 1
            cls_stats["converted"] += 1
        else:
            report["skipped_existing"] += 1
        if r.boxes > 0:
            cls_stats["with_boxes"] += 1
            cls_stats["boxes"] += r.boxes
        else:
            report["empty_boxes"] += 1
        if r.label_missing:
            report["missing_label_json"] += 1
        if r.used_png_fallback:
            report["png_fallback_used"] += 1
        if r.error:
            report["errors"].append({"class": r.class_id, "stem": r.stem, "message": r.error})
        if r.written:
            golden_min = min(golden_min, r.golden_min)
            golden_max = max(golden_max, r.golden_max)
    if golden_min != float("inf"):
        report["golden_triplet_range"] = {"min": golden_min, "max": golden_max}
    report["per_class"] = stats
    health_path = output_root / health_filename
    health_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote health report to {health_path}")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    label_root = args.label_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.exists():
        raise SystemExit(f"Source root not found: {source_root}")
    if not label_root.exists():
        raise SystemExit(f"Label root not found: {label_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[SampleResult] = []
    for idx, (class_id, npz_path) in enumerate(
        iter_source_npz(source_root, args.classes, args.stem_min, args.stem_max, args.max_per_class),
        start=1,
    ):
        res = process_single(
            class_id=class_id,
            source_npz=npz_path,
            label_root=label_root,
            output_root=output_root,
            label_suffix=args.label_suffix,
            mask_suffix=args.mask_suffix,
            use_png_fallback=args.use_png_mask_fallback,
            overwrite=args.overwrite,
        )
        results.append(res)
        if args.print_every and idx % args.print_every == 0:
            print(
                f"Processed {idx} samples... (class {class_id}, stem {res.stem}, "
                f"written={res.written}, boxes={res.boxes})"
            )
    summarize(results, output_root, args.health_json_name, source_root, label_root)
    converted = sum(1 for r in results if r.written)
    errors = [r for r in results if r.error]
    print(f"Done. Converted {converted}/{len(results)} samples. Errors: {len(errors)}")
    if errors:
        for err in errors[:10]:
            print(f"  [error] class={err.class_id} stem={err.stem}: {err.error}")


if __name__ == "__main__":
    main()


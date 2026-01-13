import argparse
import json
from pathlib import Path
import numpy as np

POSSIBLE_META_KEYS = ["bandwidth_mhz", "bandwidth", "BW_MHz", "bw_mhz", "span_mhz", "bandwidth_MHz"]


def _parse_bandwidth_to_mhz(val):
    """解析带宽值为 MHz，支持数值或带单位字符串（hz/khz/mhz）。"""
    import re
    try:
        if hasattr(val, 'item'):
            val = val.item()
        if isinstance(val, (int, float, np.number)):
            v = float(val)
            if not np.isfinite(v) or v <= 0:
                return None
            return v / 1e6 if v > 1e4 else v
        s = str(val).strip()
        if not s:
            return None
        sl = s.lower().replace(' ', '')
        m = re.match(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)', sl)
        if not m:
            return None
        num = float(m.group(1))
        if num <= 0 or not np.isfinite(num):
            return None
        if sl.endswith('mhz'):
            return num
        if sl.endswith('khz'):
            return num / 1e3
        if sl.endswith('hz'):
            return num / 1e6
        return num if num <= 1e4 else (num / 1e6)
    except Exception:
        return None


def _try_parse_json(value):
    try:
        if hasattr(value, 'item'):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='ignore')
        if not isinstance(value, str):
            value = str(value)
        return json.loads(value)
    except Exception:
        return None


def resolve_bandwidth_mhz(npz_path: Path, default: float = 100.0):
    src = "default"
    bw = None
    try:
        with np.load(str(npz_path), allow_pickle=True) as d:
            # 1) top-level key
            if "bandwidth_mhz" in d:
                bw = float(d["bandwidth_mhz"]) ; src = "npz:bandwidth_mhz"
            # 2) meta dict-like
            elif "meta" in d:
                meta = d["meta"]
                try:
                    meta = meta.item() if hasattr(meta, "item") else meta
                except Exception:
                    pass
                if isinstance(meta, dict):
                    for k in POSSIBLE_META_KEYS:
                        if k in meta:
                            bw = float(meta[k]) ; src = f"meta:{k}" ; break
            # 3) meta_json string
            if bw is None and "meta_json" in d:
                meta = _try_parse_json(d["meta_json"])
                if isinstance(meta, dict):
                    for k in POSSIBLE_META_KEYS:
                        if k in meta:
                            bw = _parse_bandwidth_to_mhz(meta[k])
                            if bw is not None:
                                src = f"meta_json:{k}"
                                break
            # 4) label_json string
            if bw is None and "label_json" in d:
                lab = _try_parse_json(d["label_json"])
                if isinstance(lab, dict):
                    for k in POSSIBLE_META_KEYS:
                        if k in lab:
                            bw = _parse_bandwidth_to_mhz(lab[k])
                            if bw is not None:
                                src = f"label_json:{k}"
                                break
    except Exception as e:
        return default, f"error:{type(e).__name__}"
    if bw is None or not np.isfinite(bw) or bw <= 0:
        return float(default), src
    return float(bw), src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="path to train_list txt")
    ap.add_argument("--max", type=int, default=20, help="max samples to probe")
    ap.add_argument("--show_missing_only", action="store_true", help="only print when falling back to default")
    ap.add_argument("--default", type=float, default=100.0, help="fallback bandwidth MHz")
    ap.add_argument("--dump", action="store_true", help="dump keys/meta_json/label_json of first sample for debugging")
    args = ap.parse_args()

    paths = [Path(p.strip()) for p in Path(args.list).read_text(encoding="utf-8").splitlines() if p.strip()]

    if args.dump and paths:
        p0 = paths[0]
        try:
            with np.load(str(p0), allow_pickle=True) as d:
                print(f"\n# dump for: {p0}")
                print("keys:", list(d.keys()))
                if "meta_json" in d:
                    meta = _try_parse_json(d["meta_json"]) ; print("meta_json keys:", (list(meta.keys()) if isinstance(meta, dict) else None))
                if "label_json" in d:
                    lab = _try_parse_json(d["label_json"]) ; print("label_json keys:", (list(lab.keys()) if isinstance(lab, dict) else None))
        except Exception as e:
            print("dump error:", e)

    total = 0
    from_npz = 0
    from_meta = 0
    from_meta_json = 0
    from_label_json = 0
    fallback = 0
    errors = 0

    print("# idx | bw_mhz | source | file")
    for i, p in enumerate(paths[: max(0, args.max) ]):
        bw, src = resolve_bandwidth_mhz(p, default=args.default)
        total += 1
        if src.startswith("npz:"):
            from_npz += 1
        elif src.startswith("meta:"):
            from_meta += 1
        elif src.startswith("meta_json:"):
            from_meta_json += 1
        elif src.startswith("label_json:"):
            from_label_json += 1
        elif src.startswith("error:"):
            errors += 1
        else:
            fallback += 1
        if not args.show_missing_only or (src == "default"):
            print(f"{i:3d} | {bw:7.2f} | {src:>18s} | {p.name}")

    summary = {
        "total": total,
        "from_npz": from_npz,
        "from_meta": from_meta,
        "from_meta_json": from_meta_json,
        "from_label_json": from_label_json,
        "fallback_default": fallback,
        "errors": errors,
        "default_used": args.default,
    }
    print("\n# summary")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

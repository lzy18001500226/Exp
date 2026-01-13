import argparse
from pathlib import Path
import urllib.request
import urllib.error
import ssl

# Simple HTTPS context that skips certificate verification (some corp proxies)
CTX = ssl.create_default_context()
try:
    CTX.check_hostname = False
    CTX.verify_mode = ssl.CERT_NONE
except Exception:
    pass

BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"

# Curated useful YOLO weights across v3 -> v11 (det/cls/seg/pose/obb where available)
ASSETS = {
    # v3 unified
    'YOLOv3': [
        'yolov3u.pt',
        'yolov3-tinyu.pt',
        'yolov3-sppu.pt',
    ],
    # v5 u-series (P5 + P6)
    'YOLOv5': [
        'yolov5nu.pt','yolov5su.pt','yolov5mu.pt','yolov5lu.pt','yolov5xu.pt',
        'yolov5n6u.pt','yolov5s6u.pt','yolov5m6u.pt','yolov5l6u.pt','yolov5x6u.pt',
    ],
    # v8 detectors + heads
    'YOLOv8': [
        # det
        'yolov8n.pt','yolov8s.pt','yolov8m.pt','yolov8l.pt','yolov8x.pt',
        # cls
        'yolov8n-cls.pt','yolov8s-cls.pt','yolov8m-cls.pt','yolov8x-cls.pt',
        # seg
        'yolov8n-seg.pt','yolov8s-seg.pt','yolov8m-seg.pt','yolov8l-seg.pt','yolov8x-seg.pt',
        # pose
        'yolov8n-pose.pt','yolov8s-pose.pt','yolov8l-pose.pt','yolov8x-pose.pt',
        # obb
        'yolov8n-obb.pt','yolov8s-obb.pt','yolov8m-obb.pt','yolov8l-obb.pt','yolov8x-obb.pt',
    ],
    # v11 detectors (common ones)
    'YOLOv11': [
        'yolov11n.pt','yolov11s.pt','yolov11m.pt','yolov11l.pt','yolov11x.pt',
        # include one oiv7-trained variant as example
        'yolo11n-oiv7.pt','yolo11s-oiv7.pt','yolo11m-oiv7.pt','yolo11l-oiv7.pt','yolo11x-oiv7.pt',
    ],
}

# Destination subfolders under Detector/Ultralytics
DEST = {
    'YOLOv3': 'Detector/Ultralytics/YOLOv3',
    'YOLOv5': 'Detector/Ultralytics/YOLOv5',
    'YOLOv8': 'Detector/Ultralytics/YOLOv8',
    'YOLOv11': 'Detector/Ultralytics/YOLOv11',
    # Additionally route classification heads to Backbone/Texture as a copy (optional)
}


def download(url: str, out_path: Path) -> tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, context=CTX) as r, open(out_path, 'wb') as f:
            CHUNK = 1 << 20
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                f.write(b)
        return True, f"OK {out_path.name}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {url}"
    except Exception as e:
        return False, f"FAIL: {url} -> {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-r', '--root', default=r'C:/Users/HP/Desktop/Exp/Model/Pretrain', help='Pretrain root dir')
    p.add_argument('--also_copy_cls_to', default=r'Backbone/Texture', help='Also copy YOLOv8 *-cls.pt to this relative subdir under root (optional)')
    args = p.parse_args()

    root = Path(args.root)
    copy_cls_dir = root / args.also_copy_cls_to if args.also_copy_cls_to else None

    logs = []
    for fam, files in ASSETS.items():
        dest_dir = root / DEST.get(fam, fam)
        for fn in files:
            url = BASE + fn
            out_path = dest_dir / fn
            ok, msg = download(url, out_path)
            logs.append(f"[{fam}] {msg}")
            # Optional copy for classification heads
            if ok and copy_cls_dir is not None and ('-cls.' in fn):
                try:
                    copy_cls_dir.mkdir(parents=True, exist_ok=True)
                    data = out_path.read_bytes()
                    (copy_cls_dir / fn).write_bytes(data)
                    logs.append(f"[COPY] {fn} -> {copy_cls_dir}")
                except Exception as e:
                    logs.append(f"[COPY-FAIL] {fn}: {e}")

    # write summary
    try:
        (root / 'Detector' / 'Ultralytics' / 'ULTRA_DOWNLOAD_SUMMARY.txt').write_text('\n'.join(logs), encoding='utf-8')
    except Exception:
        pass

    print('\n'.join(logs))


if __name__ == '__main__':
    main()

import argparse
import os
import json
from pathlib import Path
import sys
import numpy as np

# add src to path
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.utils.data import DataLoader
from data.dataset_fcs_vts import FCSVTSNpzDataset
from utils.seed import set_seed


def estimate_shift_bins(before: np.ndarray, after: np.ndarray) -> int:
    # Only meaningful if only freq shift applied with zero-fill
    # estimate leading/trailing zero-run along freq axis per row, take mode across rows
    T, Fw = after.shape
    # count leading zeros per row
    lead = []
    trail = []
    for t in range(T):
        row = after[t]
        # leading
        z = 0
        for j in range(Fw):
            if row[j] == 0.0:
                z += 1
            else:
                break
        lead.append(z)
        # trailing
        z = 0
        for j in range(Fw-1, -1, -1):
            if row[j] == 0.0:
                z += 1
            else:
                break
        trail.append(z)
    # choose the larger median as absolute shift
    s_lead = int(np.median(lead))
    s_trail = int(np.median(trail))
    return max(s_lead, s_trail)


def collect_stats(list_file: str, bandwidth_mhz: float | None, seed: int, limit: int) -> dict:
    set_seed(seed)
    stats = {}

    # Common loader options
    def make_loader(ds: FCSVTSNpzDataset):
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # 预取样本路径并裁剪数量
    with open(list_file, 'r', encoding='utf-8') as f:
        all_paths = [p.strip() for p in f if p.strip()]
    paths = all_paths if (limit is None or limit <= 0) else all_paths[:limit]

    # 1) BgMix only（仅抽样前 N 条以避免全量IO）
    ds = FCSVTSNpzDataset(list_file, is_train=True,
                          use_bg_mix=True, use_freq_shift=False, use_time_stretch=False,
                          use_noise=False, use_specaug=False,
                          bandwidth_mhz=bandwidth_mhz)
    ds.aug['bg_mix_p'] = 1.0
    # 轻扫若干样本，跳过重计算
    for i, _ in enumerate(make_loader(ds)):
        if limit and limit > 0 and i + 1 >= limit:
            break
        # pass
        continue
    stats['bg_mix'] = {'note': 'applied with p=1.0 on non-class0 samples; center-freq matching disabled; global pool only.'}

    # 2) Freq shift only（仅对前 N 条逐一估计bin偏移）
    ds = FCSVTSNpzDataset(list_file, is_train=True,
                          use_bg_mix=False, use_freq_shift=True, use_time_stretch=False,
                          use_noise=False, use_specaug=False,
                          bandwidth_mhz=bandwidth_mhz)
    ds.aug['freq_shift_p'] = 1.0
    px_max_list = []
    est_s_bins = []
    bw_list = []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        spec0 = d['spec_512'].astype('float32')
        bw = ds._get_bandwidth_mhz(d)
        px_max = int(round((ds.aug['shift_mhz'] * 512.0) / max(1e-6, bw)))
        # apply one item through dataset pipeline
        fcs, _ = ds.__getitem__(ds.items.index(path))
        spec1 = fcs.squeeze(0).numpy()
        s_est = estimate_shift_bins(spec0, spec1)
        px_max_list.append(px_max)
        est_s_bins.append(s_est)
        bw_list.append(bw)
    stats['freq_shift'] = {
        'bandwidth_mhz_unique': sorted(list({round(float(b), 3) for b in bw_list})) if bw_list else [],
        'shift_mhz': ds.aug['shift_mhz'],
        'px_max_min': int(min(px_max_list)) if px_max_list else 0,
        'px_max_max': int(max(px_max_list)) if px_max_list else 0,
        'est_shift_bins_min': int(min(est_s_bins)) if est_s_bins else 0,
        'est_shift_bins_max': int(max(est_s_bins)) if est_s_bins else 0,
        'expected_bins_for_10mhz': '≈51~52 at BW=100MHz',
        'samples_used': len(paths),
    }

    # 3) Time stretch only（轻扫 N 条确保shape）
    ds = FCSVTSNpzDataset(list_file, is_train=True,
                          use_bg_mix=False, use_freq_shift=False, use_time_stretch=True,
                          use_noise=False, use_specaug=False,
                          bandwidth_mhz=bandwidth_mhz)
    ds.aug['time_stretch_p'] = 1.0
    cnt = 0
    for _ in make_loader(ds):
        cnt += 1
        if limit and limit > 0 and cnt >= limit:
            break
    stats['time_stretch'] = {
        'scale_range': list(ds.aug['time_scale_range']),
        'output_shape': [512, 512],
        'samples_used': min(cnt, limit) if limit and limit > 0 else cnt,
    }

    # 4) SpecAug only（仅记录配置）
    ds = FCSVTSNpzDataset(list_file, is_train=True,
                          use_bg_mix=False, use_freq_shift=False, use_time_stretch=False,
                          use_noise=False, use_specaug=True,
                          bandwidth_mhz=bandwidth_mhz)
    ds.aug['specaug_p'] = 1.0
    stats['specaug'] = {
        'time_ratio': ds.aug['specaug_time_ratio'],
        'freq_ratio': ds.aug['specaug_freq_ratio'],
        'num_masks': ds.aug['specaug_num_masks'],
        'fill': 'mean',
    }

    # 5) Noise only（仅记录k范围）
    ds = FCSVTSNpzDataset(list_file, is_train=True,
                          use_bg_mix=False, use_freq_shift=False, use_time_stretch=False,
                          use_noise=True, use_specaug=False,
                          bandwidth_mhz=bandwidth_mhz)
    ds.aug['noise_p'] = 1.0
    stats['noise'] = {
        'k_min': ds.aug['noise_k_range'][0],
        'k_max': ds.aug['noise_k_range'][1],
    }

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_list', required=True)
    ap.add_argument('--bandwidth_mhz', type=float, default=None)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--out', type=str, default='aug_stats.json')
    ap.add_argument('--limit', type=int, default=64, help='limit number of samples for self-check; <=0 means all')
    args = ap.parse_args()

    stats = collect_stats(args.train_list, args.bandwidth_mhz, args.seed, args.limit)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved stats to {out_path}")


if __name__ == '__main__':
    main()

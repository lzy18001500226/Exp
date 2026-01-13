from pathlib import Path
import sys
import argparse
from torch.utils.data import DataLoader
import torch

# ensure project root
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.data.dataset_npz import NPZDataset
from SMNet.scripts.train_npz_fcs import collate_fn
from src.models.smnet_head import generate_anchors
from SMNet.scripts.train_npz_fcs import parse_anchor_sizes_levels
from src.models.smnet_loss import bbox_iou_xywh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=str(Path("D:/Exp/SMNet/FCSData/splits-upto200/train.json")))
    p.add_argument("--anchor-sizes-levels", default="6x8,7x5,10x7;10x7,13x21,19x24;19x24,29x31,37x50",
                   help="three semicolon-separated lists for p2;p3;p4")
    p.add_argument("--pos-iou", type=float, default=0.45)
    p.add_argument("--neg-iou", type=float, default=0.35)
    p.add_argument("--max-samples", type=int, default=0)
    args = p.parse_args()

    idx = Path(args.index)
    assert idx.exists(), f"index not found: {idx}"
    ds = NPZDataset(str(idx), require_label=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse anchor sizes levels
    a_levels = parse_anchor_sizes_levels(*args.anchor_sizes_levels.split(";"))
    strides = [8, 16, 32]
    grids = [(64, 64), (32, 32), (16, 16)]
    anchors_levels = [generate_anchors(h, w, s, sizes).cpu() for (h, w), s, sizes in zip(grids, strides, a_levels)]

    # stats
    total_samples = 0
    per_level_stats = [{"anchors": anchors_levels[i].shape[0], "pos": 0, "neg": 0, "ignore": 0, "matched_gts": 0, "total_gt": 0} for i in range(len(anchors_levels))]
    total_gt_all = 0

    for i, batch in enumerate(loader):
        if args.max_samples and i >= args.max_samples:
            break
        Xb, ys = batch
        y = ys[0]
        if y.numel() == 0:
            # no gt: all anchors negative
            for li in range(len(anchors_levels)):
                per_level_stats[li]["neg"] += anchors_levels[li].shape[0]
            total_samples += 1
            continue

        # build gt boxes (pixels)
        y = y.cpu()
        cls = y[:, 0].long()
        cxcywh = y[:, 1:] * 512.0
        M = cxcywh.shape[0]
        total_gt_all += M
        for li, anchors in enumerate(anchors_levels):
            iou = bbox_iou_xywh(anchors, cxcywh)  # [A, M]
            # per-anchor labels
            max_iou_per_anchor, _ = iou.max(dim=1)
            pos_mask = max_iou_per_anchor >= args.pos_iou
            neg_mask = max_iou_per_anchor <= args.neg_iou
            ignore_mask = (~pos_mask) & (~neg_mask)
            per_level_stats[li]["pos"] += int(pos_mask.sum().item())
            per_level_stats[li]["neg"] += int(neg_mask.sum().item())
            per_level_stats[li]["ignore"] += int(ignore_mask.sum().item())
            # per-gt matched
            max_iou_per_gt, _ = iou.max(dim=0)
            matched_gts = int((max_iou_per_gt >= args.pos_iou).sum().item())
            per_level_stats[li]["matched_gts"] += matched_gts
            per_level_stats[li]["total_gt"] += M

        total_samples += 1

    print(f"samples processed: {total_samples}")
    print(f"total_gts: {total_gt_all}")
    for li, s in enumerate(per_level_stats):
        anchors_per_sample = s["anchors"] * total_samples
        print(f"\nLevel p{li+2} -- anchors_per_image={s['anchors']}")
        print(f"  total_anchors (all images): {anchors_per_sample}")
        print(f"  pos: {s['pos']} ({s['pos']/anchors_per_sample:.6f})")
        print(f"  neg: {s['neg']} ({s['neg']/anchors_per_sample:.6f})")
        print(f"  ignore: {s['ignore']} ({s['ignore']/anchors_per_sample:.6f})")
        print(f"  matched_gts: {s['matched_gts']} / total_gt: {s['total_gt']} ({s['matched_gts']/max(1,s['total_gt']):.4f})")

    # summary across levels: how many GTs matched by any level
    # re-run to compute per-gt any-level match counts (smaller extra pass)
    any_matched = 0
    processed = 0
    for i, batch in enumerate(loader):
        if args.max_samples and i >= args.max_samples:
            break
        Xb, ys = batch
        y = ys[0]
        if y.numel() == 0:
            processed += 1
            continue
        y = y.cpu()
        cxcywh = y[:, 1:] * 512.0
        M = cxcywh.shape[0]
        # compute max over all levels
        max_iou_all = torch.zeros((M,))
        for anchors in anchors_levels:
            iou = bbox_iou_xywh(anchors, cxcywh)
            max_iou_per_gt, _ = iou.max(dim=0)
            max_iou_all = torch.maximum(max_iou_all, max_iou_per_gt)
        any_matched += int((max_iou_all >= args.pos_iou).sum().item())
        processed += 1
    print(f"\nGTs matched by any level: {any_matched} / {total_gt_all} ({any_matched/total_gt_all:.4f})")


if __name__ == "__main__":
    main()

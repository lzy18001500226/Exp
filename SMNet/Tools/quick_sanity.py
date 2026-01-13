from pathlib import Path
import json
import torch
import numpy as np
import sys

# ensure project root
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.data.dataset_npz import NPZDataset
from src.models.smnet_head import generate_anchors
from src.models.smnet_loss import DetectionLoss
from SMNet.scripts.train_npz_fcs import ModelWrap, collate_fn, parse_anchor_sizes, parse_anchor_sizes_levels
from torch.utils.data import DataLoader


def main():
    idx = Path(r"D:\Exp\SMNet\FCSData\splits-upto200\train.json")
    assert idx.exists(), f"index not found: {idx}"
    ds = NPZDataset(str(idx), require_label=True)
    print(f"dataset len={len(ds)}")
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 24

    # multi-level sanity
    anchor_sizes_levels = parse_anchor_sizes_levels("6x8,8x10,12x12", "12x16,16x24,24x32", "32x48,48x64,64x96")
    model = ModelWrap(num_classes=num_classes,
                      anchor_sizes=parse_anchor_sizes("8x8,12x12,16x16,24x24"),
                      head_level="p3",
                      multi_level=True,
                      anchor_sizes_levels=anchor_sizes_levels).to(device)
    criterion = DetectionLoss(num_classes=num_classes)

    strides_levels = [8, 16, 32]
    grids = [(64, 64), (32, 32), (16, 16)]
    anchors_pix_levels = [
        generate_anchors(h, w, s, sizes).to(device)
        for (h, w), s, sizes in zip(grids, strides_levels, anchor_sizes_levels)
    ]
    anchors_pix = torch.cat(anchors_pix_levels, dim=0)

    Xb, ys = next(iter(loader))
    Xb = Xb.to(device)
    # build targets (pixels)
    targets = []
    for y in ys:
        if y.numel() == 0:
            targets.append({"boxes": torch.zeros((0, 4), device=device),
                            "labels": torch.zeros((0,), dtype=torch.long, device=device)})
        else:
            y = y.to(device)
            cls = y[:, 0].long()
            cxcywh = y[:, 1:] * 512.0
            targets.append({"boxes": cxcywh, "labels": cls})

    out = model(Xb)
    from src.models.smnet_head import decode_multi_head_output
    conf_logits, pred_boxes, cls_logits = decode_multi_head_output(out, model.anchor_sizes_levels, model.strides_levels)
    loss, stat = criterion(conf_logits, pred_boxes, cls_logits, anchors_pix, targets)
    print(f"multi-level: loss={loss.item():.4f}, pos={stat['pos']}, neg={stat['neg']}")

    # single-level p2 sanity
    model2 = ModelWrap(num_classes=num_classes,
                       anchor_sizes=parse_anchor_sizes("8x8,12x12,16x16,24x24"),
                       head_level="p2",
                       multi_level=False).to(device)
    stride, grid_h, grid_w = 8, 64, 64
    anchors_pix2 = generate_anchors(grid_h, grid_w, stride, parse_anchor_sizes("8x8,12x12,16x16,24x24")).to(device)
    from src.models.smnet_head import decode_head_output
    out2 = model2(Xb)
    c2, b2, k2 = decode_head_output(out2, model2.anchor_sizes, stride)
    loss2, stat2 = criterion(c2, b2, k2, anchors_pix2, targets)
    print(f"single p2: loss={loss2.item():.4f}, pos={stat2['pos']}, neg={stat2['neg']}")


if __name__ == "__main__":
    main()

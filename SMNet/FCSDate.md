SMNet 数据准备与训练流程
========================

目录结构
--------

- `Data-512NPZ/`：原始 512×512×1 光谱 NPZ，按类别编号（`0` 背景）分目录。
- `FCSLabel/`：LabelMe JSON（以及可选 PNG 掩码）。
- `SMNet/FCSData/`：转换后的检测 NPZ、健康报告以及各阶段划分。
- `SMNet/Tools/`：保留现用脚本（`convert_to_npz.py`、`split_train_val.py`、`preview_npz.py`、`analyze_label_sizes.py` 等）。
- `SMNet/scripts/train_npz_fcs.py`：单尺度 Anchor 训练入口，默认读取黄金三通道。

环境准备
--------

- PowerShell 5.1+
- Conda 环境 `dronerfa`（已安装 `torch`, `numpy`, `opencv-python`, `Pillow` 等）
- 40 GB 以上可用磁盘空间（一次完整转换约 15 GB，三阶段训练 + 检查点约 20 GB）

步骤一：将原始 NPZ 转换为黄金三通道
-----------------------------------

`convert_to_npz.py` 会读取 `spec_512` 并输出：

- `golden_triplet`: `[3,512,512] float32，顺序固定为 `[Log-Spec, Gray-Norm, Corner-Mask]`，并强制落在 `[0,1]` 区间。
- `image`: `uint8` 伪彩（R=Log, G=Edge, B=Corner）用于人工抽检，依旧保留以兼容 `preview_npz.py`。
- 其余键（`mask`, `bboxes`, `gray_norm`, `corner_mask_clean`, `pseudo_rgb_u8` 等）为旧流水线兼容保留。

```powershell
conda activate dronerfa

python D:/Exp/SMNet/Tools/convert_to_npz.py `
  --source-root D:/Exp/Data-512NPZ `
  --label-root  D:/Exp/FCSLabel `
  --output-root D:/Exp/SMNet/FCSData `
  --stem-min 0 `
  --stem-max 200 `
  --use-png-mask-fallback `
  --overwrite
```

执行要点：

1. 命令结束后检查 `SMNet/FCSData/npz_health_report.json`，确认 `golden_triplet_range` 仍位于 `[0,1]`，并关注缺失标签/空框统计。
2. 若只更新少量类别或限定前 200 个样本，可配合 `--classes`、`--stem-min/--stem-max` 控制范围（上例将转换 0–200 区间的切片）。
3. **验证集必须执行同一脚本**。如果训练集使用了新黄金三通道而验证集仍是旧伪彩，将导致 mAP 完全失真；`NPZDataset` 会检测 fallback 并打印 WARNING，请立即补齐。
4. 可用 `preview_npz.py` 抽查输出（利用保留的 `image` 字段，**注意 `--out` 必须带 `.png` 等合法扩展名**，否则 Pillow 会报 `ValueError: unknown file extension`）：
   ```powershell
   python D:/Exp/SMNet/Tools/preview_npz.py `
     --dir D:/Exp/SMNet/FCSData/1 `
     --out D:/Exp/SMNet/FCSData/preview_cls1.png
   ```

步骤二：生成数据划分（Stage A/B/C）
-----------------------------------

三阶段策略仍保留，但每个阶段都基于黄金三通道数据。推荐使用与旧流程一致的类集合：

1. **Stage A**：仅前景 1–23（无背景），20 epoch。
2. **Stage B**：0–23 全部类别（含背景），20 epoch。
3. **Stage C**：精选 15 类（`1,3,4,5,6,8,10,11,14,15,16,17,18,19,21`），40 epoch 微调。

Stage A（仅前景 1–23）

```powershell
python D:/Exp/SMNet/Tools/split_train_val.py `
  D:/Exp/SMNet/FCSData `
  D:/Exp/SMNet/FCSData/splits_stageA `
  --ratio 0.8 `
  --seed 42 `
  --label-root D:/Exp/FCSLabel `
  --label-suffix=.json `
  --require-label-json `
  --min-labels 1 `
  --class-min 1 `
  --stem-max 200 `
  --keep-classes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
```

**Stage B（含背景 0–23）**

```powershell
python D:/Exp/SMNet/Tools/split_train_val.py `
  D:/Exp/SMNet/FCSData `
  D:/Exp/SMNet/FCSData/splits_stageB `
  --ratio 0.8 `
  --seed 42 `
  --label-root D:/Exp/FCSLabel `
  --label-suffix=.json `
  --require-label-json `
  --min-labels 1 `
  --class-min 0 `
  --stem-max 200 `
  --keep-classes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 `
  --allow-empty
```

**Stage C（15 类精选）**

```powershell
python D:/Exp/SMNet/Tools/split_train_val.py `
  D:/Exp/SMNet/FCSData `
  D:/Exp/SMNet/FCSData/splits_stageC `
  --ratio 0.8 `
  --seed 42 `
  --label-root D:/Exp/FCSLabel `
  --label-suffix=.json `
  --require-label-json `
  --min-labels 1 `
  --class-min 1 `
  --class-max 23 `
  --stem-max 200 `
  --keep-classes 1 3 4 5 6 8 10 11 14 15 16 17 18 19 21
```

务必核对命令行输出中 train/val 计数与类别集合；必要时可在 JSON 里 spot-check。背景（class=0）数据仍建议保留在 Stage B 之后再混入。
另外，Stage B 需要额外的背景列表，可用单独命令生成：

```powershell
python D:/Exp/SMNet/Tools/split_train_val.py `
  D:/Exp/SMNet/FCSData `
  D:/Exp/SMNet/FCSData/splits_stageB/background_only `
  --ratio 1.0 `
  --seed 0 `
  --label-root D:/Exp/FCSLabel `
  --label-suffix=.json `
  --require-label-json `
  --min-labels 0 `
  --class-min 0 `
  --class-max 0 `
  --keep-classes 0 `
  --allow-empty

Copy-Item `
  D:/Exp/SMNet/FCSData/splits_stageB/background_only/train.json `
  D:/Exp/SMNet/FCSData/splits_stageB/background.json
```

这样目录下就会出现 `train.json`、`val.json` 与 `background.json` 三个文件；若缺失 `background.json`，请重新执行上述两行命令，否则训练时的 `--background-json` 会报错并导致背景样本无法混入。

步骤三：训练（单尺度 AnchorHead）
---------------------------------

训练入口统一为 `scripts/train_npz_fcs.py`，其特性：

- 默认读取 `golden_triplet`，并在日志中打印“ChannelAttention active...”提示。
- `--anchor-sizes` 仅需提供一串 `WxH`，脚本会按照面积自动排序并在日志中输出最终顺序。
- `dump_vis` 会同时产出 JSON 与 PNG（R=Log, G=Gray, B=Corner），方便核对 Corner 通道是否点亮。

### 推荐 baseline Anchor 字符串

来自 `Tools/analyze_label_sizes.py` (针对 15 类) 的标准输入：

```
6x5,7x6,10x5,12x6,12x6,12x7,8x19,7x23,7x23,8x22,7x29,18x16,19x20,34x22
```

脚本会排序后再用于 AnchorHead，因此字符串顺序可保持如上（已按面积从小到大排列）。如需复核，可运行：

```powershell
python D:/Exp/SMNet/Tools/analyze_label_sizes.py `
  --label-root D:/Exp/FCSLabel `
  --classes 1 3 4 5 6 8 10 11 14 15 16 17 18 19 21 `
  --topk 5 `
  --out D:/Exp/SMNet/FCSData/anchor_stats_stageC.json
```

### 阶段 A 示例（1–23，无背景）

```powershell
python D:/Exp/SMNet/scripts/train_npz_fcs.py `
  --train-json D:/Exp/SMNet/FCSData/splits_stageA/train.json `
  --val-json   D:/Exp/SMNet/FCSData/splits_stageA/val.json `
  --epochs 60 `
  --batch 12 `
  --hflip `
  --neg-topk-ratio 3 `
  --mixup-prob 0.0 `
  --anchor-sizes "6x5,7x6,10x5,12x6,12x6,12x6,12x7,8x19,7x23,7x23,8x22,7x29,18x16,19x20,34x22" `
  --out D:/Exp/SMNet/runs/exp_stageA `
  --val-interval 2 `
  --per-class-ap
```

### 阶段 B 示例（0–23 + 背景）

```powershell
python D:/Exp/SMNet/scripts/train_npz_fcs.py `
  --train-json D:/Exp/SMNet/FCSData/splits_stageB/train.json `
  --val-json   D:/Exp/SMNet/FCSData/splits_stageB/val.json `
  --epochs 30 `
  --batch 12 `
  --hflip `
  --mixup-prob 0.35 `
  --mixup-alpha 0.3 `
  --neg-topk-ratio 3 `
  --anchor-sizes "6x5,7x6,10x5,12x6,12x6,12x6,12x7,8x19,7x23,7x23,8x22,7x29,18x16,19x20,34x22" `
  --background-json D:/Exp/SMNet/FCSData/splits_stageB/background.json `
  --background-frac 0.15 `
  --out D:/Exp/SMNet/runs/exp_stageB `
  --init-weights D:/Exp/SMNet/runs/exp_stageA/best.pth `
  --val-interval 2 `
  --per-class-ap `
  --lr 5e-4
```

### 阶段 C 示例（15 类精选）

```powershell
 `
  --lr 5e-4python D:/Exp/SMNet/scripts/train_npz_fcs.py `
  --train-json D:/Exp/SMNet/FCSData/splits_stageC/train.json `
  --val-json   D:/Exp/SMNet/FCSData/splits_stageC/val.json `
  --epochs 60 `
  --batch 12 `
  --hflip `
  --mixup-prob 0.35 `
  --mixup-alpha 0.3 `
  --neg-topk-ratio 3 `
  --anchor-sizes "6x5,7x6,10x5,12x6,12x6,12x6,12x7,8x19,7x23,7x23,8x22,7x29,18x16,19x20,34x22" `
  --out D:/Exp/SMNet/runs/exp_stageC `
  --init-weights D:/Exp/SMNet/runs/exp_stageB/best.pth `
  --val-interval 2 `
  --per-class-ap `
  --dump-vis 10 `
  --lr 1e-4
```

命令要点：

- `--dump-vis` 输出 JSON + PNG，可检查 Corner 通道（B 通道）是否点亮。
- `--resize` 仅支持 512，传入其它数值会强制改为 512 并在日志中提示。
- Mixup 与水平翻转已解耦，是否启用互不影响。
- 若 GPU 显存紧张，可用 `--accumulation-steps 2` 实现梯度累积；脚本会自动除以步数。
- **跨阶段衔接时请使用 `--init-weights`**：它只加载上一阶段的模型权重，训练会从 epoch=1 重新计数；若误用 `--resume`，会因为继承旧的 `epoch`/optimizer 状态而直接跳过整个训练循环。

黄金三通道 & Dataset 注意事项
-----------------------------

1. `NPZDataset` 默认 `feature_key='golden_triplet'`。若加载到旧字段，会打印 WARNING 并提醒重新转换——请不要长期依赖 fallback。
2. `extra_keys` 支持 `edges/corners/gray` 三类附加通道；日志中会记录“Loaded N channels from ... (+extras: ..)”。
3. 验证集必须与训练集保持相同通道，否则损失会出现“花瓣”状震荡。
4. `dump_vis` PNG：R=Log-Spec、G=Gray-Norm、B=Corner-Mask。如果 B 通道全黑，说明 Corner 预处理有问题；R 超暗则是 Log 归一化失败。

Troubleshooting
---------------

- **健康报告异常**：`golden_triplet_range` 如果突破 `[0,1]`，说明转换脚本被篡改或原始数据存在负值，请先修复输入。
- **train/val JSON 无样本**：检查 `split_train_val.py` 的 `--keep-classes` 是否排除了全部类，必要时去掉 `--class-min`。
- **PowerShell 报 `&&` 错误**：在 PowerShell 中使用 `;` 链接命令，或直接分行书写，如上例所示。
- **缺少 `torch/opencv`**：确保始终在 `conda run -n dronerfa` 或 `conda activate dronerfa` 后运行脚本。

附录：常用工具脚本
------------------

| 脚本                             | 作用                               |
| -------------------------------- | ---------------------------------- |
| `Tools/convert_to_npz.py`      | 生成黄金三通道、掩码、框及健康报告 |
| `Tools/preview_npz.py`         | 批量导出 PNG 用于人工审查          |
| `Tools/split_train_val.py`     | 根据条件切分 train/val JSON        |
| `Tools/analyze_label_sizes.py` | 统计各类框尺寸并输出 Anchor 建议   |
| `scripts/train_npz_fcs.py`     | 单尺度 AnchorHead 训练入口         |

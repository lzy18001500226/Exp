SMNet 数据准备与训练流程（FCS 信号检测）
=========================================

## 概述

SMNet 是单模态 RF 信号目标检测框架，采用**黄金三通道**特征表示和三阶段课程学习策略。本文档描述 FCS (Frequency-Correlation Signal) 信号的完整流水线。

### 核心特性

- **黄金三通道 (Golden Triplet)**: `[Log-Spec, Gray-Norm, Corner-Mask]` 三通道特征融合
- **三阶段训练**: Stage A (纯前景) → Stage B (引入背景) → Stage C (精选类微调)
- **Anchor-based 检测**: 单尺度 32×32 特征图 + 多尺度 Anchor + CenterNet-style 检测头
- **BCE/Focal + Hard Negative Mining**: 置信度损失可选 BCE 或 Focal；支持 Hard Neg Mining

---

目录结构
--------

```
D:/Exp/
├─ Data-512NPZ/               # 原始 512×512×1 光谱 NPZ（按类别 0-23）
│  ├─ 0/, 1/, ..., 23/
│  │   └─ *.npz (包含 spec_512 键)
├─ FCSLabel/                  # LabelMe JSON 标注
│  ├─ 0/, 1/, ..., 23/
│  │   └─ *.json (shapes 格式)
├─ SMNet/
│  ├─ FCSData/                # 转换后的检测 NPZ + 健康报告
│  │  ├─ npz_health_report.json
│  │  ├─ 0/, 1/, ..., 23/
│  │  │   └─ *.npz (包含 golden_triplet 键)
│  │  ├─ splits_stageA/       # Stage A 划分
│  │  │   ├─ train.json
│  │  │   └─ val.json
│  │  ├─ splits_stageB/       # Stage B 划分 + 背景池
│  │  │   ├─ train.json
│  │  │   ├─ val.json
│  │  │   └─ background.json
│  │  └─ splits_stageC/       # Stage C 划分
│  ├─ Tools/                  # 数据处理工具集
│  │  ├─ convert_to_npz.py    # 生成黄金三通道
│  │  ├─ split_train_val.py   # 数据集划分
│  │  ├─ preview_npz.py       # 可视化预览
│  │  └─ analyze_label_sizes.py  # Anchor 统计
│  ├─ scripts/
│  │  └─ train_npz_fcs.py     # 训练入口
│  └─ runs/                   # 训练输出
│     ├─ exp_stageA/
│     ├─ exp_stageB/
│     └─ exp_stageC/
```

---

环境准备
--------

### 系统要求
- **操作系统**: Windows 10/11 + PowerShell 5.1+
- **Python**: 3.9+ (Conda 环境 `dronerfa`)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (推荐 12GB+ 显存)
- **磁盘空间**: 40 GB+
  - NPZ 转换: ~15 GB
  - 三阶段训练 + 检查点: ~20 GB

### 依赖安装
```powershell
conda activate dronerfa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python Pillow tqdm
```

---

## 步骤一：转换为黄金三通道检测数据集

### 1.1 黄金三通道原理

`convert_to_npz.py` 从原始单通道光谱 `spec_512` 生成三个互补特征：

1. **Log-Spec (对数光谱)**
   - 计算: `log1p(spec)` 后百分位归一化 (p1-p99)
   - 作用: 增强低能量区域细节，压缩动态范围
   - 通道位置: `golden_triplet[0]`

2. **Gray-Norm (灰度归一化)**
   - 计算: 原始光谱的百分位归一化 (p1-p99)
   - 作用: 保留原始强度分布，提供基准参考
   - 通道位置: `golden_triplet[1]`

3. **Corner-Mask (角点掩码)**
   - 计算: Harris 角点检测 → 高斯平滑 → 形态学清理 → 二值化
   - 参数:
     - `gaussian_radius=2`: 角点响应平滑半径
     - `gamma=0.05`: Harris 响应函数抑制参数
     - `eta_ratio=0.01`: 角点阈值相对比例
     - `min_coverage=0.01`: 最小覆盖率保证
   - 作用: 标记信号特征点，提供空间结构先验
   - 通道位置: `golden_triplet[2]`

### 1.2 输出数据结构

转换后的 NPZ 包含以下键：

| 键名 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `golden_triplet` | `[3,512,512]` | float32 | **主特征**: 黄金三通道，值域 `[0,1]` |
| `image` | `[512,512,3]` | uint8 | 伪彩预览 (R=Log, G=Edge, B=Corner) |
| `mask` | `[512,512]` | uint8 | 从 LabelMe shapes 生成的类别掩码 |
| `bboxes` | `[N,5]` | float32 | 边界框 `[x1,y1,x2,y2,class]` (像素坐标) |
| `label_json` | scalar | str | 原始 JSON 标注文件内容 |
| `class_id` | scalar | int32 | 类别 ID (从目录名推断) |
| `gray_norm` | `[512,512]` | float32 | 灰度归一化 (可用于 `--extra-keys`) |
| `corner_mask_clean` | `[512,512]` | float32 | 清理后的角点掩码 |
| `source_npz` | scalar | str | 原始 NPZ 路径 |

### 1.3 执行转换

#### 基础命令（FCS 信号，0-200 stem 范围）

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

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source-root` | `D:/Exp/Data-512NPZ` | 原始 NPZ 根目录（包含 0-23 子目录） |
| `--label-root` | `D:/Exp/FCSLabel` | LabelMe JSON 标注根目录 |
| `--output-root` | `D:/Exp/SMNet/FCSData` | 输出目录 |
| `--stem-min` | None | 文件名前缀最小值（如 `0-a.npz` → 0） |
| `--stem-max` | None | 文件名前缀最大值（如 `200-b.npz` → 200） |
| `--classes` | None | 指定转换的类别列表（如 `--classes 1 3 5`） |
| `--use-png-mask-fallback` | False | JSON 缺失时尝试加载 PNG 掩码 |
| `--overwrite` | False | 覆盖已存在的输出文件 |
| `--label-suffix` | `.json` | 标注文件后缀 |
| `--print-every` | 100 | 进度打印间隔 |

#### 执行要点

1. **健康报告检查**: 转换完成后自动生成 `SMNet/FCSData/npz_health_report.json`
   ```powershell
   Get-Content D:/Exp/SMNet/FCSData/npz_health_report.json | ConvertFrom-Json
   ```
   
   关键指标：
   - `golden_triplet_range`: **必须在 `[0,1]` 范围内**，否则特征计算有误
   - `converted`: 成功转换的样本数
   - `empty_boxes`: 无框样本数（Stage A/C 会过滤掉）
   - `missing_label_json`: 缺失标注的样本数
   - `per_class`: 每类统计（source/converted/with_boxes/boxes）

2. **增量更新**: 新增标注后只需删除对应 NPZ 文件，重新运行转换（不加 `--overwrite` 则跳过已存在文件）

3. **验证集同步**: **必须对验证集数据也执行相同转换**，否则 mAP 计算会失真

### 1.4 可视化验证

使用 `preview_npz.py` 抽查转换结果：

```powershell
# 单类预览（注意 --out 必须带 .png 扩展名）
python D:/Exp/SMNet/Tools/preview_npz.py `
  --dir D:/Exp/SMNet/FCSData/1 `
  --out D:/Exp/SMNet/FCSData/preview_cls1.png

# 多类预览（生成网格图）
python D:/Exp/SMNet/Tools/preview_npz.py `
  --dir D:/Exp/SMNet/FCSData/1 `
  --dir D:/Exp/SMNet/FCSData/3 `
  --dir D:/Exp/SMNet/FCSData/5 `
  --out D:/Exp/SMNet/FCSData/preview_multi.png
```

预览图说明：
- **R 通道 (红色)**: Log-Spec，应显示清晰的信号轮廓
- **G 通道 (绿色)**: Gray-Norm，保留原始亮度分布
- **B 通道 (蓝色)**: Corner-Mask，信号关键点应点亮

**异常诊断**：
- B 通道全黑 → Corner 参数过严或信号过弱
- R 通道过暗 → Log 归一化失败，检查 `spec_512` 值域
- G 通道溢出 → 原始光谱超过 [0,1] 范围

---

## 步骤二：三阶段数据划分策略

### 2.1 课程学习原理

SMNet 采用**三阶段课程学习**解决小目标检测难题：

| 阶段 | 类别集合 | 样本类型 | 训练目标 | Epoch | 学习率 |
|------|---------|---------|---------|-------|--------|
| **Stage A** | 1-23 (纯前景) | 仅包含目标的样本 | 学习前景特征，避免背景干扰 | 20 | 1e-3 |
| **Stage B** | 0-23 (全类别) | 前景 + 15% 背景混合 | 学习前景-背景区分能力 | 20 | 5e-4 |
| **Stage C** | 15 精选类 | 筛选高质量类别 | 微调精度，减少类间混淆 | 80 | 1e-4 |

**Stage C 精选类** (15 个): `1, 3, 4, 5, 6, 8, 10, 11, 14, 15, 16, 17, 18, 19, 21`  
- 筛选依据: 样本量充足 (≥50) + 标注质量高 + 类内一致性好

### 2.2 划分工具说明

`split_train_val.py` 核心功能：

- **类别过滤**: `--keep-classes` / `--class-min` / `--class-max`
- **Stem 范围**: `--stem-min` / `--stem-max` (基于文件名前缀数字)
- **标注校验**: `--require-label-json` + `--min-labels` (最少框数)
- **外部标注**: `--label-root` (从独立目录重建标注路径)
- **随机种子**: `--seed 42` (保证可复现)
- **划分比例**: `--ratio 0.8` (train 80% / val 20%)

### 2.3 Stage A: 纯前景学习 (类别 1-23)

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

**输出文件**:
- `splits_stageA/train.json`: 训练集路径列表 (~80% 样本)
- `splits_stageA/val.json`: 验证集路径列表 (~20% 样本)

**验证要点**:
- 检查控制台输出的 `kept` 数量是否合理
- 确认无 class=0 样本混入
- 验证每类样本分布是否均衡

### 2.4 Stage B: 引入背景 (类别 0-23)

#### 主列表生成（含前景+背景样本）

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

#### 背景池生成（仅 class=0 纯背景）

```powershell
# 生成背景样本列表
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

# 复制为训练脚本识别的 background.json
Copy-Item `
  D:/Exp/SMNet/FCSData/splits_stageB/background_only/train.json `
  D:/Exp/SMNet/FCSData/splits_stageB/background.json
```

**输出文件**:
- `splits_stageB/train.json`: 包含前景的训练样本
- `splits_stageB/val.json`: 验证集
- `splits_stageB/background.json`: **纯背景池** (201 个 class=0 样本)

**背景混合机制**:
- 训练时通过 `--background-json` + `--background-frac 0.15` 控制
- 从 background.json 随机采样 15% × 训练集大小的背景样本
- 每个 epoch 重新采样，增加负样本多样性

### 2.5 Stage C: 精选类微调 (15 类)

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

**输出文件**:
- `splits_stageC/train.json`: 仅包含 15 个精选类的训练集
- `splits_stageC/val.json`: 对应验证集

**类别筛选依据**:
- 样本量: ≥50 个标注样本
- 标注质量: 边界框准确，无明显误标
- 类间区分度: 与其他类特征差异明显

### 2.6 验证划分结果

```powershell
# 统计各阶段样本数
(Get-Content D:/Exp/SMNet/FCSData/splits_stageA/train.json | ConvertFrom-Json).Count
(Get-Content D:/Exp/SMNet/FCSData/splits_stageB/background.json | ConvertFrom-Json).Count
(Get-Content D:/Exp/SMNet/FCSData/splits_stageC/val.json | ConvertFrom-Json).Count

# 检查类别分布（需要 Python 脚本，后续补充）
```

步骤三：训练（单尺度 AnchorHead）
---------------------------------

训练入口统一为 `scripts/train_npz_fcs.py`，其特性：

- 默认读取 `golden_triplet`（不足时可 fallback 到 `features/feature_stack/pseudo_rgb/image/X`）。
- `--anchor-sizes` 仅需提供一串 `WxH`，脚本会按照面积自动排序并在日志中输出最终顺序。
- `dump_vis` 会同时产出 JSON 与 PNG（R=Log, G=Gray, B=Corner），方便核对 Corner 通道是否点亮。
- 支持 `--conf-loss bce|focal` 与 `--neg-topk-ratio`（Hard Neg Mining）。
- **评估默认输出论文口径指标**：`AvgP/AvgR/AvgF1@conf=0.5`（IoU=0.5）。如需 AP50，显式加 `--ap50`。

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
  --paper-conf 0.5
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
  --paper-conf 0.5 `
  --lr 5e-4
```

### 阶段 C 示例（15 类精选）

```powershell
python D:/Exp/SMNet/scripts/train_npz_fcs.py `
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
  --paper-conf 0.5 `
  --dump-vis 10 `
  --lr 1e-4
```

命令要点：

- `--dump-vis` 输出 JSON + PNG，可检查 Corner 通道（B 通道）是否点亮。
- `--resize` 仅支持 512，传入其它数值会强制改为 512 并在日志中提示。
- Mixup 与水平翻转已解耦，是否启用互不影响。
- 若 GPU 显存紧张，可用 `--accumulation-steps 2` 实现梯度累积；脚本会自动除以步数。
- **跨阶段衔接时请使用 `--init-weights`**：它只加载上一阶段的模型权重，训练会从 epoch=1 重新计数；若误用 `--resume`，会因为继承旧的 `epoch`/optimizer 状态而直接跳过整个训练循环。
- 评估默认输出论文口径：`AvgP/AvgR/AvgF1@conf=0.5 (IoU=0.5)`；如需 AP50，显式加 `--ap50`。
- `--paper-conf/--paper-iou` 控制论文口径阈值；`--conf-thres` 仅影响 AP50 的预筛选阈值。

---

## 当前实现要点（与代码一致）

以下为当前代码已实现的训练/评估要点（无需额外开关）：

1) **回归损失：SmoothL1**
  - 位置：`smnet_loss.py`
  - 公式：$L_{reg}=\sum \mathrm{SmoothL1}(pred,gt)$

2) **Objectness 目标为 1**
  - 置信度目标为 1（非 IoU‑aware）

3) **单尺度检测**
  - 单尺度 32×32 (stride=16)

4) **评估：论文口径 AvgP/AvgR/AvgF1（IoU=0.5, conf=0.5）**
  - 如需 AP50（VOC07 11-point），加 `--ap50`

---

## P3 优化清单（常用且易落地）

以下为检测任务中**常用、且工程容易加**的优化，按优先级列出：

### P3‑1 EMA（Exponential Moving Average）

- 训练仍用原模型更新
- 验证 / 保存 best 用 EMA 模型
- 在**小数据、噪声大**时提升更明显

### P3‑2 Warmup + Cosine

- 当前脚本已有 cosine 学习率
- **必须启用 warmup**（参数 `--warmup`）
- 特别是后续切换 ATSS/SimOTA 时，前期更容易不稳

### P3‑3 Checkpoint Averaging / SWA

- 训练末期对最后 $K$ 个 checkpoint 做平均
- 可提升 P/R 稳定性与置信度标定

### P3‑4 小规模系统调参流程（避免“没头绪”）

建议使用以下四步快速定位瓶颈：

1. **Overfit 50 张图**：loss 是否能降到很低，P/R 是否接近 1
2. **打印正样本分配统计**：使用 P1‑0 的统计输出（pos/neg、IoU 分布）
3. **固定 seed，单改一个变量**：anchor / assign / loss / aug 逐项验证
4. **每次改动保留完整记录**：训练曲线 + PR@固定阈值 + PR@best‑F1 + per‑class 表

---

## 评估口径（当前实现）

当前脚本默认输出 **论文口径**：

- IoU 阈值固定为 0.5
- 置信度阈值固定为 `--paper-conf`（默认 0.5）
- 输出 `AvgP/AvgR/AvgF1`

可选：

- `--ap50` 额外输出 AP50（VOC07 11-point）

---

## NMS 设置（重要）

当前评估使用 **class-agnostic NMS**（所有类别一起抑制），阈值由 `--nms-iou` 控制。

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

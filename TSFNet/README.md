# 项目工作区说明（Exp）

本仓库聚焦“基于射频（RF）信号的无人机检测与识别”。当前主工程为 TSFNet，分类为主线，检测脚本用于管线自检与基线对照。

## 目录结构

```
Exp/
├── Data/                    # 原始/中间数据（如 npy）
├── LabelMePNG/              # 导出的 PNG 与 LabelMe 标注（含 COCO 标注）
├── Model/                   # 训练产物与缓存权重（分类/检测输出）
├── Paper/                   # 论文与参考资料
├── TSFNet/                  # 主工程（模型/数据集/训练脚本/工具）
├── experiment_groups/       # 实验划分文件（按折/清单）
├── 第1-4章.md                  # 文档草稿
├── 草稿/                       # 其他草稿/临时资料
└── README.md                # 本文件
```

TSFNet 子项目关键结构（详见 `TSFNet/README.md`）：

```
TSFNet/
└─ src/
   ├─ model.py               # 两流分类（TS-CNN/TSFNet）与 Swin 适配
   ├─ dataset.py             # 分类数据集与增强
   ├─ aug_rf.py              # RF 增强（噪声/干扰/掩蔽/轻降噪/几何）
   ├─ train.py               # 分类训练与开放集评估
   ├─ det_coco_dataset.py    # 检测 COCO 封装（自检用）
   ├─ train_det.py           # 两流检测自检脚本（研究/自检用途）
   └─ Compare/               # 检测/分类最小基线脚本（对照与 sanity check）
```

## 数据与标注流程（固定）
1) 导出 PNG 到 `LabelMePNG/`（脚本见 `TSFNet/src/tools/npy转换png/`）。
2) LabelMe 标注后，用 `TSFNet/src/tools/labelme_to_coco.py` 转为 COCO。
3) 用 `TSFNet/src/tools/coco_split_train_val.py` 进行固定划分，生成 `coco_train.json` 与 `coco_val.json`。

## 训练与评估
- 分类（主线）：`TSFNet/src/train.py`，模型在 `TSFNet/src/model.py`；支持 EMA、Warmup/Cosine、梯度裁剪与开放集评估（energy/gamma）。
- 检测（自检/基线）：
  - 两流自检：`TSFNet/src/train_det.py`，用于验证数据与管线稳定性；
  - 最小基线：`TSFNet/src/Compare/train_det_baselines.py`、`train_ResNet50.py`（已修复验证划分与类别映射并增加 [Sanity] 统计）。

建议先以小 batch（bs=1）与 `workers=0` 做 dry-run，确认无越界/NaN 后再放大规模与启用 AMP。

## 常见约定
- 类别 ID 连续为 1..K（0 为背景保留位，不作为标注类别）。
- Windows 兼容：DataLoader 种子、路径、半精度均已适配；显存吃紧可关闭 AMP 与减小 ROI/Batch。
- 输出目录：未指定时默认写入 `Model/` 子目录（脚本内部会自动创建按架构归档的 run 目录）。

更多细节与示例请查看 `TSFNet/README.md`。

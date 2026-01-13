# 项目总览（TSFNet：分类为主，检测用于自检）

本项目围绕“基于射频(RF)信号的无人机检测与识别”，采用两条主线：

- TS-CNN/TSFNet 分类：两流特征（纹理/位置）+ PANet-lite；TSFNet 在高层引入 Swin Stage4。
- 检测仅用于管线自检：确保标注/增强/装载/前后向无误；结果不作为最终对比结论。

---

## 目录结构（与仓库根保持一致）

```
TSFNet/
├─ src/
│  ├─ model.py                  # 两流分类与 Swin 适配
│  ├─ dataset.py                # 分类数据集与增强
│  ├─ aug_rf.py                 # RF 增强（噪声/干扰/掩蔽/轻降噪/几何）
│  ├─ train.py                  # 分类训练与开放集评估（energy/gamma）
│  ├─ det_coco_dataset.py       # 检测 COCO 封装（含空图处理）
│  ├─ train_det.py              # 两流检测自检脚本（含 EMA/调度/冻结/OE等开关）
│  └─ Compare/                  # 最小基线脚本（对照与 sanity check）
│     ├─ train_det_baselines.py # Faster R-CNN / RetinaNet 基线（已修复 val 划分与类映射）
│     ├─ train_ResNet50.py      # Faster R-CNN ResNet50-FPN 基线，含 [Sanity] 统计
│     └─ det_coco_dataset.py    # Compare 侧最小 COCO 封装（若存在）
│
├─ src/tools/
│  ├─ npy转换png/               # Data/*.npy → PNG（供 LabelMe）
│  ├─ labelme_to_coco.py        # LabelMe → COCO（支持空图）
│  └─ coco_split_train_val.py   # COCO 固定划分（分层采样 + 空图比例）
└─ README.md
```

> 仓库根（Exp/）下还有：`Data/`、`LabelMePNG/`、`Model/`、`experiment_groups/`、`Paper/` 等共享目录。

---

## 数据准备与固定流程
1) Data → LabelMePNG：运行 `src/tools/npy转换png/Data_npy_to_png.py`；或使用 `export_all_folds.py/ps1` 按折导出。
2) LabelMe → COCO：`src/tools/labelme_to_coco.py`，支持空图与目录名推断类别。
3) COCO 固定划分：`src/tools/coco_split_train_val.py` 生成 `coco_train.json` 与 `coco_val.json`。

---

## 训练与评估
- 分类（主线）
  - 脚本：`src/train.py`；模型：`src/model.py`。
  - 支持 Warmup + Cosine、EMA、梯度裁剪、阶段性冻结、开放集能量阈值校准（gamma）。
  - 建议先小批量/关闭 AMP 做 dry-run，自检通过后再放大规模。

- 检测（管线自检/基线）
  - 两流自检：`src/train_det.py`，可选 AMP/EMA/冻结/OE 等；输出默认写入 `Model/` 下的架构子目录。
  - 最小基线：`src/Compare/train_det_baselines.py`、`train_ResNet50.py`。
    - 已加入：
      - 随机打乱后再划分验证集（固定种子）以避免无目标样本集中；
      - `categories_ref_json`/连续映射对齐（如需）;
      - 启动时打印 `[Sanity] Val GT images with boxes: X/Y | total boxes: Z`。

---

## 约定与注意
- 类别 ID 连续为 1..K，0 为背景保留位；空图出现在 `images`，不在 `annotations` 中出现。
- Windows 兼容：DataLoader 种子/路径/半精度已适配；显存不足时关闭 AMP、减小 batch/ROI。
- 权重管理：默认不在线下载；如需预训练请配置本地缓存路径。

---

## 近期修订要点
- Compare 基线：修复验证集划分与类别映射，并加入验证集 GT 统计打印。
- 自检脚本：加入 EMA 更新节流、动态早期阈值/IoU、自检日志开关（如 `--debug_eval`）。

有变更请同步更新本 README，保持“数据→训练→评估”闭环一致。

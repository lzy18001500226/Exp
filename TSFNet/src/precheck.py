"""
precheck.py — 训练前检查与参数拼装助手（分类任务）

用途：
- 校验/集中配置数据清单与权重路径；按折组装命令行参数；
- 作为 train.py 的启动辅助器，可批量按折运行；
- 本脚本不定义或改动模型结构，检测任务请使用 train_det.py。

说明：
- 这是原 run_training.py 的重命名版本。后者将保留为兼容入口，但不再推荐使用。
"""

import sys
from pathlib import Path

# 用户可在此修改配置（适配 RTX 4060 8GB）
CFG = {
    # 数据清单（使用各折的完整清单，覆盖全部样本）
    "train_list": r"C:\\Users\\HP\\Desktop\\Exp\\experiment_groups\\1-known_for_train",
    "val_list": r"C:\\Users\\HP\\Desktop\\Exp\\experiment_groups\\1-known_for_test",
    "test_list": r"C:\\Users\\HP\\Desktop\\Exp\\experiment_groups\\1-known_for_test",
    # 批量按折运行的根目录与折号（1~9）；将覆盖上面的三项
    "exp_root": r"C:\\Users\\HP\\Desktop\\Exp\\experiment_groups",
    "sweep_folds": True,
    "folds": [1,2,3,4,5,6,7,8,9],
    # 数据根目录
    "data_root": r"C:\\Users\\HP\\Desktop\\Exp\\Data",
    # 位置流与 Swin 权重
    "pos_weights": r"C:\\Users\\HP\\Desktop\\Exp\\Model\\Backbone\\Position\\mobilenet_v3_large-8738ca79.pth",
    "swin_stage4_weights": r"C:\\Users\\HP\\Desktop\\Exp\\Model\\Swin\\pth\\swin_small_patch4_window7_224.pth",
    # 训练超参（为 4060 8G 预设）
    "epochs": "50",   # 每折50轮
    "batch_size": "4",    # 显存不足可降到 6 或 4；富余可试 12
    "workers": "2",       # Windows 建议 4~6
    # 复现性
    "seed": "42",
    "deterministic": True,
    # 训练策略
    "lr": "0.001",
    "cosine": True,
    "amp": True,
    "freeze_epochs": "5",
}

# 组装 sys.argv 并调用 train.main()
def build_argv(cfg: dict, *, train_list: str=None, val_list: str=None, test_list: str=None,
               unknown_val_list: str="", unknown_test_list: str="", out_dir: str=None) -> list[str]:
    """仅负责参数拼装；真正训练在 train.py 内完成"""
    # 允许传入覆盖的列表；否则使用 CFG 中的
    t_list = train_list or cfg["train_list"]
    v_list = val_list or cfg["val_list"]
    te_list = test_list or cfg["test_list"]

    argv = [
        "train.py",
        "--epochs", cfg["epochs"],
        "--batch_size", cfg["batch_size"],
        "--workers", cfg["workers"],
        "--pos_backbone", "mobilenetv3",
        "--pos_weights", cfg["pos_weights"],
        "--use_swin",
        "--swin_stage4_weights", cfg["swin_stage4_weights"],
        "--train_list", t_list,
        "--val_list", v_list,
        "--test_list", te_list,
        "--data_root", cfg["data_root"],
        "--auto_num_classes",
        "--lr", cfg["lr"],
        "--freeze_epochs", cfg["freeze_epochs"],
        "--seed", cfg["seed"],
    ]
    if unknown_val_list:
        argv += ["--unknown_val_list", unknown_val_list]
    if unknown_test_list:
        argv += ["--unknown_test_list", unknown_test_list]
    if out_dir:
        argv += ["--out_dir", out_dir]
    if cfg.get("cosine", False):
        argv.append("--cosine")
    if cfg.get("amp", False):
        argv.append("--amp")
    if cfg.get("deterministic", False):
        argv.append("--deterministic")
    return argv

if __name__ == "__main__":
    # 确保在同目录下可导入 train.py
    this_dir = Path(__file__).parent
    if str(this_dir) not in sys.path:
        sys.path.append(str(this_dir))
    from train import main as train_main

    print("[precheck] This helper assembles args and starts train.py for classification. For detection use train_det.py.")

    if CFG.get("sweep_folds", False):
        exp_root = CFG["exp_root"].rstrip("\\/")
        for fold in CFG.get("folds", [1]):
            tr = fr"{exp_root}\{fold}-known_for_train"
            va = fr"{exp_root}\{fold}-known_for_test"
            te = va
            unk = fr"{exp_root}\{fold}-unknown"
            # 输出目录：统一到项目根 Model/TSFNet/foldX
            project_root = this_dir.parent.parent  # .../TSFNet/src -> .../Exp
            model_dir = project_root / 'Model' / 'TSFNet'
            out = str((model_dir / f"fold{fold}").resolve())
            sys.argv = build_argv(CFG, train_list=tr, val_list=va, test_list=te,
                                  unknown_val_list=unk, unknown_test_list=unk,
                                  out_dir=out)
            print(f"\n===== FOLD {fold} =====")
            print("Running with argv:", " ".join(sys.argv))
            train_main()
    else:
        sys.argv = build_argv(CFG)
        # 打印一次配置，便于确认
        print("Running with argv:", " ".join(sys.argv))
        train_main()

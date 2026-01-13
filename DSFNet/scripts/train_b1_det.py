from pathlib import Path
import sys
from train_b0_det import main as _main

# 复用 train_b0_det 的逻辑，仅强制 backbone_type=convnext
if __name__ == '__main__':
    # 将 --backbone_type 默认设为 convnext（用户也可覆盖）
    sys.argv = [sys.argv[0], '--backbone_type', 'convnext'] + sys.argv[1:]
    _main()

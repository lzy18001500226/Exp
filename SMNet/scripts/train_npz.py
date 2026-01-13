"""Backward-compatible entry point for SMNet training.

Historically the main training script lived at scripts/train_npz.py. We now keep
the implementation in scripts/train_npz_fcs.py (which also works for VTS), but we
preserve this thin wrapper so existing automation continues to function.
"""

from train_npz_fcs import main


if __name__ == "__main__":
	main()

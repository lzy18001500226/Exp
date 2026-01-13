from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PRESET_CONFIGS = {
	"vts-from-scratch": {
		"description": "Train VTS dataset from scratch (200 epochs, no pretrained weights).",
		"args": [
			"--train-json",
			"data_vts_train.json",
			"--val-json",
			"data_vts_val.json",
			"--num-classes",
			"17",
			"--batch",
			"6",
			"--epochs",
			"200",
			"--lr",
			"0.001",
			"--wd",
			"0.0001",
			"--warmup-epochs",
			"10",
			"--out",
			"runs/exp_vts_scratch",
			"--multi-level",
			"--anchor-sizes",
			"8x8,12x12,16x16,24x24",
			"--anchor-sizes-p2",
			"45x11,48x12,48x99",
			"--anchor-sizes-p3",
			"91x12,91x21,91x31",
			"--anchor-sizes-p4",
			"92x41,127x13,184x12",
			"--focal-gamma",
			"2.5",
			"--focal-alpha",
			"0.25",
			"--class-weights",
			"2:3.0,7:2.0,14:2.5",
			"--hflip",
			"--background-json",
			"data_vts_background.json",
			"--background-frac",
			"0.3",
			"--neg-topk-ratio",
			"3",
			"--val-interval",
			"2",
			"--per-class-ap",
		],
		"background_json": "data_vts_background.json",
	},
	"vts-resume": {
		"description": "Resume VTS training from existing checkpoint (260 epochs total).",
		"args": [
			"--train-json",
			"data_vts_train.json",
			"--val-json",
			"data_vts_val.json",
			"--num-classes",
			"17",
			"--batch",
			"6",
			"--epochs",
			"260",
			"--lr",
			"0.0002",
			"--wd",
			"0.0001",
			"--warmup-epochs",
			"3",
			"--resume",
			"runs/exp_vts_scratch/best.pth",
			"--out",
			"runs/exp_vts_resume",
			"--multi-level",
			"--anchor-sizes",
			"8x8,12x12,16x16,24x24",
			"--anchor-sizes-p2",
			"45x11,48x12,48x99",
			"--anchor-sizes-p3",
			"91x12,91x21,91x31",
			"--anchor-sizes-p4",
			"92x41,127x13,184x12",
			"--focal-gamma",
			"2.5",
			"--focal-alpha",
			"0.25",
			"--class-weights",
			"2:3.0,7:2.0,14:2.5",
			"--hflip",
			"--background-json",
			"data_vts_background.json",
			"--background-frac",
			"0.3",
			"--neg-topk-ratio",
			"3",
			"--val-interval",
			"2",
			"--per-class-ap",
		],
		"background_json": "data_vts_background.json",
	},
}


FILE_ARGS = {"--train-json", "--val-json", "--resume", "--out", "--background-json"}


def _project_root() -> Path:
	return Path(__file__).resolve().parents[1]


def _train_script() -> Path:
	return Path(__file__).resolve().parent / "train_npz.py"


def _resolve_path(value: str, root: Path) -> str:
	path = Path(value)
	if not path.is_absolute():
		path = (root / path).resolve()
	return str(path)


def _normalize_args(args: list[str], root: Path) -> list[str]:
	normalized: list[str] = []
	i = 0
	while i < len(args):
		token = args[i]
		if token in FILE_ARGS and i + 1 < len(args):
			normalized.append(token)
			normalized.append(_resolve_path(args[i + 1], root))
			i += 2
		else:
			normalized.append(token)
			i += 1
	return normalized


def _format_command(cmd: list[str]) -> str:
	try:
		return shlex.join(cmd)
	except AttributeError:
		return " ".join(cmd)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Preset launcher for VTS experiments built on train_npz.py",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS.keys()), default="vts-from-scratch")
	parser.add_argument("--python", default=sys.executable, help="Python interpreter used to run train_npz.py")
	parser.add_argument("--dry-run", action="store_true", help="Print command and exit without running it")
	parser.add_argument(
		"extra_args",
		nargs=argparse.REMAINDER,
		help="Extra args appended to train_npz.py (prefix with -- to separate, e.g. -- --epochs 50)",
	)
	args = parser.parse_args()

	config = PRESET_CONFIGS[args.preset]
	root = _project_root()
	script_path = _train_script()
	if not script_path.exists():
		raise FileNotFoundError(f"Missing core trainer: {script_path}")

	print(f"=== SMNet Training Preset: {args.preset} ===")
	print(config["description"])

	background_json = config.get("background_json")
	if background_json:
		bg_path = Path(background_json)
		if not bg_path.is_absolute():
			bg_path = (root / bg_path).resolve()
		if not bg_path.exists():
			print(f"[warn] Background index not found: {bg_path}")
		else:
			print(f"Background index: {bg_path}")

	base_args = _normalize_args(config["args"], root)
	extra = list(args.extra_args or [])
	if extra and extra[0] == "--":
		extra = extra[1:]
	cmd = [args.python, str(script_path)] + base_args + extra

	print("Command:")
	print(_format_command(cmd))

	if args.dry_run:
		print("Dry run; nothing executed.")
		return

	completed = subprocess.run(cmd, check=False)
	if completed.returncode != 0:
		raise SystemExit(completed.returncode)


if __name__ == "__main__":
	main()

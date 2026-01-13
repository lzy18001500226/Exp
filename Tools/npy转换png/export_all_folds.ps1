$ErrorActionPreference = "Stop"
$root = "C:\Users\HP\Desktop\Exp\LabelMePNG"
$data = "C:\Users\HP\Desktop\Exp\Data"
$grp  = "C:\Users\HP\Desktop\Exp\experiment_groups"
New-Item -ItemType Directory -Path $root -Force | Out-Null
for ($i=1; $i -le 9; $i++) {
  $csv = Join-Path $root ("index_fold_{0}.csv" -f $i)
  conda run -n dronerfa python -u "C:\Users\HP\Desktop\Exp\TSFNet\src\tools\npy转换png\export_all_folds.py" `
    --out_dir $root --data_root $data --colormap jet --size 512 `
    --ascii_bar --bar_ncols 80 `
    --list (Join-Path $grp ("{0}-known" -f $i)) `
    --list (Join-Path $grp ("{0}-known_for_test" -f $i)) `
    --list (Join-Path $grp ("{0}-known_for_train" -f $i)) `
    --list (Join-Path $grp ("{0}-unknown" -f $i)) `
    --write_csv $csv
}

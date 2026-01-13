param(
  # Lists: each line is a JSON path under FCSLabel
  [string]$TrainList = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output/FCS_0_200_ex1314/train_list_fcs.txt",
  [string]$ValList = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output/FCS_0_200_ex1314/val_list_fcs.txt",
  [int]$Epochs = 40,
  [int]$BatchSize = 4,
  [int]$NumClasses = 22,
  [string]$SwinName = "swin_base_patch4_window7_224",
  [string]$SwinCheckpoint = "C:/Users/HP/Desktop/Exp/Model/Pretrain/Swin/timm/swin_base_patch4_window7_224_timm.pth",
  [int]$NumWorkers = 0,
  [double]$Lr = 0.001,
  [int]$Seed = 3407,
  [int]$SaveInterval = 1,
  [switch]$Amp,
  [switch]$Pretrained,
  [string]$OutDir = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output",
  [string]$RunName = "FCS_0_200_ex1314",
  [int]$StemMin = 0,
  [int]$StemMax = 100,
  # Compatibility (ignored)
  [string]$Task = "",
  [int]$SplitStemMin = -1,
  [int]$SplitStemMax = -1,
  [string]$ExcludeClasses = "13,14",
  # Research-oriented controls
  [switch]$EarlyStop,
  [int]$Patience = 10,
  [double]$MinDelta = 0.0001,
  # Lower default to surface early detections in validation
  [double]$ConfThresh = 0.05,
  # SMNet preprocessing and PF-Aware module
  [switch]$SMNetPreproc = $true,
  [switch]$UsePFAware = $true,
  [switch]$PFAwareLite = $true,
  [double]$PosencScale = 0.0,
  [int]$HeadOutUp = 1,
  [switch]$StrictVal
)

$SRC = Join-Path $PSScriptRoot "src"
$env:PYTHONPATH = "$SRC"

$DatasetRunDir = Join-Path $OutDir $RunName

# 计算实际保存路径 (与 train_b0_det.py 逻辑一致)
$TrainListPath = [System.IO.Path]::GetFullPath($TrainList)
$DatasetDir = Split-Path -Parent $TrainListPath
$RunRoot = Join-Path (Split-Path -Parent $DatasetDir) "CLS_0_200_ex1314"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmm"
$Tag = "Det-B0"
$ActualSaveDir = Join-Path $RunRoot "$Tag-$Timestamp"

Write-Host "Starting B0 DET (Swin) with:" -ForegroundColor Green
Write-Host "  TrainList: $TrainList" -ForegroundColor Yellow
Write-Host "  ValList: $ValList" -ForegroundColor Yellow
Write-Host "  SaveDir: $ActualSaveDir" -ForegroundColor Cyan
Write-Host "  ConfThresh: $ConfThresh" -ForegroundColor Yellow
Write-Host "  SMNet Preproc: $SMNetPreproc" -ForegroundColor Yellow
Write-Host "  PF-Aware: $UsePFAware (Lite: $PFAwareLite)" -ForegroundColor Yellow
Write-Host "  HeadOutUp: $HeadOutUp" -ForegroundColor Yellow

$flags = @(); if ($Amp) { $flags += "--amp" }
if ($Pretrained) { $flags += "--pretrained" }
if ($EarlyStop) { $flags += "--early_stop" }
if ($SMNetPreproc) { $flags += "--smnet_preproc" }
if ($UsePFAware) { $flags += "--use_pfaware" }
if ($PFAwareLite) { $flags += "--pfaware_lite" }
if ($StrictVal) { $flags += "--strict_val" }

conda run -n dronerfa --no-capture-output python "$PSScriptRoot\..\train_b0_det.py" `
  --backbone_type swin `
  --train_list "$TrainList" `
  --val_list "$ValList" `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --num_classes $NumClasses `
  --checkpoint "$SwinCheckpoint" `
  --num_workers $NumWorkers `
  --lr $Lr `
  --seed $Seed `
  --save_interval $SaveInterval `
  --exclude_classes "$ExcludeClasses" `
  --conf_thresh $ConfThresh `
  --patience $Patience `
  --min_delta $MinDelta `
  --posenc_scale $PosencScale `
  --head_out_up $HeadOutUp `
  @flags

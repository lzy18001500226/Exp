param(
  # Lists: each line is a JSON path under VTSLabel
  [string]$TrainList = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output/VTS_0_200_ex1314/train_list_vts.txt",
  [string]$ValList = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output/VTS_0_200_ex1314/val_list_vts.txt",
  [int]$Epochs = 40,
  [int]$BatchSize = 4,  
  [int]$NumClasses = 22,
  [string]$ConvNeXtName = "convnext_base",
  [string]$ConvNeXtCheckpoint = "C:/Users/HP/Desktop/Exp/Model/Pretrain/ConvNeXt/timm/convnext_base.pth",
  [int]$NumWorkers = 0,
  [double]$Lr = 0.001,
  [int]$Seed = 3407,
  [int]$SaveInterval = 1,
  [switch]$Amp,
  [switch]$Pretrained,
  [string]$OutDir = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output",
  [string]$RunName = "VTS_0_200_ex1314",
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
  [switch]$SMNetPreproc = $false,
  [switch]$UsePFAware = $false,
  [switch]$PFAwareLite = $false,
  [double]$PosencScale = 0.0,
  [int]$HeadOutUp = 2,
  [switch]$StrictVal
)

$SRC = Join-Path $PSScriptRoot "src"
$env:PYTHONPATH = "$SRC"

$DatasetRunDir = Join-Path $OutDir $RunName

# ✅ StemMin/StemMax 过滤: 在PS1内直接过滤清单
if ($StemMin -ge 0 -and $StemMax -ge $StemMin) {
    $FilteredTrainList = $TrainList -replace '\.txt$', "_stem$StemMin-$StemMax.txt"
    $FilteredValList = $ValList -replace '\.txt$', "_stem$StemMin-$StemMax.txt"
    
    # 过滤训练集 (使用 [System.IO.File]::WriteAllLines 避免BOM)
    $trainLines = Get-Content $TrainList | Where-Object { 
        if ($_ -match '\\(\d+)\\[^\\]+\.json$') {
            $stem = [int]$Matches[1]
            return ($stem -ge $StemMin -and $stem -le $StemMax)
        }
        return $false
    }
    [System.IO.File]::WriteAllLines($FilteredTrainList, $trainLines, [System.Text.UTF8Encoding]::new($false))
    
    # 过滤验证集
    $valLines = Get-Content $ValList | Where-Object { 
        if ($_ -match '\\(\d+)\\[^\\]+\.json$') {
            $stem = [int]$Matches[1]
            return ($stem -ge $StemMin -and $stem -le $StemMax)
        }
        return $false
    }
    [System.IO.File]::WriteAllLines($FilteredValList, $valLines, [System.Text.UTF8Encoding]::new($false))
    
    $TrainList = $FilteredTrainList
    $ValList = $FilteredValList
    $TrainCount = (Get-Content $TrainList).Count
    $ValCount = (Get-Content $ValList).Count
    Write-Host "  [FILTER] Stem ${StemMin}-${StemMax}: Train=$TrainCount, Val=$ValCount files" -ForegroundColor Cyan
}

# 计算实际保存路径 (与 train_b0_det.py 逻辑一致)
$TrainListPath = [System.IO.Path]::GetFullPath($TrainList)
$DatasetDir = Split-Path -Parent $TrainListPath
$RunRoot = Join-Path (Split-Path -Parent $DatasetDir) "CLS_0_200_ex1314"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmm"
$Tag = "Det-B1"
$ActualSaveDir = Join-Path $RunRoot "$Tag-$Timestamp"

Write-Host "Starting B1 DET (ConvNeXt) with:" -ForegroundColor Green
Write-Host "  TrainList: $TrainList" -ForegroundColor Yellow
Write-Host "  ValList: $ValList" -ForegroundColor Yellow
Write-Host "  SaveDir: $ActualSaveDir" -ForegroundColor Cyan
Write-Host "  Stem Range: $StemMin-$StemMax" -ForegroundColor Magenta
Write-Host "  BatchSize: $BatchSize | Epochs: $Epochs" -ForegroundColor Yellow
Write-Host "  ConfThresh: $ConfThresh (first 5 epochs auto 0.01)" -ForegroundColor Yellow
Write-Host "  SMNet Preproc: $SMNetPreproc" -ForegroundColor Yellow
Write-Host "  PF-Aware: $UsePFAware (Lite: $PFAwareLite)" -ForegroundColor Yellow
Write-Host "  HeadOutUp: $HeadOutUp (recommend 4 with Batch<=2+Amp)" -ForegroundColor Cyan

$flags = @(); if ($Amp) { $flags += "--amp" }
if ($Pretrained) { $flags += "--pretrained" }
if ($EarlyStop) { $flags += "--early_stop" }
if ($SMNetPreproc) { $flags += "--smnet_preproc" }
if ($UsePFAware) { $flags += "--use_pfaware" }
if ($PFAwareLite) { $flags += "--pfaware_lite" }
if ($StrictVal) { $flags += "--strict_val" }

conda run -n dronerfa --no-capture-output python "$PSScriptRoot\..\train_b0_det.py" `
  --backbone_type convnext `
  --train_list "$TrainList" `
  --val_list "$ValList" `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --num_classes $NumClasses `
  --checkpoint "$ConvNeXtCheckpoint" `
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

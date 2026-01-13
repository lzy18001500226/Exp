param(
  # Core training args
  [string]$TrainList = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output/CLS_0_200_ex1314/train_list_cls.txt",
  [string]$ValList = "",
  [string]$Stats = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output/CLS_0_200_ex1314/stats.json",
  [int]$Epochs = 30,
  [int]$BatchSize = 4,
  [int]$EmbedDim = 512,
  [int]$NumClasses = 22,
  [string]$SwinName = "swin_base_patch4_window7_224",
  [string]$SwinCheckpoint = "C:/Users/HP/Desktop/Exp/Model/Pretrain/Swin/timm/swin_base_patch4_window7_224_timm.pth",
  [int]$NumWorkers = 0,
  [double]$Lr = 0.001,
  [int]$Seed = 3407,
  [int]$SaveInterval = 1,
  [int]$StemMin = 0,
  [int]$StemMax = 200,

  # Optim/schedule
  [double]$BackboneLr = 0.0003,
  [double]$HeadLr = 0.003,
  [double]$WeightDecay = 0.01,
  [int]$FreezeBackboneEpochs = 0,
  [double]$LabelSmoothing = 0.05,
  [double]$ClipGrad = 1.0,
  [switch]$Amp,
  [switch]$Pretrained,

  # Augmentations
  [switch]$IsTrainAug,
  [switch]$UseBgMix,
  [double]$BgMixP = 0.5,
  [double]$AlphaMin = 0.1,
  [double]$AlphaMax = 0.3,
  [switch]$UseFreqShift,
  [double]$FreqShiftP = 0.8,
  [double]$ShiftMHz = 10.0,
  [switch]$UseFracFreqShift,
  [switch]$UseTimeStretch,
  [double]$TimeStretchP = 0.8,
  [double]$TimeScaleMin = 0.9,
  [double]$TimeScaleMax = 1.1,
  [switch]$UseNoise,
  [double]$NoiseP = 0.5,
  [double]$NoiseKMin = 0.01,
  [double]$NoiseKMax = 0.05,
  [switch]$UseSpecAug,
  [double]$SpecAugP = 0.5,
  [double]$SpecAugTimeRatio = 0.2,
  [double]$SpecAugFreqRatio = 0.2,
  [int]$SpecAugNumMasks = 2,
  [double]$BandwidthMHz = 0,

  # Misc
  [string]$OutDir = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output",
  [string]$RunName = "",
  [switch]$ComputeStatsStep,
  [switch]$CheckAugs,
  [int]$CheckLimit = 64,
  [switch]$NoStats,
  [string]$RunDirOverride = ""
)

$SRC = Join-Path $PSScriptRoot "src"
$env:PYTHONPATH = "$SRC"

# Defaults
if (-not $PSBoundParameters.ContainsKey('NoStats')) { $NoStats = $true }
$augSwitches = @('IsTrainAug','UseBgMix','UseFreqShift','UseTimeStretch','UseNoise','UseSpecAug')
foreach ($sw in $augSwitches) {
  if (-not $PSBoundParameters.ContainsKey($sw)) { Set-Variable -Name $sw -Value $true -Scope Local }
}
if (-not $PSBoundParameters.ContainsKey('BackboneLr')) { $BackboneLr = 3e-5 }
if (-not $PSBoundParameters.ContainsKey('HeadLr')) { $HeadLr = 3e-3 }
if (-not $PSBoundParameters.ContainsKey('FreezeBackboneEpochs')) { $FreezeBackboneEpochs = 5 }
if (-not $PSBoundParameters.ContainsKey('LabelSmoothing')) { $LabelSmoothing = 0.0 }
if (-not $PSBoundParameters.ContainsKey('ClipGrad')) { $ClipGrad = 0 }

# Auto RunName (CLS only) and file suffix
if ([string]::IsNullOrWhiteSpace($RunName)) {
  $RunName = "FCS_0_200_ex1314"
  Write-Host "Auto RunName: $RunName (train stems controlled by --stem_min/max)" -ForegroundColor Cyan
}
$Suffix = '_fcs'

$RunDir = Join-Path $OutDir $RunName
$TrainListAuto = Join-Path $RunDir "train_list$Suffix.txt"
$ValListAuto = Join-Path $RunDir "val_list$Suffix.txt"
$StatsAuto = Join-Path $RunDir "stats.json"
$AugStatsAuto = Join-Path $RunDir "aug_stats.json"

# Auto adopt existing train/val lists
if ([string]::IsNullOrWhiteSpace($TrainList) -or -not (Test-Path $TrainList)) {
  if (Test-Path $TrainListAuto) {
    $TrainList = $TrainListAuto
    Write-Host "Using TrainList: $TrainList" -ForegroundColor Green
  } else {
    Write-Error "Train list not found: $TrainListAuto. Please generate lists first (see DSFNet/README)."
    return
  }
}
if ([string]::IsNullOrWhiteSpace($ValList) -and (Test-Path $ValListAuto)) {
  $ValList = $ValListAuto
  Write-Host "Using ValList: $ValList" -ForegroundColor Green
}

# Compute dataset stats (optional)
if ($ComputeStatsStep) {
  New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
  $statsLocal = Join-Path $PSScriptRoot 'compute_stats.py'
  $statsAscii = "C:/Users/HP/Desktop/Exp/Tools/compute_stats.py"
  $statsUtf8 = "C:/Users/HP/Desktop/Exp/Tools/鍏ㄥ眬缁熻/compute_stats.py"
  if (-not (Test-Path $statsAscii) -and (Test-Path $statsUtf8)) {
    try { Copy-Item -Path $statsUtf8 -Destination $statsAscii -Force } catch { Write-Warning ("Copy compute_stats.py failed: {0}" -f $_) }
  }
  $statsScript = if (Test-Path $statsLocal) { $statsLocal } elseif (Test-Path $statsAscii) { $statsAscii } else { $statsUtf8 }
  conda run -n dronerfa --no-capture-output python $statsScript --list_file "$TrainList" --save "$StatsAuto"
  $Stats = $StatsAuto
}
if ($NoStats) { $Stats = "" }

# Augmentations self-check (optional)
if ($CheckAugs) {
  New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
  $checkScript = Join-Path $PSScriptRoot 'check_augs_stats.py'
  $argsCheck = @('--train_list', "$TrainList", '--out', "$AugStatsAuto", '--limit', "$CheckLimit")
  if ($BandwidthMHz -gt 0) { $argsCheck += @('--bandwidth_mhz', "$BandwidthMHz") }
  conda run -n dronerfa --no-capture-output python $checkScript @argsCheck
}

# Build CLI args
$flags = @()
if ($Amp) { $flags += "--amp" }
if ($Pretrained) { $flags += "--pretrained" }
if ($IsTrainAug) {
  $flags += "--is_train_aug"
  if ($UseBgMix) { $flags += "--use_bg_mix" }
  if ($UseFreqShift) { $flags += "--use_freq_shift" }
  if ($UseFracFreqShift) { $flags += "--use_frac_freq_shift" }
  if ($UseTimeStretch) { $flags += "--use_time_stretch" }
  if ($UseNoise) { $flags += "--use_noise" }
  if ($UseSpecAug) { $flags += "--use_specaug" }
}
$bwArg = @(); if ($BandwidthMHz -gt 0) { $bwArg = @('--bandwidth_mhz', "$BandwidthMHz") }
$statsArg = @(); if ($Stats -and $Stats.Trim()) { $statsArg = @('--stats', "$Stats") }
$valArg = @(); if ($ValList -and (Test-Path $ValList)) { $valArg = @('--val_list', "$ValList") }
$runDirArg = @(); if ($RunDirOverride -and $RunDirOverride.Trim()) { $runDirArg = @('--run_dir', "$RunDirOverride") }

# Launch
Write-Host "Starting training with:" -ForegroundColor Green
Write-Host "  TrainList: $TrainList" -ForegroundColor Yellow
Write-Host "  ValList: $ValList" -ForegroundColor Yellow
Write-Host "  Stats: $Stats" -ForegroundColor Yellow

conda run -n dronerfa --no-capture-output python "$PSScriptRoot/train_b0_b1.py" `
  --train_list "$TrainList" `
  @statsArg `
  @valArg `
  @runDirArg `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --embed_dim $EmbedDim `
  --num_classes $NumClasses `
  --swin_name $SwinName `
  --swin_checkpoint "$SwinCheckpoint" `
  --num_workers $NumWorkers `
  --lr $Lr `
  --seed $Seed `
  --save_interval $SaveInterval `
  --stem_min $StemMin `
  --stem_max $StemMax `
  --backbone_lr $BackboneLr `
  --head_lr $HeadLr `
  --weight_decay $WeightDecay `
  --freeze_backbone_epochs $FreezeBackboneEpochs `
  --label_smoothing $LabelSmoothing `
  --clip_grad $ClipGrad `
  --bg_mix_p $BgMixP `
  --alpha_min $AlphaMin `
  --alpha_max $AlphaMax `
  --freq_shift_p $FreqShiftP `
  --shift_mhz $ShiftMHz `
  --time_stretch_p $TimeStretchP `
  --time_scale_min $TimeScaleMin `
  --time_scale_max $TimeScaleMax `
  --noise_p $NoiseP `
  --noise_k_min $NoiseKMin `
  --noise_k_max $NoiseKMax `
  --specaug_p $SpecAugP `
  --specaug_time_ratio $SpecAugTimeRatio `
  --specaug_freq_ratio $SpecAugFreqRatio `
  --specaug_num_masks $SpecAugNumMasks `
  @bwArg `
  @flags

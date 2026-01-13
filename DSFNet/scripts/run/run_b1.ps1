# Minimal PowerShell launcher for B1 training (VTS-only via ConvNeXt)
param(
  # Core training args
  [string]$TrainList = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output/VTS_0_200_ex1314/train_list_vts.txt",
  [string]$ValList = "",
  [string]$Stats = "",
  [int]$Epochs = 20,
  [int]$BatchSize = 4,
  [int]$EmbedDim = 512,
  [int]$NumClasses = 22,
  [string]$ConvNeXtName = "convnext_base",
  [string]$ConvNeXtCheckpoint = "C:/Users/HP/Desktop/Exp/Model/Pretrain/ConvNeXt/timm/convnext_base.pth",
  [int]$NumWorkers = 0,
  [double]$Lr = 0.001,
  [int]$Seed = 3407,
  [int]$SaveInterval = 1,
  [int]$StemMin = 0,
  [int]$StemMax = 200,

  # Optim/schedule
  [double]$BackboneLr = 0.00005,
  [double]$HeadLr = 0.0005,
  [double]$WeightDecay = 0.01,
  [int]$FreezeBackboneEpochs = 5,
  [double]$LabelSmoothing = 0.1,
  [double]$ClipGrad = 1.0,
  [switch]$Amp,
  [switch]$Pretrained,

  # Augmentations (same as B0)
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
if (-not $PSBoundParameters.ContainsKey('BackboneLr')) { $BackboneLr = 5e-5 }
if (-not $PSBoundParameters.ContainsKey('HeadLr')) { $HeadLr = 5e-4 }
if (-not $PSBoundParameters.ContainsKey('FreezeBackboneEpochs')) { $FreezeBackboneEpochs = 5 }
if (-not $PSBoundParameters.ContainsKey('LabelSmoothing')) { $LabelSmoothing = 0.1 }
if (-not $PSBoundParameters.ContainsKey('ClipGrad')) { $ClipGrad = 1.0 }

# Auto RunName (VTS only)
if ([string]::IsNullOrWhiteSpace($RunName)) {
  $RunName = "VTS_0_200_ex1314"
  Write-Host "Auto RunName: $RunName (train stems controlled by --stem_min/max)" -ForegroundColor Cyan
}
$Suffix = '_vts'

$RunDir = Join-Path $OutDir $RunName
$TrainListAuto = Join-Path $RunDir "train_list$Suffix.txt"
$ValListAuto = Join-Path $RunDir "val_list$Suffix.txt"

# Auto adopt existing train/val lists
if ([string]::IsNullOrWhiteSpace($TrainList) -or -not (Test-Path $TrainList)) {
  if (Test-Path $TrainListAuto) { $TrainList = $TrainListAuto; Write-Host "Using TrainList: $TrainList" -ForegroundColor Green } else { Write-Error "Train list not found: $TrainListAuto"; return }
}
if ([string]::IsNullOrWhiteSpace($ValList) -and (Test-Path $ValListAuto)) { $ValList = $ValListAuto; Write-Host "Using ValList: $ValList" -ForegroundColor Green }
if ($NoStats) { $Stats = "" }

# Build CLI args
$flags = @(); if ($Amp) { $flags += "--amp" }
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

# Launch (train.py 仍使用 BaseFusionModel/Swin 流程，此处暂用同入口；B1 将在 train.py 中通过一个开关分支到 ConvNeXtEncoder)
Write-Host "Starting B1 training (ConvNeXt, VTS-only) with:" -ForegroundColor Green
Write-Host "  TrainList: $TrainList" -ForegroundColor Yellow
Write-Host "  ValList: $ValList" -ForegroundColor Yellow
Write-Host "  Stats: $Stats" -ForegroundColor Yellow
Write-Host "  ConvNeXt ckpt: $ConvNeXtCheckpoint (use -Pretrained to fallback timm cache)" -ForegroundColor Yellow

conda run -n dronerfa --no-capture-output python "$PSScriptRoot/train_b0_b1.py" `
  --train_list "$TrainList" `
  @statsArg `
  @valArg `
  @runDirArg `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --embed_dim $EmbedDim `
  --num_classes $NumClasses `
  --swin_name $ConvNeXtName `
  --swin_checkpoint "$ConvNeXtCheckpoint" `
  --backbone_type convnext `
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

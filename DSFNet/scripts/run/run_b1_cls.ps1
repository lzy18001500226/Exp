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
  [double]$BackboneLr = 5e-5,
  [double]$HeadLr = 5e-4,
  [double]$WeightDecay = 0.01,
  [int]$FreezeBackboneEpochs = 5,
  [double]$LabelSmoothing = 0.1,
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
  [int]$CheckLimit = 64
)

$SRC = Join-Path $PSScriptRoot "src"
$env:PYTHONPATH = "$SRC"

# Defaults
$augSwitches = @('IsTrainAug','UseBgMix','UseFreqShift','UseTimeStretch','UseNoise','UseSpecAug')
foreach ($sw in $augSwitches) { if (-not $PSBoundParameters.ContainsKey($sw)) { Set-Variable -Name $sw -Value $true -Scope Local } }
if (-not $PSBoundParameters.ContainsKey('BackboneLr')) { $BackboneLr = 5e-5 }
if (-not $PSBoundParameters.ContainsKey('HeadLr')) { $HeadLr = 5e-4 }
if (-not $PSBoundParameters.ContainsKey('FreezeBackboneEpochs')) { $FreezeBackboneEpochs = 5 }
if (-not $PSBoundParameters.ContainsKey('LabelSmoothing')) { $LabelSmoothing = 0.1 }
if (-not $PSBoundParameters.ContainsKey('ClipGrad')) { $ClipGrad = 1.0 }

# Auto RunName (VTS only)
if ([string]::IsNullOrWhiteSpace($RunName)) { $RunName = "VTS_0_200_ex1314"; Write-Host "Auto RunName: $RunName" -ForegroundColor Cyan }
$Suffix = '_vts'

$DatasetRunDir = Join-Path $OutDir $RunName
$TrainListAuto = Join-Path $DatasetRunDir "train_list$Suffix.txt"
$ValListAuto = Join-Path $DatasetRunDir "val_list$Suffix.txt"

# Adopt existing train/val lists
if ([string]::IsNullOrWhiteSpace($TrainList) -or -not (Test-Path $TrainList)) { if (Test-Path $TrainListAuto) { $TrainList = $TrainListAuto; Write-Host "Using TrainList: $TrainList" -ForegroundColor Green } else { Write-Error "Train list not found: $TrainListAuto"; return } }
if ([string]::IsNullOrWhiteSpace($ValList) -and (Test-Path $ValListAuto)) { $ValList = $ValListAuto; Write-Host "Using ValList: $ValList" -ForegroundColor Green }

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

Write-Host "Starting B1 CLS training with:" -ForegroundColor Green
Write-Host "  TrainList: $TrainList" -ForegroundColor Yellow
Write-Host "  ValList: $ValList" -ForegroundColor Yellow
Write-Host "  SaveDir: auto (CLS_*/Cls-B1-****)" -ForegroundColor Yellow

conda run -n dronerfa --no-capture-output python "$PSScriptRoot/train_b1_cls.py" `
  --backbone_type convnext `
  --train_list "$TrainList" `
  @statsArg `
  @valArg `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --embed_dim $EmbedDim `
  --num_classes $NumClasses `
  --swin_name $ConvNeXtName `
  --swin_checkpoint "$ConvNeXtCheckpoint" `
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
  @bwArg `
  @flags

# Minimal PowerShell launcher for B2 training (FCS+VTS concat)
param(
  # Core training args (paired lists)
  [string]$TrainListFCS = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output/FCS_VTS_0_200_ex1314/train_list_fcs.txt",
  [string]$VTSList = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output/FCS_VTS_0_200_ex1314/train_list_vts.txt",
  [string]$ValListFCS = "",
  [string]$ValListVTS = "",
  [int]$Epochs = 30,
  [int]$BatchSize = 2,
  [int]$EmbedDim = 512,
  [int]$NumClasses = 22,
  [ValidateSet('swin','convnext')]
  [string]$Backbone = 'swin',
  [int]$NumWorkers = 0,
  [double]$Lr = 0.001,
  [int]$Seed = 3407,
  [int]$SaveInterval = 1,
  [int]$StemMin = 0,
  [int]$StemMax = 200,

  # Optim/schedule
  [double]$BackboneLr = 3e-5,
  [double]$HeadLr = 3e-3,
  [double]$WeightDecay = 0.01,
  [int]$FreezeBackboneEpochs = 5,
  [double]$LabelSmoothing = 0.0,
  [double]$ClipGrad = 0,
  [switch]$Amp,

  # Augmentations (enabled by default like B0/B1)
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
  [string]$PairRoot = "C:/Users/HP/Desktop/Exp/Model/DSFNet-Output/FCS_VTS_0_200_ex1314",
  [string]$RunDirOverride = "",
  [switch]$NoStats
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

# Auto adopt lists under PairRoot
$TrainListAutoFCS = Join-Path $PairRoot "train_list_fcs.txt"
$TrainListAutoVTS = Join-Path $PairRoot "train_list_vts.txt"
$ValListAutoFCS = Join-Path $PairRoot "val_list_fcs.txt"
$ValListAutoVTS = Join-Path $PairRoot "val_list_vts.txt"

if ([string]::IsNullOrWhiteSpace($TrainListFCS) -or -not (Test-Path $TrainListFCS)) {
  if (Test-Path $TrainListAutoFCS) { $TrainListFCS = $TrainListAutoFCS; Write-Host "Using TrainList(FCS): $TrainListFCS" -ForegroundColor Green } else { Write-Error "FCS train list not found: $TrainListAutoFCS"; return }
}
if ([string]::IsNullOrWhiteSpace($VTSList) -or -not (Test-Path $VTSList)) {
  if (Test-Path $TrainListAutoVTS) { $VTSList = $TrainListAutoVTS; Write-Host "Using TrainList(VTS): $VTSList" -ForegroundColor Green } else { Write-Error "VTS train list not found: $TrainListAutoVTS"; return }
}
if ([string]::IsNullOrWhiteSpace($ValListFCS) -and (Test-Path $ValListAutoFCS)) {
  $ValListFCS = $ValListAutoFCS; Write-Host "Using ValList(FCS): $ValListFCS" -ForegroundColor Green
}
if ([string]::IsNullOrWhiteSpace($ValListVTS) -and (Test-Path $ValListAutoVTS)) {
  $ValListVTS = $ValListAutoVTS; Write-Host "Using ValList(VTS): $ValListVTS" -ForegroundColor Green
}

# Default RunDir when not overridden (workspace root = one level above DSFNet)
$DSFNetRoot = Split-Path $PSScriptRoot -Parent
$WorkspaceRoot = Split-Path $DSFNetRoot -Parent
$RunRoot = Join-Path (Join-Path $WorkspaceRoot "Model") "DSFNet-Output/CLS_0_200_ex1314"
$TimeStr = Get-Date -Format "yyyyMMdd-HHmm"
$RunDirDefault = Join-Path $RunRoot "Run-B2-$TimeStr"

# Build CLI args
$flags = @(); if ($Amp) { $flags += "--amp" }
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
$valArg = @(); if ($ValListFCS -and (Test-Path $ValListFCS)) { $valArg = @('--val_list', "$ValListFCS") }
$vtsValArg = @(); if ($ValListVTS -and (Test-Path $ValListVTS)) { $vtsValArg = @('--vts_val_list', "$ValListVTS") }
$runDirPath = $RunDirOverride; if (-not $runDirPath -or -not $runDirPath.Trim()) { $runDirPath = $RunDirDefault }
$runDirArg = @('--run_dir', "$runDirPath")

Write-Host "Starting B2 training (concat) with:" -ForegroundColor Green
Write-Host "  TrainList(FCS): $TrainListFCS" -ForegroundColor Yellow
Write-Host "  TrainList(VTS): $VTSList" -ForegroundColor Yellow
Write-Host "  ValList(FCS):   $ValListFCS" -ForegroundColor Yellow
Write-Host "  ValList(VTS):   $ValListVTS" -ForegroundColor Yellow
Write-Host "  RunDir:         $runDirPath" -ForegroundColor Yellow
Write-Host "  Backbone:       $Backbone" -ForegroundColor Yellow

conda run -n dronerfa --no-capture-output python "$PSScriptRoot/train_b2.py" `
  --mode concat `
  --train_list "$TrainListFCS" `
  --vts_list "$VTSList" `
  @valArg `
  @vtsValArg `
  @runDirArg `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --embed_dim $EmbedDim `
  --num_classes $NumClasses `
  --backbone_type $Backbone `
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

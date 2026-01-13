param(
  [string]$TrainList = "D:/Exp/Model/DSFNet-Output/coco/train_json.list",
  [string]$ValList   = "D:/Exp/Model/DSFNet-Output/coco/val_json.list",
  [int]$Epochs = 60,
  [int]$BatchSize = 6,
  [int]$NumClasses = 22,
  [string]$SwinName = "swin_base_patch4_window7_224",
  [string]$SwinCheckpoint = "D:/Exp/Model/Pretrain/Swin/timm/swin_base_patch4_window7_224_timm.pth",
  [int]$NumWorkers = 0,
  [double]$Lr = 1e-3,
  [int]$Seed = 3407,
  [int]$SaveInterval = 1,
  [switch]$Amp,
  [switch]$Pretrained,
  [string]$OutDir = "D:/Exp/Model/DSFNet-Output",
  [string]$RunName = "FCS_0_200_ex1314",
  [string]$ExcludeClasses = "13,14",
  [switch]$EarlyStop,
  [int]$Patience = 12,
  [double]$MinDelta = 0.0005,
  [double]$ConfThresh = 0.05,
  # SMNet core toggles
  [switch]$SMNetPreproc,
  [switch]$UsePFAware,
  [switch]$PFAwareLite,
  [double]$PosencScale = 0.05,
  [double]$PFAwareGateTarget = 1.0,
  [int]$HeadOutUp = 3,
  [switch]$StrictVal,
  # Decode/static thresholds (for FCS small-object friendly defaults)
  [switch]$DisableCurriculum,
  [string]$DecodeScales = "p2,p3",
  [double]$DecodeConf = 0.03,
  [int]$DecodeTopK = 120,
  [double]$DecodeMinWH = 2.0,
  [double]$DecodeNMS = 0.5,
  [double]$MaxWhP2 = 64,
  [double]$MaxWhP3 = 160,
  [double]$MaxWhP4 = 512,
  # Loss weights
  [double]$LossHm = 1.0,
  [double]$LossWh = 0.5,
  [double]$LossOff = 1.0,
  # Data augmentation control
  [switch]$DetNoAug,
  # Diagnostics
  [switch]$EnableGradMonitor
)

# UTF-8 and Python I/O on Windows PowerShell 5.1
try { if ($PSVersionTable.PSVersion.Major -ge 7) { $PSStyle.OutputRendering = 'PlainText' } } catch {}
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false); $OutputEncoding = [Console]::OutputEncoding } catch {}
try { cmd /c chcp 65001 > $null } catch {}
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$SRC = Join-Path $PSScriptRoot "src"
$env:PYTHONPATH = "$SRC"

# Enable recommended defaults if not explicitly provided
if (-not $PSBoundParameters.ContainsKey('SMNetPreproc')) { $SMNetPreproc = $true }
if (-not $PSBoundParameters.ContainsKey('UsePFAware')) { $UsePFAware = $true }
if (-not $PSBoundParameters.ContainsKey('PFAwareLite')) { $PFAwareLite = $true }
# 设定建议默认：FCS 默认小目标友好静态解码（如用于 VTS，请自行改为 `DecodeScales="all"` 并移除 DisableCurriculum）
if (-not $PSBoundParameters.ContainsKey('DisableCurriculum')) { $DisableCurriculum = $true }

# Compute save dir consistent with train_b0_det.py
$TrainListPath = [System.IO.Path]::GetFullPath($TrainList)
$DatasetDir = Split-Path -Parent $TrainListPath
$RunRoot = Join-Path (Split-Path -Parent $DatasetDir) "CLS_0_200_ex1314"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmm"
$Tag = "Det-B0"
$ActualSaveDir = Join-Path $RunRoot "$Tag-$Timestamp"

Write-Host "Starting SMNet DET (Swin+B0) with:" -ForegroundColor Green
Write-Host "  TrainList: $TrainList" -ForegroundColor Yellow
Write-Host "  ValList:   $ValList" -ForegroundColor Yellow
Write-Host "  SaveDir:   $ActualSaveDir" -ForegroundColor Cyan
Write-Host "  SMNet Preproc: $SMNetPreproc | PF-Aware: $UsePFAware (Lite: $PFAwareLite) | PosEnc: $PosencScale" -ForegroundColor Yellow
Write-Host "  HeadOutUp: $HeadOutUp | ConfThresh: $ConfThresh | PFAwareGateTarget: $PFAwareGateTarget" -ForegroundColor Yellow
Write-Host "  Decode: disable_curriculum=$DisableCurriculum, scales=$DecodeScales, conf=$DecodeConf, topk=$DecodeTopK, min_wh=$DecodeMinWH, nms=$DecodeNMS" -ForegroundColor Yellow
Write-Host "  Loss: hm=$LossHm, wh=$LossWh, off=$LossOff | DetNoAug=$DetNoAug" -ForegroundColor Yellow

$flags = @()
if ($Amp) { $flags += "--amp" }
if ($Pretrained) { $flags += "--pretrained" }
if ($EarlyStop) { $flags += "--early_stop" }
if ($SMNetPreproc) { $flags += "--smnet_preproc" }
if ($UsePFAware) { $flags += "--use_pfaware" }
if ($PFAwareLite) { $flags += "--pfaware_lite" }
if ($StrictVal) { $flags += "--strict_val" }
if ($EnableGradMonitor) { $flags += "--enable_grad_monitor" }
if ($DisableCurriculum) { $flags += "--disable_curriculum" }
if ($DetNoAug) { $flags += "--det_no_aug" }

# Use the user's conda env if available; otherwise fall back to python on PATH
$pythonLauncher = $null
try {
  # Check if conda is available on PATH
  $condaCmd = Get-Command conda -ErrorAction Stop
  $pythonLauncher = @('conda','run','-n','dronerfa','--no-capture-output','python')
} catch {
  # fallback to system python
  $pythonLauncher = @('python')
}

# Build python invocation: executable + pre-args + script + script-args
$exe = $pythonLauncher[0]
$preArgs = if ($pythonLauncher.Count -gt 1) { $pythonLauncher[1..($pythonLauncher.Count - 1)] } else { @() }
$scriptPath = (Join-Path $PSScriptRoot "..\train_b0_det.py")

$scriptArgs = @(
  '--backbone_type','swin',
  '--train_list',$TrainList,
  '--val_list',$ValList,
  '--epochs',$Epochs,
  '--batch_size',$BatchSize,
  '--num_classes',$NumClasses,
  '--checkpoint',$SwinCheckpoint,
  '--num_workers',$NumWorkers,
  '--lr',$Lr,
  '--seed',$Seed,
  '--save_interval',$SaveInterval,
  '--exclude_classes',$ExcludeClasses,
  '--conf_thresh',$ConfThresh,
  '--patience',$Patience,
  '--min_delta',$MinDelta,
  '--posenc_scale',$PosencScale,
  '--pfaware_gate_target',$PFAwareGateTarget,
  '--head_out_up',$HeadOutUp,
  '--decode_scales',$DecodeScales,
  '--decode_conf',$DecodeConf,
  '--decode_topk',$DecodeTopK,
  '--decode_min_wh',$DecodeMinWH,
  '--decode_nms',$DecodeNMS,
  '--max_wh_p2',$MaxWhP2,
  '--max_wh_p3',$MaxWhP3,
  '--max_wh_p4',$MaxWhP4,
  '--loss_hm',$LossHm,
  '--loss_wh',$LossWh,
  '--loss_off',$LossOff
)

# Append flag-style switches
if ($flags.Count -gt 0) { $scriptArgs += $flags }

# Invoke: executable, its pre-arguments (if any), the script path, then script args
Write-Host "Invoking: $exe $($preArgs -join ' ') $scriptPath $($scriptArgs -join ' ')" -ForegroundColor DarkCyan
if ($preArgs.Count -gt 0) {
  & $exe @preArgs $scriptPath @scriptArgs
} else {
  & $exe $scriptPath @scriptArgs
}

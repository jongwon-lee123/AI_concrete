param(
  [Parameter(Mandatory=$true)]
  [string]$ImageDir,

  [double]$BrassOdMm = 250,
  [double]$MoldBaseMm = 100,
  [string]$OutDir = "",
  [switch]$Debug
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FlowPy = Join-Path $ScriptDir "flow.py"

if (-not (Test-Path $FlowPy)) {
  Write-Error "flow.py not found: $FlowPy"
  exit 1
}

if (-not (Test-Path $ImageDir)) {
  Write-Error "Image directory not found: $ImageDir"
  exit 1
}

$ResolvedImageDir = (Resolve-Path $ImageDir).Path
if ([string]::IsNullOrWhiteSpace($OutDir)) {
  # Default: save right next to input images.
  $OutDir = $ResolvedImageDir
}

$Args = @(
  $FlowPy,
  "--image-dir", $ResolvedImageDir,
  "--brass-od-mm", $BrassOdMm,
  "--mold-base-mm", $MoldBaseMm,
  "--out-dir", $OutDir
)

if ($Debug) {
  $Args += "--debug"
  Write-Host "[debug] flow.py=$FlowPy"
  Write-Host "[debug] image_dir=$ResolvedImageDir"
  Write-Host "[debug] out_dir=$OutDir"
}

python @Args

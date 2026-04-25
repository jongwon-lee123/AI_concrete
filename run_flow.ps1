param(
  [Parameter(Mandatory=$true)]
  [string]$ImageDir,

  [double]$BrassOdMm = 250,
  [double]$MoldBaseMm = 100,
  [string]$OutDir = ".\out",
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

$Args = @(
  $FlowPy,
  "--image-dir", $ImageDir,
  "--brass-od-mm", $BrassOdMm,
  "--mold-base-mm", $MoldBaseMm,
  "--out-dir", $OutDir
)

if ($Debug) {
  $Args += "--debug"
  Write-Host "[debug] flow.py=$FlowPy"
  Write-Host "[debug] image_dir=$ImageDir"
  Write-Host "[debug] out_dir=$OutDir"
}

python @Args

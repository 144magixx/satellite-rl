param(
    [Parameter(Mandatory=$true)][string]$Pins,
    [Parameter(Mandatory=$true)][string]$PinsSha
)
$ErrorActionPreference = 'Stop'
$pinsPath = (Resolve-Path -LiteralPath $Pins).Path
if ((Get-FileHash -LiteralPath $pinsPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PinsSha.ToLowerInvariant()) { throw 'Root pins SHA differs' }
$contract = Get-Content -LiteralPath $pinsPath -Raw -Encoding UTF8 | ConvertFrom-Json
function Read-PinnedJson($reference) {
    if ((Get-FileHash -LiteralPath $reference.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $reference.sha256) { throw 'Root reference SHA differs' }
    return Get-Content -LiteralPath $reference.path -Raw -Encoding UTF8 | ConvertFrom-Json
}
$admission = Read-PinnedJson $contract.controller_launch_admission_ref
$delegation = Read-PinnedJson $contract.delegation_ref
if (-not $admission.root_approved -or -not $admission.passed -or $admission.phase -ne $delegation.mode -or $admission.plan_sha256 -ne $contract.plan_ref.sha256) { throw 'Separate root controller launch admission required' }
if ($admission.adapter_source_sha256 -ne $contract.adapter_source_sha256 -or $admission.controller_source_sha256 -ne $delegation.controller_source_sha256) { throw 'Root launch source binding differs' }
$scriptSha = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($scriptSha -ne $contract.launcher_source_sha256 -or $scriptSha -ne $admission.launcher_source_sha256) { throw 'Launcher source differs from root pin' }
$sourceRoot = (Resolve-Path -LiteralPath $contract.local_source_root).Path
$adapterPath = Join-Path $sourceRoot 'src/satellite_rl/power30/orchestration_adapter.py'
$controllerPath = Join-Path $sourceRoot 'src/satellite_rl/power30/orchestration.py'
if ((Get-FileHash -LiteralPath $adapterPath).Hash.ToLowerInvariant() -ne $contract.adapter_source_sha256 -or (Get-FileHash -LiteralPath $controllerPath).Hash.ToLowerInvariant() -ne $delegation.controller_source_sha256) { throw 'Actual local source differs' }
if (-not $env:BUPT_PASSWORD) { throw 'BUPT_PASSWORD environment credential missing' }
if (-not $contract.exclusive_stage_directories -or $contract.exclusive_stage_directories.Count -lt 2) { throw 'Both engineering and scientific stage directories required' }
foreach ($directory in $contract.exclusive_stage_directories) {
    if (Test-Path -LiteralPath (Join-Path $directory 'singleton')) { throw 'Existing stage owner or stale lock needs root reconciliation; never steal it' }
}
foreach ($pausePath in $contract.local_pause_markers) { if (Test-Path -LiteralPath $pausePath) { throw 'New human pause overrides launch' } }
if (-not $contract.local_pause_markers) { throw 'Local pause contract missing' }
$python = (Resolve-Path -LiteralPath $contract.local_python).Path
if ($pinsPath.Contains('"')) { throw 'Unsupported quote in pins path' }
$oldPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $sourceRoot 'src'
    $arguments = @('-m','satellite_rl.power30.orchestration_adapter','--pins',('"' + $pinsPath + '"'),'--pins-sha',$PinsSha)
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $sourceRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $contract.stdout_path -RedirectStandardError $contract.stderr_path
} finally { $env:PYTHONPATH = $oldPythonPath }
$ownerPath = Join-Path $contract.controller_directory 'singleton/owner.json'
for ($index=0; $index -lt 50 -and -not (Test-Path -LiteralPath $ownerPath); $index++) {
    if ($process.HasExited) { throw 'Controller exited before singleton qualification; inspect its saved status/logs' }
    Start-Sleep -Milliseconds 200
    $process.Refresh()
}
if (-not (Test-Path -LiteralPath $ownerPath)) { throw 'No singleton owner receipt; do not relaunch blindly' }
$owner = Get-Content -LiteralPath $ownerPath -Raw -Encoding UTF8 | ConvertFrom-Json
$inspection = & $python -c 'import json,psutil,sys; sys.path.insert(0,sys.argv[2]); from satellite_rl.power30.orchestration_adapter import local_boot_matches; p=psutil.Process(int(sys.argv[1])); observed="boot-time:"+str(psutil.boot_time()); print(json.dumps(dict(pid=p.pid,process_birth=p.create_time(),boot_id=observed,owner_boot_matches=local_boot_matches(sys.argv[3],observed),executable=p.exe())))' $process.Id (Join-Path $sourceRoot 'src') $owner.boot_id | ConvertFrom-Json
if ($owner.pid -ne $process.Id -or $owner.process_birth -ne $inspection.process_birth -or -not $inspection.owner_boot_matches -or $inspection.executable -ne $python) { throw 'Actual controller PID/birth/boot/interpreter differs; do not steal/retry' }
[pscustomobject]@{pid=$process.Id; process_birth=$inspection.process_birth; boot_id=$inspection.boot_id; launcher_sha256=$scriptSha; pins_sha256=$PinsSha; status='hidden_controller_owner_verified'} | ConvertTo-Json -Compress

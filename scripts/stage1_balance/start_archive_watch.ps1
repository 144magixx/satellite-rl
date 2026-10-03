param(
    [int]$IntervalSeconds = 180
)

$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot 'archive_completed.py'
$output = 'D:\paper\outputs\experiments\first_stage_s1_02_20261004\preflight'
$receipt = Join-Path $output 'archive_watch_launch.json'
$python = 'C:\Users\Admin\anaconda3\python.exe'

if ($IntervalSeconds -lt 30 -or $IntervalSeconds -gt 3600) {
    throw 'IntervalSeconds must be 30–3600.'
}
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw 'Archive watcher script is missing.'
}
if (Test-Path -LiteralPath $receipt) {
    $prior = Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($prior.pid)" -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -like "*$scriptPath*" -and
        $process.CreationDate.ToUniversalTime().ToString('o') -eq $prior.created_utc) {
        $prior | ConvertTo-Json -Depth 5
        exit 0
    }
    $watchStatus = Join-Path $output 'archive_watch_status.json'
    if (Test-Path -LiteralPath $watchStatus) {
        $state = Get-Content -LiteralPath $watchStatus -Raw | ConvertFrom-Json
        if ($state.status -eq 'mirror_complete_pending_final_delivery') {
            $prior | ConvertTo-Json -Depth 5
            exit 0
        }
    }
    throw 'Previous archive watcher is absent or changed; inspect its status before restart.'
}

$arguments = @('-u', $scriptPath, '--watch', '--interval-seconds', [string]$IntervalSeconds)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory 'D:\paper' -WindowStyle Hidden -RedirectStandardOutput (Join-Path $output 'archive_watch.stdout.log') -RedirectStandardError (Join-Path $output 'archive_watch.stderr.log') -PassThru
$identity = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)"
$result = [ordered]@{
    status = 'started'
    pid = $process.Id
    created_utc = $identity.CreationDate.ToUniversalTime().ToString('o')
    script = $scriptPath
    interval_seconds = $IntervalSeconds
    started_utc = [DateTimeOffset]::UtcNow.ToString('o')
}
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receipt -Encoding utf8
$result | ConvertTo-Json -Depth 5

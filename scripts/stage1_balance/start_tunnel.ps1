$ErrorActionPreference = 'Stop'
$output = 'D:\paper\outputs\experiments\first_stage_s1_02_20261004\preflight'
$receipt = Join-Path $output 'tunnel_8892_launch.json'
$script = 'C:\Users\Admin\.codex\worktrees\stage1-balance-actions\paper\scripts\remote_stage1.py'
$python = 'C:\Users\Admin\anaconda3\python.exe'

$existing = Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort 8892 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    throw 'Local port 8892 is already listening; inspect its owner.'
}
if (Test-Path -LiteralPath $receipt) {
    throw 'A tunnel launch receipt already exists; inspect it before restarting.'
}

$arguments = @('-u', $script, 'tunnel', '--host', '10.160.17.102', '--port',
    '20418', '--user', 'u2024110224', '--local-port', '8892', '--remote-port', '8892')
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory 'D:\paper' -WindowStyle Hidden -RedirectStandardOutput (Join-Path $output 'tunnel_8892.stdout.log') -RedirectStandardError (Join-Path $output 'tunnel_8892.stderr.log') -PassThru
$identity = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)"
$result = [ordered]@{
    status = 'started'
    pid = $process.Id
    created_utc = $identity.CreationDate.ToUniversalTime().ToString('o')
    local_url = 'http://127.0.0.1:8892/#coverage'
    remote_target = 'u2024110224@10.160.17.102:20418'
    remote_port = 8892
}
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receipt -Encoding utf8
$result | ConvertTo-Json -Depth 5

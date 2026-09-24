param(
    [int]$Port = 8765,
    [string]$ArchiveRoot = '',
    [string]$DqnPython = "$env:USERPROFILE\anaconda3\envs\DQN\python.exe",
    [int]$MaxWorkers = 1,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$workbenchRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $ArchiveRoot) { $ArchiveRoot = $workbenchRoot }
$ArchiveRoot = [System.IO.Path]::GetFullPath($ArchiveRoot)
if (-not (Test-Path -LiteralPath $DqnPython -PathType Leaf)) { throw "DQN Python不存在：$DqnPython" }
if ($Port -lt 1024 -or $Port -gt 65535) { throw '端口必须在1024–65535之间' }
$workbenchUrl = "http://127.0.0.1:$Port/"
try {
    $workbenchExisting = Invoke-RestMethod -Uri ($workbenchUrl + 'api/system') -TimeoutSec 3
} catch { $workbenchExisting = $null }
if ($workbenchExisting) {
    if ([System.IO.Path]::GetFullPath($workbenchExisting.archive_root) -ne $ArchiveRoot) {
        throw '该端口正在服务另一个实验目录，请指定其他端口'
    }
    Write-Host "工作台已运行：$workbenchUrl"
    if (-not $NoBrowser) { Start-Process $workbenchUrl }
    exit 0
}
$workbenchLogRoot = Join-Path $ArchiveRoot 'outputs\workbench\service'
New-Item -ItemType Directory -Force -Path $workbenchLogRoot | Out-Null
$workbenchLogStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
foreach ($workbenchLogName in @('server.stdout.log', 'server.stderr.log')) {
    $workbenchPreviousLog = Join-Path $workbenchLogRoot $workbenchLogName
    if (Test-Path -LiteralPath $workbenchPreviousLog -PathType Leaf) {
        Copy-Item -LiteralPath $workbenchPreviousLog -Destination (Join-Path $workbenchLogRoot ($workbenchLogName + '.' + $workbenchLogStamp))
    }
}
$workbenchOldPythonPath = $env:PYTHONPATH
$workbenchOldUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONPATH = Join-Path $workbenchRoot 'src'
    $env:PYTHONUTF8 = '1'
    $workbenchArguments = @('-u', '-m', 'satellite_rl.workbench.server', '--archive-root', ('"' + $ArchiveRoot + '"'), '--port', $Port, '--python', ('"' + $DqnPython + '"'), '--max-workers', $MaxWorkers)
    $workbenchProcess = Start-Process -FilePath $DqnPython -ArgumentList $workbenchArguments -WorkingDirectory $workbenchRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $workbenchLogRoot 'server.stdout.log') -RedirectStandardError (Join-Path $workbenchLogRoot 'server.stderr.log') -PassThru
    $workbenchProcess.Id | Set-Content -LiteralPath (Join-Path $workbenchLogRoot 'server.pid') -Encoding ascii
} finally {
    $env:PYTHONPATH = $workbenchOldPythonPath
    $env:PYTHONUTF8 = $workbenchOldUtf8
}
$workbenchReady = $false
for ($workbenchAttempt = 0; $workbenchAttempt -lt 30; $workbenchAttempt++) {
    try {
        $null = Invoke-RestMethod -Uri ($workbenchUrl + 'api/catalog') -TimeoutSec 2
        $workbenchReady = $true
        break
    } catch {
        if ($workbenchProcess.HasExited) { break }
        Start-Sleep -Milliseconds 500
    }
}
if (-not $workbenchReady) { throw "工作台启动失败，请查看 $workbenchLogRoot\server.stderr.log" }
Write-Host "卫星资源分配实验工作台：$workbenchUrl"
Write-Host "进程：$($workbenchProcess.Id)；实验保存目录：$ArchiveRoot"
if (-not $NoBrowser) { Start-Process $workbenchUrl }

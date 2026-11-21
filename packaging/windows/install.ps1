# 安装 obs-auto-mosaic 到 OBS 的第三方插件目录（需要管理员权限）
# 用法：在本文件所在目录运行  powershell -ExecutionPolicy Bypass -File .\install.ps1
$ErrorActionPreference = "Stop"

$src = Join-Path $PSScriptRoot "obs-face-mosaic"
$dstRoot = Join-Path $env:ProgramData "obs-studio\plugins"
$dst = Join-Path $dstRoot "obs-face-mosaic"

if (-not (Test-Path (Join-Path $src "bin\64bit\obs-face-mosaic.dll"))) {
    Write-Host "找不到 obs-face-mosaic 文件夹，请在解压后的目录里运行本脚本。" -ForegroundColor Red
    exit 1
}
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "请用「以管理员身份运行」的 PowerShell / 终端执行本脚本。" -ForegroundColor Red
    exit 1
}
if (Get-Process obs64 -ErrorAction SilentlyContinue) {
    Write-Host "OBS 正在运行，请先完全退出 OBS 再安装。" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path $dstRoot | Out-Null
# 覆盖安装：先删掉旧版本的程序文件（保留 ort-cuda，方便只更新主包）
if (Test-Path $dst) {
    Get-ChildItem (Join-Path $dst "bin\64bit") -File -ErrorAction SilentlyContinue | Remove-Item -Force
    Remove-Item -Recurse -Force (Join-Path $dst "bin\64bit\ort-dml") -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Join-Path $dst "data") -ErrorAction SilentlyContinue
}
Copy-Item -Recurse -Force $src $dstRoot
Write-Host "已安装到 $dst" -ForegroundColor Green
Write-Host "打开 OBS，在来源上右键 →「滤镜」→「效果滤镜」→「+」→「自动马赛克」。"

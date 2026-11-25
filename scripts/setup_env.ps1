# scripts/setup_env.ps1 - Tạo/ sửa môi trường ảo của project (Windows + PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1        # cài PyTorch bản GPU (CUDA 12.1)
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -Cpu   # cài PyTorch bản CPU (chỉ chạy thử)
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -Recreate  # xoá venv cũ rồi làm lại từ đầu
#
# Sau khi chạy: venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
param(
    [switch]$Cpu,
    [switch]$Recreate
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root 'venv'
$Py = Join-Path $Venv 'Scripts\python.exe'
$TorchVersion = '2.5.1+cu121'
$CudaIndex = 'https://download.pytorch.org/whl/cu121'
$CpuIndex = 'https://download.pytorch.org/whl/cpu'

Write-Host "Project : $Root" -ForegroundColor Cyan
Write-Host "Venv    : $Venv"

if ($Recreate -and (Test-Path $Venv)) {
    Write-Host 'Xoá venv cũ...' -ForegroundColor Yellow
    Remove-Item -LiteralPath $Venv -Recurse -Force
}

if (-not (Test-Path $Py)) {
    Write-Host 'Tạo venv mới...' -ForegroundColor Yellow
    python -m venv $Venv
}

Write-Host 'Cập nhật pip...' -ForegroundColor Yellow
& $Py -m pip install --upgrade pip

Write-Host 'Cài thư viện trong requirements.txt...' -ForegroundColor Yellow
& $Py -m pip install -r (Join-Path $Root 'requirements.txt')

if ($Cpu) {
    Write-Host 'Cài PyTorch bản CPU...' -ForegroundColor Yellow
    & $Py -m pip install torch --index-url $CpuIndex
} else {
    Write-Host "Cài PyTorch $TorchVersion (GPU)..." -ForegroundColor Yellow
    & $Py -m pip install "torch==$TorchVersion" --index-url $CudaIndex
}

# torchvision/torchaudio không dùng trong project này -> bỏ cho nhẹ và tránh lệch version
foreach ($pkg in @('torchvision', 'torchaudio')) {
    & $Py -m pip uninstall -y $pkg 2>$null | Out-Null
}

Write-Host 'Kiểm tra môi trường...' -ForegroundColor Yellow
& $Py -c @"
import sys, torch
print('python      :', sys.version.split()[0])
print('torch       :', torch.__version__)
print('cuda build  :', torch.version.cuda)
print('cuda usable :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu         :', torch.cuda.get_device_name(0))
else:
    print('gpu         : (không có - chỉ chạy được --device cpu)')
"@

Write-Host ''
Write-Host 'Xong. Bước tiếp theo:' -ForegroundColor Green
Write-Host '  venv\Scripts\python.exe src\model\Train_mf.py --list'
Write-Host '  venv\Scripts\python.exe src\model\Train_lightgcn.py --list'

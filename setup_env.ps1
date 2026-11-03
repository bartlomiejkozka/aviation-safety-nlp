<#
.SYNOPSIS
    Environment setup script for the aviation-safety-nlp project (Advanced Data Mining).
    Automatically verifies Python 3.14, synchronizes dependencies using uv,
    registers the Jupyter kernel, and prepares the workspace.
#>

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Environment Setup: aviation-safety-nlp (Python 3.14)   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check for uv installation
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[!] 'uv' was not found in your PATH." -ForegroundColor Yellow
    Write-Host "[*] Installing 'uv' via official Astral installer..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
    
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "[-] Failed to detect uv in the current session." -ForegroundColor Red
        Write-Host "    Please restart your terminal or install uv manually: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Red
        Exit 1
    }
}

$uvVersion = uv --version
Write-Host "[+] Detected: $uvVersion" -ForegroundColor Green

# 2. Ensure Python 3.14 is installed via uv
Write-Host "`n[*] [1/3] Ensuring Python 3.14 is available..." -ForegroundColor Yellow
uv python install 3.14

# 3. Synchronize virtual environment (.venv) and project dependencies
Write-Host "`n[*] [2/3] Syncing virtual environment and installing packages..." -ForegroundColor Yellow
uv sync

# 4. Register Jupyter kernel
Write-Host "`n[*] [3/3] Registering Jupyter kernel..." -ForegroundColor Yellow
uv run python -m ipykernel install --user --name "aviation-safety-nlp" --display-name "Python 3.14 (aviation-safety-nlp)"

# 5. Optional NLTK resource setup
Write-Host "`n[*] Pre-downloading common NLTK resources (stopwords, punkt)..." -ForegroundColor Yellow
try {
    uv run python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True); print('[+] NLTK resources ready.')"
} catch {
    Write-Host "[!] Note: Could not download NLTK data automatically (offline or restricted network). You can download them later if needed." -ForegroundColor Yellow
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "   Environment setup completed successfully!              " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "`nHow to use your environment:" -ForegroundColor Cyan
Write-Host "  1. Direct execution with uv (recommended):"
Write-Host "       uv run python temp/pre-classification.py" -ForegroundColor White
Write-Host "       uv run jupyter lab" -ForegroundColor White
Write-Host "`n  2. Traditional virtual environment activation:"
Write-Host "       .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "`n  3. Adding new packages:"
Write-Host "       uv add <package-name>" -ForegroundColor White
Write-Host "==========================================================`n" -ForegroundColor Green

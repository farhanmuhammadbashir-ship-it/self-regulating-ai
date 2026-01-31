# run_demo.ps1
Write-Host "Starting Self-Regulating AI Demo..." -ForegroundColor Cyan

# Check for Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}

# Install requirements if needed (optional check, but good for first run)
if (Test-Path "requirements.txt") {
    Write-Host "Checking dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt | Out-Null
}

# Run the simulation
Write-Host "Running simulation..." -ForegroundColor Green
python experiments/simulate_failure.py

Write-Host "Demo completed." -ForegroundColor Cyan

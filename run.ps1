[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "ingest", "cli", "ask", "test", "status", "help")]
    [string]$Command = "cli",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SystemPython = "python"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"

Set-Location $ProjectRoot

function Show-Help {
    Write-Host ""
    Write-Host "Local RAG AI Assistant runner"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\run.ps1 setup"
    Write-Host "  .\run.ps1 ingest"
    Write-Host "  .\run.ps1 cli"
    Write-Host "  .\run.ps1 ask ""What time does the daily standup start?"""
    Write-Host "  .\run.ps1 test"
    Write-Host "  .\run.ps1 status"
    Write-Host ""
    Write-Host "Notes:"
    Write-Host "  setup  creates .venv if needed and installs requirements."
    Write-Host "  ingest builds data\rag.db from data\sample_docs."
    Write-Host "  cli    starts the interactive Q&A assistant."
    Write-Host "  ask    runs one question and exits."
    Write-Host "  test   runs the unit test suite without model downloads."
    Write-Host ""
}

function Assert-CommandSucceeded {
    param(
        [string]$Action
    )

    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed with exit code $LASTEXITCODE."
    }
}

function Ensure-Venv {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Host "Creating virtual environment at .venv..."
        & $SystemPython -m venv .venv
        Assert-CommandSucceeded "Virtual environment creation"
    }
}

function Assert-VenvExists {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Virtual environment not found. Run '.\run.ps1 setup' first."
    }
}

function Invoke-ProjectPython {
    param(
        [string[]]$PythonArgs
    )

    Assert-VenvExists
    & $VenvPython @PythonArgs
    Assert-CommandSucceeded "Python command"
}

switch ($Command) {
    "help" {
        Show-Help
    }
    "setup" {
        Ensure-Venv
        Write-Host "Installing requirements..."
        & $VenvPython -m pip install -r $RequirementsPath
        Assert-CommandSucceeded "Dependency installation"
        Write-Host ""
        Write-Host "Setup complete."
        Write-Host "Next: .\run.ps1 ingest"
    }
    "ingest" {
        Write-Host "Building the local SQLite knowledge base..."
        Write-Host "Foundry Local may download or load the embedding model on first run."
        Invoke-ProjectPython @("-B", "-m", "src.ingest")
    }
    "cli" {
        Write-Host "Starting the Local RAG AI Assistant..."
        Write-Host "Type 'exit' or 'quit' to close."
        Invoke-ProjectPython @("-B", "-m", "src.cli")
    }
    "ask" {
        $Question = ($RemainingArgs -join " ").Trim()
        if (-not $Question) {
            throw "Missing question. Example: .\run.ps1 ask ""What time does the daily standup start?"""
        }

        Invoke-ProjectPython @("-B", "-m", "src.rag", $Question)
    }
    "test" {
        Write-Host "Running unit tests..."
        Invoke-ProjectPython @("-B", "-m", "pytest", "-p", "no:cacheprovider")
    }
    "status" {
        Invoke-ProjectPython @(
            "-B",
            "-c",
            "from src import config; from src.storage import count_chunks; print(f'Project: {config.PROJECT_ROOT}'); print(f'Database: {config.DATABASE_PATH}'); print(f'Sample docs: {config.SAMPLE_DOCS_PATH}'); print(f'Stored chunks: {count_chunks()}')"
        )
    }
}

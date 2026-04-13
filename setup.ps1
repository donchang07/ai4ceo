#Requires -Version 5.1
<#
.SYNOPSIS
  ai4ceo environment setup for Windows — run this script in PowerShell only.

.DESCRIPTION
  Installs uv via the official install.ps1, installs Python with uv, then always runs uv sync
  so dependencies match pyproject.toml. Use -RecreateVenv to delete and recreate .venv.
  Do not run setup.sh on Windows; use this file (or setup.cmd from Explorer/cmd).

.EXAMPLE
  # From the ai4ceo folder in PowerShell (extension optional):
  .\setup

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
  .\setup.ps1
  .\setup.cmd
  .\setup.ps1 -RecreateVenv
#>

param(
    # 전체 재설치 시에만 사용: 기존 .venv 삭제 후 동기화
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"

$PYTHON_VER = "3.13.1"
$UV_INSTALL_URI = "https://astral.sh/uv/install.ps1"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Message
    Write-Host "============================================================"
}

function Write-Diag {
    param([string]$Message)
    Write-Host ("[diag] " + $Message)
}

function Test-CurrentUserExecutionPolicyReady {
    try {
        $policy = Get-ExecutionPolicy -Scope CurrentUser
        return @("RemoteSigned", "Unrestricted", "Bypass") -contains $policy
    } catch {
        return $false
    }
}

function Ensure-UserPathHasDir {
    param([Parameter(Mandatory)][string]$Dir)
    if (-not (Test-Path -LiteralPath $Dir)) {
        return
    }
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -and ($userPath -like "*${Dir}*")) {
        return
    }
    if ($userPath) {
        [Environment]::SetEnvironmentVariable("Path", "${userPath};${Dir}", "User")
    } else {
        [Environment]::SetEnvironmentVariable("Path", $Dir, "User")
    }
    if ($env:Path -notlike "*${Dir}*") {
        $env:Path = "${Dir};${env:Path}"
    }
    Write-Host ("Added to user PATH: " + $Dir)
}

if ($PSScriptRoot) {
    Set-Location -LiteralPath $PSScriptRoot
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

function Get-UvCandidateDirs {
    return @(
        (Join-Path $env:USERPROFILE ".local\bin"),
        (Join-Path $env:USERPROFILE ".cargo\bin"),
        (Join-Path $env:LOCALAPPDATA "Programs\uv\bin"),
        (Join-Path $env:USERPROFILE "scoop\shims")
    )
}

function Add-UvDirsToProcessPath {
    foreach ($dir in Get-UvCandidateDirs) {
        if ((Test-Path -LiteralPath $dir) -and ($env:Path -notlike "*${dir}*")) {
            $env:Path = "${dir};${env:Path}"
        }
    }
}

function Get-UvExePath {
    Add-UvDirsToProcessPath
    foreach ($dir in Get-UvCandidateDirs) {
        $exe = Join-Path $dir "uv.exe"
        if (Test-Path -LiteralPath $exe) {
            return $exe
        }
    }
    $cmd = Get-Command "uv.exe" -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }
    return $null
}

function Wait-ForUvExe {
    param(
        [int]$MaxRetries = 10,
        [int]$DelaySeconds = 1
    )
    for ($i = 1; $i -le $MaxRetries; $i++) {
        Refresh-PathFromRegistry
        Ensure-UserPathHasUvBin
        $found = Get-UvExePath
        if ($found) {
            return $found
        }
        Start-Sleep -Seconds $DelaySeconds
    }
    return $null
}

function Install-UvWithFallback {
    param([string]$Uri)

    # 1) Primary path: run downloaded script in current process
    try {
        $installScript = Invoke-RestMethod -Uri $Uri -UseBasicParsing
        Invoke-Expression $installScript
        return $true
    } catch {
        Write-Host ("WARN: In-process uv install failed: " + $_.Exception.Message)
    }

    # 2) Fallback path: run via a child PowerShell session
    try {
        $fallbackCommand = 'powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"'
        $fallback = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $fallbackCommand) -NoNewWindow -Wait -PassThru
        return ($fallback.ExitCode -eq 0)
    } catch {
        Write-Host ("WARN: Child-process uv install fallback failed: " + $_.Exception.Message)
    }

    return $false
}

function Ensure-UserPathHasUvBin {
    foreach ($dir in Get-UvCandidateDirs) {
        Ensure-UserPathHasDir -Dir $dir
    }
}

function Ensure-UserPathHasProjectRoot {
    if (-not $PSScriptRoot) {
        return
    }
    Ensure-UserPathHasDir -Dir $PSScriptRoot
}

function Install-RunstShim {
    if (-not $PSScriptRoot) {
        return
    }
    $shimDir = Join-Path $env:USERPROFILE ".local\bin"
    if (-not (Test-Path -LiteralPath $shimDir)) {
        New-Item -ItemType Directory -Path $shimDir -Force | Out-Null
    }
    $shimPath = Join-Path $shimDir "runst.cmd"
    $target = Join-Path $PSScriptRoot "runst.bat"
    $shim = "@echo off`r`ncall `"$target`" %*`r`n"
    Set-Content -LiteralPath $shimPath -Value $shim -Encoding Ascii -Force
    Ensure-UserPathHasDir -Dir $shimDir
    Write-Host ("Installed command shim: " + $shimPath)
}

function Refresh-PathFromRegistry {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "${machine};${user}"
    Add-UvDirsToProcessPath
}

# uv writes progress to stderr; calling via Start-Process avoids NativeCommandError under $ErrorActionPreference Stop.
function Invoke-UvProcess {
    param(
        [Parameter(Mandatory)][string]$UvExe,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    $prevEa = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $UvExe @Arguments
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) {
            return 0
        }
        return [int]$exitCode
    } finally {
        $ErrorActionPreference = $prevEa
    }
}

function Invoke-UvProcessWithRetry {
    param(
        [Parameter(Mandatory)][string]$UvExe,
        [Parameter(Mandatory)][string[]]$Arguments,
        [int]$MaxRetries = 2,
        [int]$DelaySeconds = 2,
        [Parameter(Mandatory)][string]$ActionName
    )
    for ($i = 0; $i -le $MaxRetries; $i++) {
        $exitCode = Invoke-UvProcess -UvExe $UvExe -Arguments $Arguments
        if ($exitCode -eq 0) {
            return 0
        }
        if ($i -lt $MaxRetries) {
            Write-Host ("WARN: " + $ActionName + " failed (exit " + $exitCode + "). Retrying...")
            Start-Sleep -Seconds $DelaySeconds
        } else {
            return $exitCode
        }
    }
    return 1
}

function Stop-VenvRelatedProcesses {
    param([string]$ProjectRoot)
    $killed = 0
    $venvPrefix = (Join-Path $ProjectRoot ".venv").ToLowerInvariant()
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object {
            if (-not $_.Path) { return $false }
            $procPath = $_.Path.ToLowerInvariant()
            return $procPath.StartsWith($venvPrefix)
        } |
        ForEach-Object {
            try {
                Stop-Process -Id $_.Id -Force -ErrorAction Stop
                $killed++
            } catch { }
        }
    if ($killed -gt 0) {
        Write-Host ("Closed running Python/venv-related processes: " + $killed)
        Start-Sleep -Seconds 2
    }
}

function Remove-VenvDirectoryResilient {
    param([string]$ProjectRoot)
    $venvPath = Join-Path $ProjectRoot ".venv"
    if (-not (Test-Path -LiteralPath $venvPath)) {
        return $true
    }
    Stop-VenvRelatedProcesses -ProjectRoot $ProjectRoot
    Start-Sleep -Seconds 1
    try {
        Remove-Item -LiteralPath $venvPath -Recurse -Force -ErrorAction Stop
        Write-Host "Removed .venv for clean recovery."
        return $true
    } catch {
        Write-Host "WARN: Failed to remove .venv during recovery."
        return $false
    }
}

function Invoke-UvSyncResilient {
    param(
        [Parameter(Mandatory)][string]$UvExe,
        [Parameter(Mandatory)][string]$PythonVer
    )
    $args = @("sync", "--python", $PythonVer)
    $exitSync = Invoke-UvProcessWithRetry -UvExe $UvExe -Arguments $args -MaxRetries 2 -DelaySeconds 3 -ActionName "uv sync"
    if ($exitSync -eq 0) {
        return 0
    }

    Write-Host "WARN: uv sync failed. Trying Windows file-lock recovery..."
    Stop-VenvRelatedProcesses -ProjectRoot $PWD.Path

    # One more attempt after forcibly closing likely lock holders.
    $exitRecover = Invoke-UvProcessWithRetry -UvExe $UvExe -Arguments $args -MaxRetries 1 -DelaySeconds 3 -ActionName "uv sync (after lock recovery)"
    if ($exitRecover -eq 0) {
        return 0
    }

    Write-Host "WARN: uv sync still failing. Trying full automatic .venv rebuild..."
    $removed = Remove-VenvDirectoryResilient -ProjectRoot $PWD.Path
    if (-not $removed) {
        return $exitRecover
    }
    $exitRebuild = Invoke-UvProcessWithRetry -UvExe $UvExe -Arguments $args -MaxRetries 1 -DelaySeconds 3 -ActionName "uv sync (after venv rebuild)"
    return $exitRebuild
}

function Show-EnvironmentDiagnostics {
    Write-Step '[Diagnostics] Environment snapshot'
    try {
        Write-Diag ("PowerShell: " + $PSVersionTable.PSVersion.ToString())
    } catch { }
    Write-Diag ("UserProfile: " + $env:USERPROFILE)
    try {
        $policyCurrent = Get-ExecutionPolicy -Scope CurrentUser
        Write-Diag ("ExecutionPolicy(CurrentUser): " + $policyCurrent)
    } catch { }
    foreach ($dir in Get-UvCandidateDirs) {
        Write-Diag ("CandidateDir exists? " + $dir + " => " + (Test-Path -LiteralPath $dir))
    }
    $cmd = Get-Command "uv.exe" -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        Write-Diag ("Get-Command uv.exe: " + $cmd.Source)
    } else {
        Write-Diag "Get-Command uv.exe: not found"
    }
}

# In double-quoted strings, [ ] is wildcard syntax; use single quotes for step labels.
Write-Step '[0/6] Prerequisites (no system Python / pip / uv required)'
if (-not (Get-Command "Invoke-RestMethod" -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Invoke-RestMethod not available. PowerShell 5.1 or later is required."
    exit 1
}
Write-Host "OK: Invoke-RestMethod available for HTTPS download."
if (-not (Test-CurrentUserExecutionPolicyReady)) {
    Write-Host "NOTICE: CurrentUser execution policy is restrictive."
    Write-Host "If installation is blocked, run once and retry:"
    Write-Host "  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force"
}
Add-UvDirsToProcessPath

Write-Step '[1/6] uv package manager'
$uvExe = Get-UvExePath
if (-not $uvExe) {
    Write-Host "uv not found. Installing via official install.ps1 (Python not required)..."
    $installOk = Install-UvWithFallback -Uri $UV_INSTALL_URI
    if (-not $installOk) {
        Write-Host "ERROR: uv install failed via both primary and fallback method."
        Write-Host "Check network / firewall / proxy, or run:"
        Write-Host '  powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"'
        exit 1
    }
    $uvExe = Wait-ForUvExe -MaxRetries 12 -DelaySeconds 1
}
if (-not $uvExe) {
    Write-Host "ERROR: uv.exe not found after install/retry."
    Write-Host "Try these in order:"
    Write-Host "  1) Close this terminal and open a new PowerShell window."
    Write-Host "  2) Run: uv --version"
    Write-Host "  3) Run this script again from the project root."
    exit 1
}
Write-Host ("uv: " + $uvExe)
try {
    & $uvExe --version
} catch {
    Write-Host "ERROR: uv executable was found, but it failed to run."
    Write-Host "Close this terminal, open a new PowerShell window, and run setup again."
    exit 1
}

Write-Step '[2/6] Ensure user PATH includes uv directory'
Ensure-UserPathHasUvBin
Ensure-UserPathHasProjectRoot
Install-RunstShim

Write-Step ('[3/6] Install Python ' + $PYTHON_VER + ' via uv (no system Python needed)')
$exitInstall = Invoke-UvProcessWithRetry -UvExe $uvExe -Arguments @("python", "install", $PYTHON_VER) -MaxRetries 2 -DelaySeconds 3 -ActionName "uv python install"
if ($exitInstall -ne 0) {
    Write-Host ("ERROR: Failed to install Python " + $PYTHON_VER)
    exit $exitInstall
}

Write-Step '[4/6] Check pyproject.toml and optional .venv rebuild'
if (-not (Test-Path -LiteralPath "pyproject.toml")) {
    Write-Host "ERROR: pyproject.toml not found. Run this script from the project root."
    exit 1
}

if ($RecreateVenv -and (Test-Path -LiteralPath ".venv")) {
    Write-Host "-RecreateVenv: Removing existing .venv..."
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and ($_.Path -like "*\.venv\*") } |
        ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    try {
        Remove-Item -LiteralPath ".venv" -Recurse -Force -ErrorAction Stop
        Write-Host "Removed .venv."
    } catch {
        Write-Host "ERROR: Could not delete .venv. Delete the folder manually, then run this script again."
        exit 1
    }
} elseif (Test-Path -LiteralPath ".venv") {
    Write-Host "Keeping existing .venv. Full rebuild: .\setup.ps1 -RecreateVenv"
} else {
    Write-Host "No .venv yet — uv sync will create it."
}

Write-Step '[5/6] uv sync (may take 1-3 minutes; runs every time to match pyproject.toml)'
$exitSync = Invoke-UvSyncResilient -UvExe $uvExe -PythonVer $PYTHON_VER
if ($exitSync -ne 0) {
    Write-Host "ERROR: uv sync failed."
    Show-EnvironmentDiagnostics
    Write-Host "Likely cause on Windows: locked files in .venv or antivirus/file indexing race."
    Write-Host "Close editors/kernels that use this project and rerun setup.bat."
    Write-Host "Try once manually for detailed output:"
    Write-Host ("  `"" + $uvExe + "`" sync --python " + $PYTHON_VER)
    exit $exitSync
}

Write-Step '[6/6] Verification'
$prevEa = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $uvExe pip list 2>$null | Select-String -Pattern "langchain|openai|streamlit|python" -CaseSensitive:$false
$ErrorActionPreference = $prevEa

Write-Step '[7/7] Streamlit sanity check (project environment)'
$exitStreamlitCheck = Invoke-UvProcessWithRetry -UvExe $uvExe -Arguments @("run", "python", "-c", "import streamlit; print(streamlit.__version__)") -MaxRetries 1 -DelaySeconds 2 -ActionName "streamlit import check"
if ($exitStreamlitCheck -ne 0) {
    Write-Host "ERROR: Streamlit is not importable in the uv-managed project environment."
    Write-Host "Run once manually:"
    Write-Host ("  `"" + $uvExe + "`" sync --python " + $PYTHON_VER)
    exit $exitStreamlitCheck
}

Write-Host ""
Write-Host "============================================================"
Write-Host ("Setup complete. Python " + $PYTHON_VER)
Write-Host "============================================================"
Write-Host ("uv executable: " + $uvExe)
Write-Host ""
Write-Host "Activating venv..."
try {
    & ".\.venv\Scripts\Activate.ps1"
    Write-Host "Activated: .venv"
} catch {
    Write-Host "WARNING: Failed to auto-activate venv. Activate manually:"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
}
Write-Host ""
Write-Host "IMPORTANT: To avoid 'streamlit not found' issues, run Streamlit via uv:"
Write-Host "  uv run streamlit run <your_app.py>"
Write-Host "OR (recommended on Windows PATH issues):"
Write-Host "  .\runst.bat <your_app.py>"
Write-Host ""
Write-Host "If script execution is blocked, run once (CurrentUser):"
Write-Host "  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
Write-Host ""

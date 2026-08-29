[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $repoRoot "configs\public_demo.yaml"
$logDirectory = Join-Path $repoRoot "outputs\public-demo"
$backendProcess = $null
$tunnelProcess = $null

function Resolve-RepositoryFile {
    param(
        [Parameter(Mandatory)]
        [string]$Value,
        [Parameter(Mandatory)]
        [string]$Name
    )

    $candidate = if ([IO.Path]::IsPathRooted($Value)) {
        $Value
    } else {
        Join-Path $repoRoot $Value
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "$Name must name an existing local file."
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Quote-ProcessArgument {
    param(
        [Parameter(Mandatory)]
        [string]$Value
    )

    if ($Value.Contains('"')) {
        throw "Process arguments must not contain quote characters."
    }
    return '"' + $Value + '"'
}

function Wait-ForHttpSuccess {
    param(
        [Parameter(Mandatory)]
        [string]$Uri,
        [Parameter(Mandatory)]
        [Diagnostics.Process]$Process,
        [Parameter(Mandatory)]
        [int]$TimeoutSeconds
    )

    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($Process.HasExited) {
            throw "Child process exited before $Uri became ready."
        }
        try {
            $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                return
            }
        } catch {
            Start-Sleep -Milliseconds 250
        }
    }
    throw "Timed out waiting for $Uri."
}

function Stop-ChildProcess {
    param(
        [Diagnostics.Process]$Process
    )

    if ($null -ne $Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id
        $Process.WaitForExit(5000) | Out-Null
    }
}

try {
    Set-Location -LiteralPath $repoRoot
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw 'Missing .venv\Scripts\python.exe. Install with: py -3.14 -m venv .venv'
    }
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Missing configs/public_demo.yaml."
    }
    if ($env:DEMO_ENABLED -ne "1") {
        throw "Set DEMO_ENABLED=1 explicitly before starting the public demo."
    }
    if ([string]::IsNullOrWhiteSpace($env:PUBLIC_ORIGIN)) {
        throw "Set PUBLIC_ORIGIN to the exact GitHub Pages HTTPS origin."
    }
    $origin = [Uri]$env:PUBLIC_ORIGIN
    if (
        $origin.Scheme -ne "https" -or
        -not [string]::IsNullOrEmpty($origin.UserInfo) -or
        $origin.AbsolutePath -ne "/" -or
        -not [string]::IsNullOrEmpty($origin.Query) -or
        -not [string]::IsNullOrEmpty($origin.Fragment)
    ) {
        throw "PUBLIC_ORIGIN must be one credential-free HTTPS origin."
    }
    if ([string]::IsNullOrWhiteSpace($env:MINIGPT_CHECKPOINT)) {
        throw "Set MINIGPT_CHECKPOINT before starting the public demo."
    }
    if ([string]::IsNullOrWhiteSpace($env:MINIGPT_TOKENIZER)) {
        throw "Set MINIGPT_TOKENIZER before starting the public demo."
    }
    $checkpointPath = Resolve-RepositoryFile -Value $env:MINIGPT_CHECKPOINT -Name "MINIGPT_CHECKPOINT"
    $tokenizerPath = Resolve-RepositoryFile -Value $env:MINIGPT_TOKENIZER -Name "MINIGPT_TOKENIZER"
    $ngrokCommand = Get-Command ngrok.exe -ErrorAction SilentlyContinue
    if ($null -eq $ngrokCommand) {
        throw "ngrok.exe is not installed or is not available on PATH."
    }

    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $backendOutput = Join-Path $logDirectory "backend.stdout.log"
    $backendError = Join-Path $logDirectory "backend.stderr.log"
    $tunnelOutput = Join-Path $logDirectory "ngrok.stdout.log"
    $tunnelError = Join-Path $logDirectory "ngrok.stderr.log"
    $backendArguments = @(
        "-m",
        "minigpt",
        "demo-serve",
        "--config",
        (Quote-ProcessArgument $configPath),
        "--checkpoint",
        (Quote-ProcessArgument $checkpointPath),
        "--tokenizer",
        (Quote-ProcessArgument $tokenizerPath),
        "--host",
        "127.0.0.1",
        "--port",
        "8000"
    )
    $backendProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList $backendArguments `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $backendOutput `
        -RedirectStandardError $backendError `
        -WindowStyle Hidden `
        -PassThru
    Wait-ForHttpSuccess `
        -Uri "http://127.0.0.1:8000/healthz" `
        -Process $backendProcess `
        -TimeoutSeconds 45

    $tunnelArguments = @("http", "http://127.0.0.1:8000")
    $tunnelProcess = Start-Process `
        -FilePath $ngrokCommand.Source `
        -ArgumentList $tunnelArguments `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $tunnelOutput `
        -RedirectStandardError $tunnelError `
        -WindowStyle Hidden `
        -PassThru
    Wait-ForHttpSuccess `
        -Uri "http://127.0.0.1:4040/api/tunnels" `
        -Process $tunnelProcess `
        -TimeoutSeconds 20
    $tunnels = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 3
    $publicUrl = $tunnels.tunnels |
        Where-Object { $_.proto -eq "https" -and $_.public_url -is [string] } |
        Select-Object -First 1 -ExpandProperty public_url
    if ([string]::IsNullOrWhiteSpace($publicUrl)) {
        throw "ngrok did not publish an HTTPS tunnel."
    }

    Write-Host "miniGPT public demo is online: $publicUrl"
    Write-Host "GitHub Pages CORS origin: $($origin.GetLeftPart([UriPartial]::Authority))"
    Write-Host "Logs: $logDirectory"
    Write-Host "Press Ctrl+C to stop the backend and tunnel."
    while ($true) {
        if ($backendProcess.HasExited) {
            throw "miniGPT demo backend exited unexpectedly."
        }
        if ($tunnelProcess.HasExited) {
            throw "ngrok tunnel exited unexpectedly."
        }
        Start-Sleep -Seconds 1
    }
} finally {
    Stop-ChildProcess -Process $tunnelProcess
    Stop-ChildProcess -Process $backendProcess
    Set-Location -LiteralPath $repoRoot
}

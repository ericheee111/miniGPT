# Start or validate the loopback-only miniGPT Story Forge public demo.
[CmdletBinding()]
param(
    [switch]$ValidateOnly,
    [switch]$SkipFunnel,
    [int]$Port = 8001,
    [string]$CheckpointSha256 = "",
    [string]$TokenizerSha256 = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "start_public_demo_tailscale.ps1")

if ($Port -eq 8000) {
    throw "Port 8000 is reserved for the legacy rollback backend; Story Forge uses 8001."
}
if ($Port -lt 1 -or $Port -gt 65535) {
    throw "Port must be in [1, 65535]."
}

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$PythonPath = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$ConfigPath = Join-Path $RepositoryRoot "configs\story_forge_public_demo.yaml"
$RuntimeDirectory = Join-Path $RepositoryRoot "outputs\story-forge-demo"
$RuntimeStatePath = Join-Path $RuntimeDirectory "runtime-state.json"
$BackendTarget = "http://127.0.0.1:$Port"
$BackendHealthUrl = "$BackendTarget/healthz"
$BackendInfoUrl = "$BackendTarget/demo/info"

# Rebind the shared Tailscale helper functions to this managed backend.
$script:RepositoryRoot = $RepositoryRoot
$script:RuntimeDirectory = $RuntimeDirectory
$script:RuntimeStatePath = $RuntimeStatePath
$script:BackendTarget = $BackendTarget
$script:BackendHealthUrl = $BackendHealthUrl
$script:BackendInfoUrl = $BackendInfoUrl

function Resolve-StoryForgeFile {
    param(
        [Parameter(Mandatory)][string]$Value,
        [Parameter(Mandatory)][string]$Name
    )

    $candidate = if ([IO.Path]::IsPathRooted($Value)) {
        $Value
    } else {
        Join-Path $RepositoryRoot $Value
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "$Name must name an existing local file."
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Get-FunnelForTarget {
    param(
        [Parameter(Mandatory)][string]$StatusJson,
        [Parameter(Mandatory)][string]$Target
    )

    $savedTarget = $script:BackendTarget
    try {
        $script:BackendTarget = $Target
        return Get-MiniGPTFunnel -StatusJson $StatusJson
    } finally {
        $script:BackendTarget = $savedTarget
    }
}

function Assert-StoryForgeInfo {
    param(
        [Parameter(Mandatory)][string]$Uri
    )

    $info = Invoke-RestMethod -Uri $Uri -TimeoutSec 8
    if ($info.demo_mode -ne "public") {
        throw "Story Forge public mode is not enabled."
    }
    if ($info.project_version -ne "1.1.0" -or $info.model_id -ne "minigpt-story-forge") {
        throw "The backend did not load the reviewed Story Forge 1.1 model."
    }
    if (
        $info.features.story_forge -ne $true -or
        $info.features.prediction_lab -ne $true -or
        $info.features.systems_lab -ne $true
    ) {
        throw "Story Forge feature flags are unavailable."
    }
}

function Wait-ForStoryForgeBackend {
    param(
        [Parameter(Mandatory)][Diagnostics.Process]$Process,
        [int]$TimeoutSeconds = 60
    )

    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($Process.HasExited) {
            throw "Story Forge backend exited before its loopback health check succeeded."
        }
        if (Test-HttpSuccess -Uri $BackendHealthUrl) {
            Assert-StoryForgeInfo -Uri $BackendInfoUrl
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Timed out waiting for the Story Forge loopback health check."
}

function Wait-ForPublicStoryForge {
    param(
        [Parameter(Mandatory)][string]$PublicUrl,
        [int]$TimeoutSeconds = 60
    )

    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        try {
            $health = Invoke-WebRequest -Uri "$PublicUrl/healthz" -UseBasicParsing -TimeoutSec 5
            if ($health.StatusCode -eq 200) {
                Assert-StoryForgeInfo -Uri "$PublicUrl/demo/info"
                return
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    throw "Timed out validating Story Forge through the public Funnel."
}

function Write-StoryForgeRuntimeState {
    param(
        [Parameter(Mandatory)][Diagnostics.Process]$BackendProcess,
        [AllowNull()][string]$PublicUrl,
        [AllowNull()][int]$PreviousFunnelPort
    )

    $document = [ordered]@{
        schema_version = 1
        backend_pid = $BackendProcess.Id
        backend_start_ticks = $BackendProcess.StartTime.ToUniversalTime().Ticks
        backend_executable = $PythonPath
        local_port = $Port
        local_target = $BackendTarget
        public_url = $PublicUrl
        previous_funnel_port = $PreviousFunnelPort
    }
    $temporary = "$RuntimeStatePath.tmp"
    $encoding = [Text.UTF8Encoding]::new($false)
    [IO.File]::WriteAllText($temporary, ($document | ConvertTo-Json), $encoding)
    Move-Item -LiteralPath $temporary -Destination $RuntimeStatePath -Force
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Missing .venv\Scripts\python.exe."
}
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "Missing configs/story_forge_public_demo.yaml."
}
if ([string]::IsNullOrWhiteSpace($env:MINIGPT_CHECKPOINT)) {
    throw "Set MINIGPT_CHECKPOINT to the local Story Forge checkpoint."
}
if ([string]::IsNullOrWhiteSpace($env:MINIGPT_TOKENIZER)) {
    throw "Set MINIGPT_TOKENIZER to the local Story Forge tokenizer."
}

$checkpointPath = Resolve-StoryForgeFile `
    -Value $env:MINIGPT_CHECKPOINT `
    -Name "MINIGPT_CHECKPOINT"
$tokenizerPath = Resolve-StoryForgeFile `
    -Value $env:MINIGPT_TOKENIZER `
    -Name "MINIGPT_TOKENIZER"

$validatorArguments = @(
    (Quote-ProcessArgument (Join-Path $RepositoryRoot "scripts\validate_story_forge_model.py")),
    "--checkpoint", (Quote-ProcessArgument $checkpointPath),
    "--tokenizer", (Quote-ProcessArgument $tokenizerPath)
)
if (-not [string]::IsNullOrWhiteSpace($CheckpointSha256)) {
    $validatorArguments += @("--checkpoint-sha256", $CheckpointSha256)
}
if (-not [string]::IsNullOrWhiteSpace($TokenizerSha256)) {
    $validatorArguments += @("--tokenizer-sha256", $TokenizerSha256)
}
$validatorProcess = Start-Process `
    -FilePath $PythonPath `
    -ArgumentList $validatorArguments `
    -WorkingDirectory $RepositoryRoot `
    -NoNewWindow `
    -Wait `
    -PassThru
if ($validatorProcess.ExitCode -ne 0) {
    throw "Story Forge model validation failed; live services were not changed."
}
if ($ValidateOnly) {
    Write-Host "Story Forge model validated; no process or Funnel was changed."
    exit 0
}

if ($env:DEMO_ENABLED -ne "1") {
    throw "Set DEMO_ENABLED=1 explicitly before starting the Story Forge demo."
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

$tailscalePath = $null
$previousFunnelPort = $null
if (-not $SkipFunnel) {
    $tailscalePath = Resolve-TailscaleCommand
    $tailscaleStatus = Invoke-Tailscale -CommandPath $tailscalePath -Arguments @("status", "--json")
    Assert-TailscaleLoggedIn -StatusJson $tailscaleStatus
    $initialFunnelStatus = Invoke-Tailscale `
        -CommandPath $tailscalePath `
        -Arguments @("funnel", "status", "--json")
    $legacyFunnel = Get-FunnelForTarget `
        -StatusJson $initialFunnelStatus `
        -Target "http://127.0.0.1:8000"
    $storyFunnel = Get-FunnelForTarget -StatusJson $initialFunnelStatus -Target $BackendTarget
    if ($null -eq $legacyFunnel -and $null -eq $storyFunnel -and (
        Test-FunnelPortInUse -StatusJson $initialFunnelStatus -Port "443"
    )) {
        throw "Tailscale HTTPS port 443 already serves an unrelated target."
    }
    if ($null -ne $legacyFunnel) {
        $previousFunnelPort = 8000
    }
}

New-Item -ItemType Directory -Path $RuntimeDirectory -Force | Out-Null
$state = Read-RuntimeState
$backendProcess = Get-ManagedBackendProcess -State $state -PythonPath $PythonPath
$startedBackend = $false
$switchedFunnel = $false
$publicUrl = $null
try {
    if ($null -ne $backendProcess -and (Test-HttpSuccess -Uri $BackendHealthUrl)) {
        Assert-StoryForgeInfo -Uri $BackendInfoUrl
        $storedPrevious = Get-JsonPropertyValue -Object $state -Name "previous_funnel_port"
        if ($storedPrevious -eq 8000) {
            $previousFunnelPort = 8000
        }
    } else {
        if ($null -ne $backendProcess) {
            Stop-ExactProcess -Process $backendProcess
        } elseif (Test-TcpPortInUse -Port $Port) {
            throw "Port $Port is already used by an unmanaged local service."
        }
        $stdout = Join-Path $RuntimeDirectory "backend.stdout.log"
        $stderr = Join-Path $RuntimeDirectory "backend.stderr.log"
        $backendArguments = @(
            "-m", "minigpt", "demo-serve",
            "--config", (Quote-ProcessArgument $ConfigPath),
            "--checkpoint", (Quote-ProcessArgument $checkpointPath),
            "--tokenizer", (Quote-ProcessArgument $tokenizerPath),
            "--host", "127.0.0.1",
            "--port", "$Port"
        )
        $backendProcess = Start-Process `
            -FilePath $PythonPath `
            -ArgumentList $backendArguments `
            -WorkingDirectory $RepositoryRoot `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -WindowStyle Hidden `
            -PassThru
        $startedBackend = $true
        Wait-ForStoryForgeBackend -Process $backendProcess
    }

    if (-not $SkipFunnel) {
        $currentStatus = Invoke-Tailscale `
            -CommandPath $tailscalePath `
            -Arguments @("funnel", "status", "--json")
        $currentStoryFunnel = Get-FunnelForTarget -StatusJson $currentStatus -Target $BackendTarget
        if ($null -eq $currentStoryFunnel) {
            $null = Invoke-Tailscale `
                -CommandPath $tailscalePath `
                -Arguments @("funnel", "--bg", "--yes", "$Port")
            $switchedFunnel = $true
            $currentStatus = Invoke-Tailscale `
                -CommandPath $tailscalePath `
                -Arguments @("funnel", "status", "--json")
            $currentStoryFunnel = Get-FunnelForTarget `
                -StatusJson $currentStatus `
                -Target $BackendTarget
        }
        if ($null -eq $currentStoryFunnel) {
            throw "Tailscale Funnel did not publish the Story Forge loopback target."
        }
        $publicUrl = $currentStoryFunnel.PublicUrl
        Wait-ForPublicStoryForge -PublicUrl $publicUrl
    }

    Write-StoryForgeRuntimeState `
        -BackendProcess $backendProcess `
        -PublicUrl $publicUrl `
        -PreviousFunnelPort $previousFunnelPort
    Write-Host "miniGPT Story Forge is running on $BackendTarget"
    if ($null -ne $publicUrl) {
        Write-Host "Public URL: $publicUrl"
    } else {
        Write-Host "Funnel unchanged because -SkipFunnel was selected."
    }
    Write-Host "Runtime state: $RuntimeStatePath"
} catch {
    if ($switchedFunnel -and $null -ne $tailscalePath) {
        try {
            if ($previousFunnelPort -eq 8000) {
                $null = Invoke-Tailscale `
                    -CommandPath $tailscalePath `
                    -Arguments @("funnel", "--bg", "--yes", "8000")
            } else {
                $script:BackendTarget = $BackendTarget
                Remove-MiniGPTFunnel -CommandPath $tailscalePath
            }
        } catch {
            Write-Warning "Funnel rollback failed; inspect 'tailscale funnel status --json'."
        }
    }
    if ($startedBackend) {
        Stop-ExactProcess -Process $backendProcess
    }
    throw
}

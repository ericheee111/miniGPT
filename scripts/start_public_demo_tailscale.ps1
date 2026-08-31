[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$script:RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$script:RuntimeDirectory = Join-Path $script:RepositoryRoot "outputs\public-demo"
$script:RuntimeStatePath = Join-Path $script:RuntimeDirectory "runtime-state.json"
$script:BackendHealthUrl = "http://127.0.0.1:8000/healthz"
$script:BackendInfoUrl = "http://127.0.0.1:8000/demo/info"
$script:BackendTarget = "http://127.0.0.1:8000"

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
        Join-Path $script:RepositoryRoot $Value
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "$Name must name an existing local file."
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Resolve-TailscaleCommand {
    param(
        [string]$Name = "tailscale.exe"
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Tailscale CLI is not installed or is not available on PATH."
    }
    return $command.Source
}

function Invoke-Tailscale {
    param(
        [Parameter(Mandatory)]
        [string]$CommandPath,
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $output = & $CommandPath @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        $detail = ($output | Out-String).Trim()
        throw "Tailscale command failed: $detail"
    }
    return ($output | Out-String).Trim()
}

function ConvertFrom-StrictJson {
    param(
        [Parameter(Mandatory)]
        [string]$Json,
        [Parameter(Mandatory)]
        [string]$Context
    )

    try {
        return $Json | ConvertFrom-Json -ErrorAction Stop
    } catch {
        throw "$Context returned invalid JSON."
    }
}

function Get-JsonPropertyValue {
    param(
        [AllowNull()]
        [object]$Object,
        [Parameter(Mandatory)]
        [string]$Name
    )

    if ($null -eq $Object) {
        return $null
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Assert-TailscaleLoggedIn {
    param(
        [Parameter(Mandatory)]
        [string]$StatusJson
    )

    $status = ConvertFrom-StrictJson -Json $StatusJson -Context "tailscale status --json"
    $backendState = Get-JsonPropertyValue -Object $status -Name "BackendState"
    $self = Get-JsonPropertyValue -Object $status -Name "Self"
    $online = Get-JsonPropertyValue -Object $self -Name "Online"
    if ($backendState -ne "Running" -or $online -ne $true) {
        throw "Tailscale is not logged in and online. Run 'tailscale up' and complete login first."
    }
}

function Get-MiniGPTFunnel {
    param(
        [Parameter(Mandatory)]
        [string]$StatusJson
    )

    $status = ConvertFrom-StrictJson -Json $StatusJson -Context "tailscale funnel status --json"
    $allowFunnel = Get-JsonPropertyValue -Object $status -Name "AllowFunnel"
    $web = Get-JsonPropertyValue -Object $status -Name "Web"
    if ($null -eq $allowFunnel -or $null -eq $web) {
        return $null
    }

    $funnelMatches = @()
    foreach ($allowProperty in @($allowFunnel.PSObject.Properties)) {
        if ($allowProperty.Value -ne $true) {
            continue
        }
        $hostPort = $allowProperty.Name
        $webEntry = Get-JsonPropertyValue -Object $web -Name $hostPort
        $handlers = Get-JsonPropertyValue -Object $webEntry -Name "Handlers"
        $rootHandler = Get-JsonPropertyValue -Object $handlers -Name "/"
        $proxy = Get-JsonPropertyValue -Object $rootHandler -Name "Proxy"
        if ($proxy -ne $script:BackendTarget) {
            continue
        }
        $handlerProperties = @($handlers.PSObject.Properties)
        if ($handlerProperties.Count -ne 1 -or $handlerProperties[0].Name -ne "/") {
            throw "The miniGPT Funnel shares HTTPS port 443 with other handlers."
        }
        $separator = $hostPort.LastIndexOf(":")
        if ($separator -le 0) {
            throw "Tailscale Funnel status contains an invalid host and port."
        }
        $hostname = $hostPort.Substring(0, $separator)
        $port = $hostPort.Substring($separator + 1)
        if ($port -notmatch "^\d+$") {
            throw "Tailscale Funnel status contains an invalid HTTPS port."
        }
        $url = if ($port -eq "443") {
            "https://$hostname"
        } else {
            "https://${hostname}:$port"
        }
        $funnelMatches += [PSCustomObject]@{
            HostPort = $hostPort
            PublicUrl = $url
            Proxy = $proxy
        }
    }
    if ($funnelMatches.Count -gt 1) {
        throw "Tailscale Funnel status contains multiple miniGPT endpoints."
    }
    if ($funnelMatches.Count -eq 0) {
        return $null
    }
    return $funnelMatches[0]
}

function Test-FunnelPortInUse {
    param(
        [Parameter(Mandatory)]
        [string]$StatusJson,
        [string]$Port = "443"
    )

    $status = ConvertFrom-StrictJson -Json $StatusJson -Context "tailscale funnel status --json"
    foreach ($sectionName in @("Web", "AllowFunnel")) {
        $section = Get-JsonPropertyValue -Object $status -Name $sectionName
        if ($null -eq $section) {
            continue
        }
        foreach ($property in @($section.PSObject.Properties)) {
            if ($property.Name.EndsWith(":$Port", [StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
        }
    }
    return $false
}

function Get-MiniGPTFunnelAction {
    param(
        [Parameter(Mandatory)]
        [string]$StatusJson
    )

    if ($null -ne (Get-MiniGPTFunnel -StatusJson $StatusJson)) {
        return "reuse"
    }
    if (Test-FunnelPortInUse -StatusJson $StatusJson -Port "443") {
        throw "Tailscale HTTPS port 443 already serves another target."
    }
    return "create"
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

function Test-HttpSuccess {
    param(
        [Parameter(Mandatory)]
        [string]$Uri
    )

    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-TcpPortInUse {
    param(
        [Parameter(Mandatory)]
        [int]$Port
    )

    $listeners = @(Get-NetTCPConnection `
        -State Listen `
        -LocalPort $Port `
        -ErrorAction SilentlyContinue)
    return $listeners.Count -gt 0
}

function Wait-ForBackend {
    param(
        [Parameter(Mandatory)]
        [Diagnostics.Process]$Process,
        [int]$TimeoutSeconds = 45
    )

    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($Process.HasExited) {
            throw "miniGPT backend exited before its loopback health check succeeded."
        }
        if (Test-HttpSuccess -Uri $script:BackendHealthUrl) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Timed out waiting for the miniGPT loopback health check."
}

function Read-RuntimeState {
    if (-not (Test-Path -LiteralPath $script:RuntimeStatePath -PathType Leaf)) {
        return $null
    }
    $json = [IO.File]::ReadAllText($script:RuntimeStatePath)
    return ConvertFrom-StrictJson -Json $json -Context "public demo runtime state"
}

function Get-ManagedBackendProcess {
    param(
        [AllowNull()]
        [object]$State,
        [Parameter(Mandatory)]
        [string]$PythonPath
    )

    if ($null -eq $State) {
        return $null
    }
    $processId = Get-JsonPropertyValue -Object $State -Name "backend_pid"
    $startTicks = Get-JsonPropertyValue -Object $State -Name "backend_start_ticks"
    if ($processId -isnot [long] -and $processId -isnot [int]) {
        return $null
    }
    try {
        $process = Get-Process -Id $processId -ErrorAction Stop
        $resolvedProcessPath = (Resolve-Path -LiteralPath $process.Path).Path
        $resolvedPythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
        if (
            $resolvedProcessPath -ne $resolvedPythonPath -or
            $process.StartTime.ToUniversalTime().Ticks -ne $startTicks
        ) {
            return $null
        }
        return $process
    } catch {
        return $null
    }
}

function Assert-PublicBackendInfo {
    $info = Invoke-RestMethod -Uri $script:BackendInfoUrl -TimeoutSec 3
    if ($info.demo_mode -ne "public") {
        throw "Port 8000 did not return the enabled miniGPT public-demo schema."
    }
}

function Write-RuntimeState {
    param(
        [Parameter(Mandatory)]
        [Diagnostics.Process]$BackendProcess,
        [Parameter(Mandatory)]
        [string]$PythonPath,
        [Parameter(Mandatory)]
        [string]$PublicUrl
    )

    $document = [ordered]@{
        schema_version = 1
        backend_pid = $BackendProcess.Id
        backend_start_ticks = $BackendProcess.StartTime.ToUniversalTime().Ticks
        backend_executable = $PythonPath
        local_target = $script:BackendTarget
        public_url = $PublicUrl
    }
    $temporary = "$($script:RuntimeStatePath).tmp"
    $encoding = [Text.UTF8Encoding]::new($false)
    [IO.File]::WriteAllText($temporary, ($document | ConvertTo-Json), $encoding)
    Move-Item -LiteralPath $temporary -Destination $script:RuntimeStatePath -Force
}

function Stop-ExactProcess {
    param(
        [AllowNull()]
        [Diagnostics.Process]$Process
    )

    if ($null -ne $Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id
        $Process.WaitForExit(5000) | Out-Null
    }
}

function Invoke-MiniGPTPublicDemoStart {
    Set-Location -LiteralPath $script:RepositoryRoot
    $tailscalePath = Resolve-TailscaleCommand
    $tailscaleStatus = Invoke-Tailscale -CommandPath $tailscalePath -Arguments @("status", "--json")
    Assert-TailscaleLoggedIn -StatusJson $tailscaleStatus

    $pythonPath = Join-Path $script:RepositoryRoot ".venv\Scripts\python.exe"
    $configPath = Join-Path $script:RepositoryRoot "configs\public_demo.yaml"
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

    New-Item -ItemType Directory -Path $script:RuntimeDirectory -Force | Out-Null
    $state = Read-RuntimeState
    $backendProcess = Get-ManagedBackendProcess -State $state -PythonPath $pythonPath
    $startedBackend = $false
    $configuredFunnel = $false
    try {
        if ($null -ne $backendProcess -and (Test-HttpSuccess -Uri $script:BackendHealthUrl)) {
            Assert-PublicBackendInfo
        } else {
            if ($null -ne $backendProcess) {
                Stop-ExactProcess -Process $backendProcess
            } elseif (Test-TcpPortInUse -Port 8000) {
                throw "Port 8000 is already used by an unmanaged local service."
            }
            $backendOutput = Join-Path $script:RuntimeDirectory "backend.stdout.log"
            $backendError = Join-Path $script:RuntimeDirectory "backend.stderr.log"
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
                -WorkingDirectory $script:RepositoryRoot `
                -RedirectStandardOutput $backendOutput `
                -RedirectStandardError $backendError `
                -WindowStyle Hidden `
                -PassThru
            $startedBackend = $true
            Wait-ForBackend -Process $backendProcess
            Assert-PublicBackendInfo
        }

        $funnelStatus = Invoke-Tailscale `
            -CommandPath $tailscalePath `
            -Arguments @("funnel", "status", "--json")
        $funnelAction = Get-MiniGPTFunnelAction -StatusJson $funnelStatus
        $funnel = Get-MiniGPTFunnel -StatusJson $funnelStatus
        if ($funnelAction -eq "create") {
            $null = Invoke-Tailscale `
                -CommandPath $tailscalePath `
                -Arguments @("funnel", "--bg", "--yes", "8000")
            $configuredFunnel = $true
            $funnelStatus = Invoke-Tailscale `
                -CommandPath $tailscalePath `
                -Arguments @("funnel", "status", "--json")
            $funnel = Get-MiniGPTFunnel -StatusJson $funnelStatus
        }
        if ($null -eq $funnel) {
            throw "Tailscale Funnel did not publish the miniGPT loopback target."
        }

        Write-RuntimeState `
            -BackendProcess $backendProcess `
            -PythonPath $pythonPath `
            -PublicUrl $funnel.PublicUrl
        Write-Host "miniGPT public demo is online."
        Write-Host "DEMO_API_BASE=$($funnel.PublicUrl)"
        Write-Host "GitHub Pages CORS origin: $($origin.GetLeftPart([UriPartial]::Authority))"
        Write-Host "Logs and runtime state: $script:RuntimeDirectory"
    } catch {
        if ($configuredFunnel) {
            try {
                $null = Invoke-Tailscale `
                    -CommandPath $tailscalePath `
                    -Arguments @("funnel", "--https=443", "off")
            } catch {
                Write-Warning "Funnel cleanup failed; run scripts\stop_public_demo_tailscale.ps1."
            }
        }
        if ($startedBackend) {
            Stop-ExactProcess -Process $backendProcess
        }
        throw
    }
}

if ($MyInvocation.InvocationName -ne ".") {
    Invoke-MiniGPTPublicDemoStart
}

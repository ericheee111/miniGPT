# Stop only the Story Forge backend recorded in its runtime state.
# -DisableFunnel additionally restores the recorded port-8000 target, or removes
# the Story Forge target when no previous miniGPT Funnel was recorded.
[CmdletBinding()]
param(
    [switch]$DisableFunnel
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "start_public_demo_tailscale.ps1")

$script:RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$script:RuntimeDirectory = Join-Path $script:RepositoryRoot "outputs\story-forge-demo"
$script:RuntimeStatePath = Join-Path $script:RuntimeDirectory "runtime-state.json"
$pythonPath = Join-Path $script:RepositoryRoot ".venv\Scripts\python.exe"
$state = Read-RuntimeState
$backendProcess = Get-ManagedBackendProcess -State $state -PythonPath $pythonPath
$localPort = Get-JsonPropertyValue -Object $state -Name "local_port"
$previousFunnelPort = Get-JsonPropertyValue -Object $state -Name "previous_funnel_port"

if ($DisableFunnel -and ($localPort -is [int] -or $localPort -is [long])) {
    $script:BackendTarget = "http://127.0.0.1:$localPort"
    $tailscalePath = Resolve-TailscaleCommand
    $statusJson = Invoke-Tailscale `
        -CommandPath $tailscalePath `
        -Arguments @("funnel", "status", "--json")
    $storyFunnel = Get-MiniGPTFunnel -StatusJson $statusJson
    if ($null -ne $storyFunnel) {
        if ($previousFunnelPort -eq 8000) {
            $null = Invoke-Tailscale `
                -CommandPath $tailscalePath `
                -Arguments @("funnel", "--bg", "--yes", "8000")
            $script:BackendTarget = "http://127.0.0.1:8000"
            $updatedStatus = Invoke-Tailscale `
                -CommandPath $tailscalePath `
                -Arguments @("funnel", "status", "--json")
            if ($null -eq (Get-MiniGPTFunnel -StatusJson $updatedStatus)) {
                throw "The recorded port-8000 Funnel target was not restored."
            }
        } else {
            Remove-MiniGPTFunnel -CommandPath $tailscalePath
        }
    }
}

Stop-ExactProcess -Process $backendProcess
if (Test-Path -LiteralPath $script:RuntimeStatePath -PathType Leaf) {
    Remove-Item -LiteralPath $script:RuntimeStatePath -Force
}
Write-Host "Story Forge managed backend stopped."

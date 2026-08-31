[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "start_public_demo_tailscale.ps1")

function Invoke-MiniGPTPublicDemoStop {
    Set-Location -LiteralPath $script:RepositoryRoot
    $state = Read-RuntimeState
    $pythonPath = Join-Path $script:RepositoryRoot ".venv\Scripts\python.exe"
    $backendProcess = Get-ManagedBackendProcess -State $state -PythonPath $pythonPath
    $funnelStopped = $false

    try {
        $tailscalePath = Resolve-TailscaleCommand
        $funnelStatus = Invoke-Tailscale `
            -CommandPath $tailscalePath `
            -Arguments @("funnel", "status", "--json")
        $funnel = Get-MiniGPTFunnel -StatusJson $funnelStatus
        if ($null -ne $funnel) {
            $null = Invoke-Tailscale `
                -CommandPath $tailscalePath `
                -Arguments @("funnel", "--https=443", "off")
            $updatedStatus = Invoke-Tailscale `
                -CommandPath $tailscalePath `
                -Arguments @("funnel", "status", "--json")
            if ($null -ne (Get-MiniGPTFunnel -StatusJson $updatedStatus)) {
                throw "Tailscale Funnel still publishes the miniGPT target."
            }
        }
        $funnelStopped = $true
    } finally {
        Stop-ExactProcess -Process $backendProcess
    }

    if ($funnelStopped -and (Test-Path -LiteralPath $script:RuntimeStatePath -PathType Leaf)) {
        Remove-Item -LiteralPath $script:RuntimeStatePath -Force
    }
    Write-Host "miniGPT public demo backend and Funnel are stopped."
}

if ($MyInvocation.InvocationName -ne ".") {
    Invoke-MiniGPTPublicDemoStop
}

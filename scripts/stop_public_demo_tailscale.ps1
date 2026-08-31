[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "start_public_demo_tailscale.ps1")

function Invoke-MiniGPTPublicDemoStop {
    Set-Location -LiteralPath $script:RepositoryRoot
    $state = Read-RuntimeState
    $pythonPath = Join-Path $script:RepositoryRoot ".venv\Scripts\python.exe"
    $backendProcess = Get-ManagedBackendProcess -State $state -PythonPath $pythonPath
    $tailscalePath = Resolve-TailscaleCommand
    Remove-MiniGPTFunnel -CommandPath $tailscalePath
    Stop-ExactProcess -Process $backendProcess

    if (Test-Path -LiteralPath $script:RuntimeStatePath -PathType Leaf) {
        Remove-Item -LiteralPath $script:RuntimeStatePath -Force
    }
    Write-Host "miniGPT public demo backend and Funnel are stopped."
}

if ($MyInvocation.InvocationName -ne ".") {
    Invoke-MiniGPTPublicDemoStop
}

param (
    [string]$TASK
)

if (-not $TASK) {
    Write-Host "Error: No task specified."
    Write-Host "Usage: .\run-task.ps1 qa"
    Write-Host "       .\run-task.ps1 verification"
    exit 1
}

for ($ID = 15; $ID -le 15; $ID++) {
    Write-Host "Running $TASK for ID $ID..."

    poetry run run-drama `
        --model "gpt-4o-mini-2024-07-18" `
        --id $ID `
        --task $TASK `
        --report_folder reports
}
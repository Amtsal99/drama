param (
    [string]$TASK
)

if (-not $TASK) {
    Write-Host "Error: No task specified."
    Write-Host "Usage: .\run-task.ps1 qa"
    Write-Host "       .\run-task.ps1 verification"
    exit 1
}

for ($ID = 1; $ID -le 100; $ID++) {
    Write-Host "Running $TASK for ID $ID..."

    poetry run run-drama `
        --model "gemini-2.5-flash" `
        --id $ID `
        --task $TASK `
        --report_folder reports
}
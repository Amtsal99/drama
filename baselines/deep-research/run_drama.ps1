param (
    [string]$TASK
)

if (-not $TASK) {
    Write-Host "Error: No task specified."
    Write-Host "Usage: .\run.ps1 [qa|verification|...]"
    exit 1
}

python -u main.py `
    --test_file "../../drama-bench/subset/$TASK/query.json" `
    --test_task "$TASK" `
    --output_dir "results/$TASK" `
    --max_workers 1
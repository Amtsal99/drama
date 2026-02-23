param(
    [string]$Task
)

if ([string]::IsNullOrWhiteSpace($Task)) {
    Write-Error "Error: No task specified."
    Write-Host "Usage: .\run_task.ps1 -Task [qa|verification|...]"
    exit 1
}

if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -notmatch "^#" -and $_ -match "=" } | ForEach-Object {
        $key, $value = $_ -split '=', 2
        # Menghapus quote jika ada
        $value = $value -replace '^"|"$', '' -replace "^'|'$", ''
        [System.Environment]::SetEnvironmentVariable($key, $value, [System.EnvironmentVariableTarget]::Process)
    }
}

python format_data.py "../../drama-bench/subset/$Task/query.json" "./data/$Task.jsonl"

$LogFileOut = "test_tasks_${Task}_stdout.log"
$LogFileErr = "test_tasks_${Task}_stderr.log"
$OutputDir = "results/$Task"
$ArgsList = @(
    "-u", "run.py",
    "--test_file", "./data/$Task.jsonl",
    "--test_task", "$Task",
    "--headless",
    "--api_model", "gpt-4o-mini-2024-07-18",
    "--max_iter", "15",
    "--max_attached_imgs", "3",
    "--temperature", "1",
    "--fix_box_color",
    "--output_dir", "$OutputDir",
    "--download_dir", "downloads",
    "--max_workers", "1",
    "--seed", "42"
)

Write-Host "Starting Python task in background..."
Write-Host "Standard Log: $LogFileOut"
Write-Host "Error Log   : $LogFileErr"

Start-Process -FilePath "python" `
              -ArgumentList $ArgsList `
              -RedirectStandardOutput $LogFileOut `
              -RedirectStandardError $LogFileErr `
              -WindowStyle Hidden

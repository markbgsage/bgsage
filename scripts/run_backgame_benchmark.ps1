# Launch (or relaunch) the back game benchmark generation, one detached worker
# per subfolder.
#
# Why one process per subfolder rather than one at a time on every CPU:
# evaluation scales badly with threads. Measured 3-ply seconds per simulated turn
# (one cube action + one checker play) on real back game boards:
#
#     32 threads 0.124    8 threads 0.165    3 threads 0.258    1 thread 0.407
#
# so 32 threads buys only 3.3x over one. Aggregate throughput is
# processes / sec-per-turn, so with 11 subfolders the best split is 11 x 3
# threads (33 threads over 32 cores) = ~42.6 turns/s, against 8.0 turns/s
# running the subfolders one at a time on all 32.
#
# Safe to re-run at any time. Workers resume from their own checkpoints, a
# subfolder that already hit its target exits immediately, and a subfolder whose
# worker is still alive is skipped by the lock inside backgame_benchmark.py - so
# re-running after an accidental kill restarts exactly the dead workers and
# leaves the live ones alone.
#
#   .\scripts\run_backgame_benchmark.ps1                   # 10,000 each at 3ply
#   .\scripts\run_backgame_benchmark.ps1 -Count 1000
#   .\scripts\run_backgame_benchmark.ps1 -Level truncated2
#   .\scripts\run_backgame_benchmark.ps1 -Restart          # discard and start over
#
# Progress:  py -3.14 scripts\backgame_benchmark.py status
# Stop:      Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
#              Where-Object { $_.CommandLine -like '*backgame_benchmark.py*' } |
#              ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

param(
    [int]$Count = 10000,
    [string]$Level = '3ply',
    [int]$Threads = 3,
    [int]$MaxGames = 20000,
    # Launch only these categories. Use it to fit the run into a smaller share
    # of the machine: fewer processes is the ONLY way to cut CPU meaningfully,
    # because the engine floors a 3-ply analyzer at 2 threads, so -Threads 1 and
    # -Threads 2 are the same thing.
    [string[]]$Only,
    [switch]$Restart,
    [string]$Python = 'C:\Users\mghig\AppData\Local\Programs\Python\Python314\python.exe'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root 'logs'
if (-not (Test-Path $logs)) { New-Item -ItemType Directory -Path $logs | Out-Null }

if (-not (Test-Path $Python)) { throw "Python not found: $Python" }

$src = Join-Path $root 'backgame_ref_positions\Positions for Mark'
if (-not (Test-Path $src)) { throw "Reference positions not found: $src" }
$folders = Get-ChildItem -Path $src -Directory | Select-Object -ExpandProperty Name | Sort-Object
if ($Only) {
    $unknown = $Only | Where-Object { $folders -notcontains $_ }
    if ($unknown) { throw "Unknown category: $($unknown -join ', ')" }
    $folders = $folders | Where-Object { $Only -contains $_ }
}

$script = Join-Path 'scripts' 'backgame_benchmark.py'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

# Categories that already have a live worker. The lock inside the Python script
# is the real guard, but skipping them here matters for a second reason: the log
# redirection below TRUNCATES its target, so launching a doomed process for a
# live category would destroy that worker's progress log.
$live = @{}
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*backgame_benchmark.py*' } |
    ForEach-Object {
        if ($_.CommandLine -match '--folder "([^"]+)"') { $live[$matches[1]] = $_.ProcessId }
    }

foreach ($f in $folders) {
    if ($live.ContainsKey($f)) {
        Write-Output ("running  pid {0,-6} {1}  (skipped)" -f $live[$f], $f)
        continue
    }
    $slug = ($f -replace '[^A-Za-z0-9]', '_')
    # Start-Process joins ArgumentList with spaces and quotes nothing, so a
    # folder name with spaces has to carry its own quotes.
    $procArgs = @('-u', $script, 'generate',
                  '--count', "$Count", '--folder', ('"' + $f + '"'),
                  '--level', $Level, '--threads', "$Threads",
                  '--max-games', "$MaxGames")
    if ($Restart) { $procArgs += '--restart' }

    $p = Start-Process -FilePath $Python -ArgumentList $procArgs -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "backgame_${Level}_${slug}_$stamp.log") `
        -RedirectStandardError  (Join-Path $logs "backgame_${Level}_${slug}_$stamp.err.log") `
        -PassThru
    Write-Output ("launched pid {0,-6} {1}" -f $p.Id, $f)
}

Write-Output ""
Write-Output "$($folders.Count) workers launched at $Level, $Threads threads each."
Write-Output "Progress: py -3.14 scripts\backgame_benchmark.py status"

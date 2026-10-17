param(
    [string]$Python = 'python',
    [double]$MinimumFreeGiB = 6,
    [double]$RuntimeFloorGiB = 2
)
$ErrorActionPreference = 'Stop'
$engineRoot = Split-Path -Parent $PSScriptRoot
$buildPath = Join-Path $engineRoot 'build/video_move_engine'
$attempt = '{0}_{1}' -f ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')), $PID
$logPath = Join-Path $engineRoot "logs/video_move_engine/$attempt"

function Get-FreeGiB {
    return (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB / 1GB
}

function Get-SourceHashes {
    $paths = @((Join-Path $engineRoot 'CMakeLists.txt'),
        (Join-Path $engineRoot 'cpp/CMakeLists_cpu.txt'), $PSCommandPath)
    $paths += Get-ChildItem -LiteralPath (Join-Path $engineRoot 'cpp/include') -Recurse -File |
        Where-Object { $_.Extension -in '.h', '.hpp' } | Select-Object -ExpandProperty FullName
    $paths += Get-ChildItem -LiteralPath (Join-Path $engineRoot 'cpp/src') -File |
        Where-Object { $_.Extension -eq '.cpp' } | Select-Object -ExpandProperty FullName
    $paths += Get-ChildItem -LiteralPath (Join-Path $engineRoot 'cpp/pybind') -File |
        Where-Object { $_.Extension -in '.h', '.cpp' } | Select-Object -ExpandProperty FullName
    $result = [ordered]@{}
    foreach ($path in ($paths | Sort-Object -Unique)) {
        $relative = $path.Substring($engineRoot.Length + 1).Replace('\', '/')
        $result[$relative] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return $result
}

function Stop-OwnedTree([int]$RootId) {
    # Walk only descendants of the process this script created. Never select
    # processes by executable name: other sessions may also be compiling.
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$RootId")
    foreach ($child in $children) { Stop-OwnedTree ([int]$child.ProcessId) }
    Stop-Process -Id $RootId -ErrorAction SilentlyContinue
}

function Run-Guarded([string]$Name, [string[]]$Arguments) {
    $stdout = Join-Path $logPath "$Name.stdout.log"
    $stderr = Join-Path $logPath "$Name.stderr.log"
    $quoted = $Arguments | ForEach-Object {
        if ($_ -match '"') { throw 'Embedded quotes are not supported in build arguments' }
        '"' + $_ + '"'
    }
    $started = [DateTime]::UtcNow
    $process = Start-Process -FilePath $cmakePath -ArgumentList $quoted -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WorkingDirectory $engineRoot
    $minimumObserved = Get-FreeGiB
    $guardStopped = $false
    while (-not $process.HasExited) {
        $free = Get-FreeGiB
        $minimumObserved = [Math]::Min($minimumObserved, $free)
        if ($free -lt $RuntimeFloorGiB) {
            Stop-OwnedTree $process.Id
            $guardStopped = $true
            break
        }
        Start-Sleep -Milliseconds 1000
        $process.Refresh()
    }
    $process.WaitForExit()
    $entry = [ordered]@{
        phase = $Name; started_utc = $started.ToString('o'); process_id = $process.Id
        seconds = ([DateTime]::UtcNow - $started).TotalSeconds
        exit_code = $process.ExitCode; guard_stopped = $guardStopped
        minimum_free_gib = $minimumObserved; arguments = $Arguments
    }
    $entry | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $logPath "$Name.json")
    if ($guardStopped) { throw "Private $Name stopped at memory floor; partial build retained" }
    if ($process.ExitCode -ne 0) { throw "Private $Name failed; inspect $stdout and $stderr" }
    return $entry
}

if ($MinimumFreeGiB -lt $RuntimeFloorGiB -or $RuntimeFloorGiB -lt 1) {
    throw 'Memory limits must leave at least 1 GiB and startup must exceed the runtime floor'
}
$firstFree = Get-FreeGiB
if ($firstFree -lt $MinimumFreeGiB) { throw "Build not started: $firstFree GiB available" }
Start-Sleep -Seconds 3
$secondFree = Get-FreeGiB
if ($secondFree -lt $MinimumFreeGiB) { throw "Build not started: $secondFree GiB available" }

$pythonPath = (& $Python -c 'import sys; print(sys.executable)').Trim()
if ($LASTEXITCODE -ne 0) { throw 'Python discovery failed' }
$pybindPath = (& $pythonPath -m pybind11 --cmakedir).Trim()
if ($LASTEXITCODE -ne 0) { throw 'pybind11 discovery failed' }
$cmakePath = (Get-Command cmake -ErrorAction Stop).Source
$ninjaPath = (Get-Command ninja -ErrorAction Stop).Source
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vsRoot = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
    if (-not $vsRoot) { throw 'MSVC C++ toolchain not found' }
    $vcvars = Join-Path $vsRoot 'VC/Auxiliary/Build/vcvars64.bat'
    $compilerEnvironment = & $env:ComSpec /c "`"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) { throw 'MSVC environment initialization failed' }
    foreach ($line in $compilerEnvironment) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
}
New-Item -ItemType Directory -Path $logPath -Force | Out-Null
$before = Get-SourceHashes
$before | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $logPath 'source_before.json')
$configure = Run-Guarded 'configure' @('-S', $engineRoot, '-B', $buildPath, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_MAKE_PROGRAM=$ninjaPath", '-DCMAKE_CXX_COMPILER=cl',
    '-DBGBOT_BUILD_APPS=OFF', '-DPYBIND11_FINDPYTHON=ON', "-DPython_EXECUTABLE=$pythonPath",
    "-Dpybind11_DIR=$pybindPath")
$build = Run-Guarded 'build' @('--build', $buildPath, '--target', 'bgbot_cpp', '--parallel', '1')
$after = Get-SourceHashes
$after | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $logPath 'source_after.json')
if (($before | ConvertTo-Json -Compress) -cne ($after | ConvertTo-Json -Compress)) {
    throw 'Source changed during private build; artifact is not certified'
}
$artifact = @(Get-ChildItem -LiteralPath $buildPath -Filter 'bgbot_cpp*.pyd')
if ($artifact.Count -ne 1) { throw 'Expected exactly one private compiled extension' }
$compiler = (Get-Command cl -ErrorAction Stop).Source
$manifest = [ordered]@{
    completed_utc = [DateTime]::UtcNow.ToString('o'); source_unchanged = $true
    configure = $configure; build = $build; source_sha256 = $after
    python = $pythonPath; compiler = $compiler; compiler_version = (Get-Item $compiler).VersionInfo.FileVersion
    pybind_cmake = $pybindPath; cmake = $cmakePath; ninja = $ninjaPath
    startup_free_gib = @($firstFree, $secondFree); runtime_floor_gib = $RuntimeFloorGiB
    artifact = $artifact[0].FullName
    artifact_sha256 = (Get-FileHash -LiteralPath $artifact[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    manifest_path = Join-Path $logPath 'manifest.json'
    active_binary_replaced = $false; purpose = 'Private full CPU engine for exact move-array validation'
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $logPath 'manifest.json')
[pscustomobject]$manifest | Select-Object artifact, artifact_sha256, source_unchanged, manifest_path | ConvertTo-Json

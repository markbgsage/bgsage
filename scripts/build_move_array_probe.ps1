param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$engineRoot = Split-Path -Parent $PSScriptRoot
$buildPath = Join-Path $engineRoot 'build/video_move_batch'
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
& $cmakePath -S (Join-Path $engineRoot 'cpp/move_array_probe') -B $buildPath -G Ninja `
    '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_MAKE_PROGRAM=$ninjaPath" `
    '-DCMAKE_CXX_COMPILER=cl' '-DPYBIND11_FINDPYTHON=ON' `
    "-DPython_EXECUTABLE=$pythonPath" "-Dpybind11_DIR=$pybindPath"
if ($LASTEXITCODE -ne 0) { throw 'Private move-array probe configure failed' }
& $cmakePath --build $buildPath --parallel 1
if ($LASTEXITCODE -ne 0) { throw 'Private move-array probe build failed' }
Get-ChildItem -LiteralPath $buildPath -Filter '_bgsage_move_array_probe*.pyd' |
    Select-Object FullName, Length

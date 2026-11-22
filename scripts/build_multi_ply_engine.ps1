param(
    [Parameter(Mandatory=$true)][string]$Python,
    [string]$BuildName = 'multi_ply_engine',
    [string]$Source,
    [string]$Pybind,
    [int]$Jobs = 4
)
# Build a separate Release extension without replacing the baseline in build/.
$ErrorActionPreference = 'Stop'
$engineRoot = Split-Path -Parent $PSScriptRoot
if (-not $Source) { $Source = $engineRoot }
$Source = (Resolve-Path -LiteralPath $Source).Path
if ($Source -ne $engineRoot -and !$Source.StartsWith($engineRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Source must be within the engine checkout'
}
if ($BuildName -notmatch '^multi_ply_[a-zA-Z0-9_]+$') { throw 'Invalid build directory name' }
$build = Join-Path $engineRoot "build/$BuildName"
$Python = (Resolve-Path -LiteralPath $Python).Path
if (-not $Pybind) { $Pybind = Join-Path $engineRoot 'build/multi_ply_tools/pybind11/share/cmake/pybind11' }
if (!(Test-Path -LiteralPath $Pybind)) { $Pybind = (& $Python -m pybind11 --cmakedir).Trim() }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vsRoot = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
$vcvars = Join-Path $vsRoot 'VC/Auxiliary/Build/vcvars64.bat'
$compilerEnv = & $env:ComSpec /c "`"$vcvars`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed' }
foreach ($line in $compilerEnv) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$cmake = Join-Path $vsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$ninja = Join-Path $vsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
& $cmake -S $Source -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_MAKE_PROGRAM=$ninja" -DCMAKE_CXX_COMPILER=cl -DBGBOT_BUILD_APPS=OFF -DPYBIND11_FINDPYTHON=ON "-DPython_EXECUTABLE=$Python" "-Dpybind11_DIR=$Pybind"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
& $cmake --build $build --target bgbot_cpp --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'Native compilation failed' }

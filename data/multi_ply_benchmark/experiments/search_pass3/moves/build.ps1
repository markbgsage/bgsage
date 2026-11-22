param(
    [string]$BuildDirectory = 'build/search_pass3_moves_replay',
    [string]$BaselineDirectory = 'build/multi_ply_second_final',
    [string[]]$Arms = @('base', 'ordered', 'reverse', 'both', 'sorted', 'fast', 'relative', 'lookup'),
    [switch]$NativeOnly
)
$ErrorActionPreference = 'Stop'
[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]3840
$root = (Resolve-Path "$PSScriptRoot/../../../../..").Path
$work = [IO.Path]::GetFullPath((Join-Path $root $BuildDirectory))
$build = (Resolve-Path "$root/build").Path
if (!$work.StartsWith($build + [IO.Path]::DirectorySeparatorChar) -or (Test-Path $work)) {
    throw 'Choose a new directory under build for the replay.'
}
New-Item -ItemType Directory -Path $work | Out-Null
$baseline = (Resolve-Path (Join-Path $root $BaselineDirectory)).Path
$revision = '82e63972379b77ab52ebafc716620e813ff7baa1'
$sources = "$work/sources.zip"
& git -C $root archive --format=zip "--output=$sources" $revision cpp/include cpp/src/moves.cpp cpp/src/pubeval.cpp
if ($LASTEXITCODE) { throw 'Could not export the checkpoint sources' }
Expand-Archive "$PSScriptRoot/variants.zip" "$work/patches"
Expand-Archive $sources "$work/reference"
foreach ($arm in $Arms) {
    $dir = "$work/$arm"
    Expand-Archive $sources $dir
    if ($arm -ne 'base') {
        $prefix = [IO.Path]::GetRelativePath($root, $dir).Replace('\','/')
        & git -C $root apply "--directory=$prefix" "$work/patches/$arm.patch"
        if ($LASTEXITCODE) { throw "Cannot apply $arm" }
    }
}

$line = ((Get-Content "$baseline/build.ninja") | Select-String '^  LINK_LIBRARIES = .*python\d+\.lib' | Select-Object -First 1).Line
if ($line -notmatch '(\S+:\\[^ ]+\\python\d+\.lib)') { throw 'Missing CPython import library' }
$pythonLib = $Matches[1]
$target = (Get-ChildItem $baseline -Filter 'bgbot_cpp.*.pyd').Name
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
$vars = Join-Path $vs 'VC/Auxiliary/Build/vcvars64.bat'
$compiler = & $env:ComSpec /c ('"' + $vars + '" >nul && set')
foreach ($setting in $compiler) {
    if ($setting -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$libs = "$baseline/bgbot_lib.lib", "$baseline/bgbot_cuda_stub.lib"
$binding = "$baseline/CMakeFiles/bgbot_cpp.dir/cpp/pybind/bindings.cpp.obj"
$systemLibs = 'kernel32.lib','user32.lib','gdi32.lib','winspool.lib','shell32.lib','ole32.lib','oleaut32.lib','uuid.lib','comdlg32.lib','advapi32.lib'
$flags = '/nologo','/DWIN32','/D_WINDOWS','/GR','/EHsc','/O2','/Ob2','/DNDEBUG','/std:c++17','/MD','/fp:fast','/GL','/arch:AVX2','/Icpp/include','/c'

Push-Location "$work/reference"
try {
    & cl @flags /Dpossible_boards_unsorted=reference_unsorted /Dpossible_boards=reference_sorted /Dpossible_boards_one_die=reference_one /Foreference.obj cpp/src/moves.cpp
    if ($LASTEXITCODE) { throw 'Reference compile failed' }
} finally { Pop-Location }

foreach ($arm in $Arms) {
    Push-Location "$work/$arm"
    try {
        $files = @('cpp/src/moves.cpp')
        $objects = @('moves.obj')
        if ($arm -eq 'lookup') { $files += 'cpp/src/pubeval.cpp'; $objects += 'pubeval.obj' }
        & cl @flags @files "$PSScriptRoot/profile.cpp" "$PSScriptRoot/verify.cpp"
        if ($LASTEXITCODE) { throw "Compile failed for $arm" }
        & link /nologo /LTCG /CGTHREADS:2 /OUT:profile.exe profile.obj @objects @libs @systemLibs
        if ($LASTEXITCODE) { throw "Profiler link failed for $arm" }
        & link /nologo /LTCG /CGTHREADS:2 /OUT:verify.exe verify.obj "$work/reference/reference.obj" @objects @libs @systemLibs
        if ($LASTEXITCODE) { throw "Verifier link failed for $arm" }
        if (!$NativeOnly) {
            & link /nologo /DLL /LTCG /CGTHREADS:2 /INCREMENTAL:NO "/OUT:$target" @objects $binding @libs $pythonLib @systemLibs
            if ($LASTEXITCODE) { throw "Extension link failed for $arm" }
        }
    } finally { Pop-Location }
}

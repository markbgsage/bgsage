param(
    [string]$BuildDirectory = 'build/search_pass2_moves_replay',
    [string]$BaselineDirectory = 'build/multi_ply_truncated_final',
    [switch]$SourcesOnly
)
$ErrorActionPreference = 'Stop'
[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]15
$root = (Resolve-Path "$PSScriptRoot/../../../../..").Path
$work = [IO.Path]::GetFullPath((Join-Path $root $BuildDirectory))
$baseline = (Resolve-Path (Join-Path $root $BaselineDirectory)).Path
$buildRoot = (Resolve-Path (Join-Path $root 'build')).Path
if (!$work.StartsWith($buildRoot + [IO.Path]::DirectorySeparatorChar) -or (Test-Path $work)) {
    throw 'Choose a new directory under the engine build directory.'
}
$revision = '33c4aa1f6c28d12b846e66e745eef6c2692c5052'
New-Item -ItemType Directory -Path $work | Out-Null
$archive = Join-Path $work 'sources.zip'
& git -C $root archive --format=zip "--output=$archive" $revision cpp/include cpp/src/moves.cpp cpp/src/pubeval.cpp
if ($LASTEXITCODE) { throw 'Could not export the pinned sources' }
Expand-Archive (Join-Path $PSScriptRoot 'variants.zip') (Join-Path $work 'patches')
$variants = 'base', 'sparse', 'first', 'inplace', 'hybrid'
foreach ($arm in $variants) {
    $source = Join-Path $work $arm
    Expand-Archive $archive $source
    if ($arm -ne 'base') {
        $prefix = [IO.Path]::GetRelativePath($root, $source).Replace('\','/')
        $patch = Join-Path $work "patches/$arm.patch"
        & git -C $root apply "--directory=$prefix" $patch
        if ($LASTEXITCODE) { throw "Patch failed for $arm" }
    }
}
if ($SourcesOnly) { return }

$config = Get-Content (Join-Path $baseline 'build.ninja')
$line = ($config | Select-String '^  LINK_LIBRARIES = .*python\d+\.lib' | Select-Object -First 1).Line
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
$libs = "$(Join-Path $baseline 'bgbot_lib.lib')", "$(Join-Path $baseline 'bgbot_cuda_stub.lib')"
$binding = Join-Path $baseline 'CMakeFiles/bgbot_cpp.dir/cpp/pybind/bindings.cpp.obj'
$systemLibs = 'kernel32.lib','user32.lib','gdi32.lib','winspool.lib','shell32.lib','ole32.lib','oleaut32.lib','uuid.lib','comdlg32.lib','advapi32.lib'
foreach ($arm in $variants) {
    Push-Location (Join-Path $work $arm)
    try {
        & cl /nologo /DWIN32 /D_WINDOWS /GR /EHsc /O2 /Ob2 /DNDEBUG /std:c++17 /MD /fp:fast /GL /arch:AVX2 /Icpp/include /c cpp/src/moves.cpp cpp/src/pubeval.cpp "$PSScriptRoot/check.cpp"
        if ($LASTEXITCODE) { throw "Compilation failed for $arm" }
        & link /nologo /DLL /LTCG /CGTHREADS:2 /INCREMENTAL:NO "/OUT:$target" moves.obj pubeval.obj $binding @libs $pythonLib @systemLibs
        if ($LASTEXITCODE) { throw "Extension link failed for $arm" }
        & link /nologo /LTCG /CGTHREADS:2 /INCREMENTAL:NO /SUBSYSTEM:CONSOLE /OUT:check.exe check.obj moves.obj pubeval.obj @libs @systemLibs
        if ($LASTEXITCODE) { throw "Native check link failed for $arm" }
    } finally { Pop-Location }
}

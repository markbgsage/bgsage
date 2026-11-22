param([string[]]$Arms=@('route','batch','fuse'), [switch]$Checks)
$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '../../../../..')).Path
$studyBase=Join-Path $root 'build/multi_ply_cpu_fixed'
$caseRoot=Join-Path $root 'build/trunc_eval_study'
[Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]0xf0
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs=(& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
$vars=Join-Path $vs 'VC/Auxiliary/Build/vcvars64.bat'
$compiler=& $env:ComSpec /c "`"$vars`" >nul && set"
if ($LASTEXITCODE -ne 0){throw 'Developer compiler environment unavailable'}
foreach($line in $compiler) {if ($line -match '^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}}
$config=Get-Content (Join-Path $studyBase 'build.ninja')
$linkLine=($config|Select-String '^  LINK_LIBRARIES = .*python\d+\.lib'|Select-Object -First 1).Line
if ($linkLine -notmatch '(\S+:\\[^ ]+\\python\d+\.lib)'){throw 'Python library unavailable'}
$py=$Matches[1]
$target=(Get-ChildItem $studyBase -Filter 'bgbot_cpp.*.pyd'|Select-Object -First 1).Name
if ($Checks) {
    $dir=New-Item -ItemType Directory -Force (Join-Path $caseRoot 'native_checks')
    Push-Location $dir
    try {
        & cl /nologo /MD /O2 /GL /EHsc /std:c++17 /fp:fast /arch:AVX2 /DNDEBUG "/I$root/cpp/include" "$PSScriptRoot/native_checks.cpp" "$root/cpp/src/neural_net.cpp" "$root/cpp/src/strategy.cpp" /link /LTCG /CGTHREADS:2 "$studyBase/bgbot_lib.lib" "$studyBase/bgbot_cuda_stub.lib" /OUT:native_checks.exe
        if ($LASTEXITCODE -ne 0) {throw 'Native check build failed'}
        & ./native_checks.exe "$root/data/bearoff_1sided.db"
        if ($LASTEXITCODE -ne 0) {throw 'Native checks failed'}
    } finally {Pop-Location}
    return
}
foreach ($arm in $Arms) {
    Push-Location (Join-Path $caseRoot $arm)
    try {
        $objects=@()
        foreach ($src in (Get-ChildItem -File *.cpp)) {
            & cl /nologo /MD /O2 /Ob2 /GL /EHsc /std:c++17 /fp:fast /arch:AVX2 /DNDEBUG "/I$root/cpp/include" /c $src.Name
            if ($LASTEXITCODE -ne 0) {throw "Compile failed for $src"}
            $objects+=@($src.BaseName+'.obj')
        }
        & link /nologo /DLL /LTCG /CGTHREADS:2 /INCREMENTAL:NO "/OUT:$target" @objects "$studyBase/CMakeFiles/bgbot_cpp.dir/cpp/pybind/bindings.cpp.obj" "$studyBase/bgbot_lib.lib" "$studyBase/bgbot_cuda_stub.lib" $py kernel32.lib user32.lib advapi32.lib
        if ($LASTEXITCODE -ne 0){throw 'Link failed'}
    } finally {Pop-Location}
}

param([Parameter(Mandatory=$true)][string]$Name,[switch]$Extension)
$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
$work=Join-Path $root "build/cpu_feature_masks/$Name"
$baseline=Join-Path $root 'build/multi_ply_release'
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs=(& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
$vars=Join-Path $vs 'VC/Auxiliary/Build/vcvars64.bat'
$compiler=& $env:ComSpec /c "`"$vars`" >nul && set"
foreach($line in $compiler) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process') }
}
New-Item -ItemType Directory -Force -Path $work | Out-Null
Push-Location $work
try {
    $objects=@()
    if (Test-Path 'encoding.cpp') {
        & cl /nologo /MD /O2 /Ob2 /Gy /GL /EHsc /std:c++17 /fp:fast /arch:AVX2 /DNDEBUG "/I$root/cpp/include" /c encoding.cpp
        if ($LASTEXITCODE -ne 0) {throw 'Encoder compile failed'}
        $objects+=@('encoding.obj')
    }
    if($Extension) {
        $config=Get-Content (Join-Path $baseline 'build.ninja')
        $link=($config|Select-String '^  LINK_LIBRARIES = .*python\d+\.lib'|Select-Object -First 1).Line
        if($link -notmatch '(\S+:\\[^ ]+\\python\d+\.lib)'){throw 'Cannot resolve CPython lib'}
        $py=$Matches[1]
        $target=(Get-ChildItem $baseline -Filter 'bgbot_cpp.*.pyd'|Select-Object -First 1).Name
        & link /nologo /DLL /LTCG /CGTHREADS:1 /INCREMENTAL:NO "/OUT:$target" @objects "$baseline/CMakeFiles/bgbot_cpp.dir/cpp/pybind/bindings.cpp.obj" "$baseline/bgbot_lib.lib" "$baseline/bgbot_cuda_stub.lib" $py kernel32.lib user32.lib advapi32.lib
    } else {
        & cl /nologo /MD /O2 /Ob2 /Gy /EHsc /std:c++17 /fp:fast /arch:AVX2 /DNDEBUG "/I$root/cpp/include" /c "$PSScriptRoot/probe.cpp"
        if ($LASTEXITCODE -ne 0) {throw 'Probe compile failed'}
        & link /nologo /LTCG /CGTHREADS:1 /OPT:REF /OUT:probe.exe probe.obj @objects "$baseline/bgbot_lib.lib"
    }
    if ($LASTEXITCODE -ne 0) {throw 'Private link failed'}
} finally {Pop-Location}

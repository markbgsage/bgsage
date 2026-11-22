param([Parameter(Mandatory=$true)][string[]]$Arms, [switch]$DeltaCheck)
$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '../../../../..')).Path
$studyBase=Join-Path $root 'build/multi_ply_truncated_final'
$caseRoot=Join-Path $root 'build/det_eval2'
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
foreach ($arm in $Arms) {
    Push-Location (Join-Path $caseRoot $arm)
    try {
        if ($DeltaCheck) {
            & cl /nologo /MD /O2 /Ob2 /GL /EHsc /std:c++17 /fp:fast /arch:AVX2 /DNDEBUG /Iinclude /c "$PSScriptRoot/check_deltas.cpp"
            if ($LASTEXITCODE -ne 0) { throw 'Delta parity probe compilation failed' }
            & link /nologo /LTCG /CGTHREADS:2 /OUT:check_deltas.exe check_deltas.obj encoding.obj neural_net.obj "$studyBase/bgbot_lib.lib" "$studyBase/bgbot_cuda_stub.lib"
            if ($LASTEXITCODE -ne 0) { throw 'Delta parity probe linking failed' }
            & ./check_deltas.exe "$root/build/nn_trials/corpus.txt" "$root/build/cpu_feature_masks/all.bin"
            if ($LASTEXITCODE -ne 0) { throw 'Delta parity probe failed' }
            continue
        }
        $objects=@()
        foreach ($src in (Get-ChildItem -File *.cpp)) {
            & cl /nologo /MD /O2 /Ob2 /GL /EHsc /std:c++17 /fp:fast /arch:AVX2 /DNDEBUG "/Iinclude" /c $src.Name
            if ($LASTEXITCODE -ne 0) {throw "Compile failed for $src"}
            $objects+=@($src.BaseName+'.obj')
        }
        & link /nologo /DLL /LTCG /CGTHREADS:2 /INCREMENTAL:NO "/OUT:$target" @objects "$studyBase/CMakeFiles/bgbot_cpp.dir/cpp/pybind/bindings.cpp.obj" "$studyBase/bgbot_lib.lib" "$studyBase/bgbot_cuda_stub.lib" $py kernel32.lib user32.lib advapi32.lib
        if ($LASTEXITCODE -ne 0){throw 'Link failed'}
    } finally {Pop-Location}
}

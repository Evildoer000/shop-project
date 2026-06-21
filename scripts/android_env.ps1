$RepoRoot = Split-Path -Parent $PSScriptRoot
$JdkHome = Join-Path $RepoRoot ".local_android_env\jdk17\jdk-17.0.19+10"
$AndroidSdkRoot = Join-Path $RepoRoot ".local_android_env\android-sdk"

if (-not (Test-Path (Join-Path $JdkHome "bin\java.exe"))) {
    throw "JDK 17 was not found at $JdkHome"
}

if (-not (Test-Path (Join-Path $AndroidSdkRoot "platform-tools\adb.exe"))) {
    throw "Android SDK platform-tools were not found at $AndroidSdkRoot"
}

$env:JAVA_HOME = $JdkHome
$env:ANDROID_HOME = $AndroidSdkRoot
$env:ANDROID_SDK_ROOT = $AndroidSdkRoot
$env:GRADLE_USER_HOME = Join-Path $RepoRoot ".local_android_env\gradle-home"
$env:PATH = "$JdkHome\bin;$AndroidSdkRoot\cmdline-tools\latest\bin;$AndroidSdkRoot\platform-tools;$AndroidSdkRoot\emulator;$env:PATH"

Write-Host "JAVA_HOME=$env:JAVA_HOME"
Write-Host "ANDROID_HOME=$env:ANDROID_HOME"
Write-Host "GRADLE_USER_HOME=$env:GRADLE_USER_HOME"
java -version
adb version

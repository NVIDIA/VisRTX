# CUDA Installation Script for Windows using Redistributables
# Downloads and installs CUDA components from NVIDIA redistributable archives
# with SHA256 checksum verification.

param(
    [Parameter(Mandatory=$true)]
    [string]$CudaVersion,

    [Parameter(Mandatory=$true)]
    [string]$InstallPath,

    [Parameter(Mandatory=$false)]
    [string]$ComponentsFile
)

$ErrorActionPreference = "Stop"

# Base URL for CUDA redistributables
$redistBaseUrl = "https://developer.download.nvidia.com/compute/cuda/redist"

# Platform identifier for Windows
$platform = "windows-x86_64"

# Required components (can be overridden via components.json)
$defaultComponents = @(
    "cuda_nvcc",
    "cuda_cudart",
    "cuda_cccl",
    "cuda_nvml_dev",
    "visual_studio_integration"
)

function Compare-Version {
    param(
        [string]$Version1,
        [string]$Version2
    )
    # Returns: -1 if v1 < v2, 0 if equal, 1 if v1 > v2
    $v1Parts = $Version1.Split('.') | ForEach-Object { [int]$_ }
    $v2Parts = $Version2.Split('.') | ForEach-Object { [int]$_ }

    for ($i = 0; $i -lt [Math]::Max($v1Parts.Count, $v2Parts.Count); $i++) {
        $p1 = if ($i -lt $v1Parts.Count) { $v1Parts[$i] } else { 0 }
        $p2 = if ($i -lt $v2Parts.Count) { $v2Parts[$i] } else { 0 }
        if ($p1 -lt $p2) { return -1 }
        if ($p1 -gt $p2) { return 1 }
    }
    return 0
}

function Get-RequiredComponents {
    param(
        [string]$ComponentsFile,
        [string]$CudaVersion
    )

    if ($ComponentsFile -and (Test-Path $ComponentsFile)) {
        $config = Get-Content $ComponentsFile | ConvertFrom-Json
        $components = @()
        foreach ($name in $config.components.PSObject.Properties.Name) {
            $comp = $config.components.$name

            # Check if this component is for our platform
            if ($comp.platforms -and $comp.platforms -notcontains $platform) {
                continue
            }

            # Check min_version requirement
            if ($comp.min_version) {
                if ((Compare-Version $CudaVersion $comp.min_version) -lt 0) {
                    Write-Host "Skipping $name (requires CUDA >= $($comp.min_version))"
                    continue
                }
            }

            # Include if required OR if min_version is satisfied (for optional components)
            if ($comp.required -or $comp.min_version) {
                $components += $name
            }
        }
        return $components
    }
    return $defaultComponents
}

function Download-File {
    param(
        [string]$Url,
        [string]$OutFile,
        [string]$ExpectedHash
    )

    Write-Host "Downloading: $Url"
    # Use Invoke-WebRequest (System.Net.WebClient is deprecated in modern .NET)
    Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing

    if ($ExpectedHash) {
        $actualHash = (Get-FileHash -Path $OutFile -Algorithm SHA256).Hash.ToLower()
        if ($actualHash -ne $ExpectedHash.ToLower()) {
            Remove-Item $OutFile -ErrorAction SilentlyContinue
            throw "Checksum mismatch for $OutFile`nExpected: $ExpectedHash`nActual: $actualHash"
        }
        Write-Host "  Checksum verified: $actualHash"
    }
}

function Install-CudaComponent {
    param(
        [string]$ComponentName,
        [object]$ComponentInfo,
        [string]$InstallPath,
        [string]$TempDir
    )

    $platformInfo = $ComponentInfo.$platform
    if (-not $platformInfo) {
        Write-Host "  Component $ComponentName not available for $platform, skipping"
        return $false
    }

    $relativePath = $platformInfo.relative_path
    $sha256 = $platformInfo.sha256
    $downloadUrl = "$redistBaseUrl/$relativePath"
    $zipFile = Join-Path $TempDir "$ComponentName.zip"

    Write-Host "Installing component: $ComponentName"
    Download-File -Url $downloadUrl -OutFile $zipFile -ExpectedHash $sha256

    # Extract to temp directory first
    $extractDir = Join-Path $TempDir "${ComponentName}_extract"
    Write-Host "  Extracting to: $extractDir"
    Expand-Archive -Path $zipFile -DestinationPath $extractDir -Force

    # Find the actual content directory (usually named like cuda_nvcc-version)
    $contentDirs = Get-ChildItem -Path $extractDir -Directory
    if ($contentDirs.Count -eq 1) {
        $sourceDir = $contentDirs[0].FullName
    } else {
        $sourceDir = $extractDir
    }

    # Copy contents to install path, merging directories
    Write-Host "  Merging to: $InstallPath"
    Copy-ItemWithMerge -Source $sourceDir -Destination $InstallPath

    # Cleanup
    Remove-Item $zipFile -Force
    Remove-Item $extractDir -Recurse -Force

    return $true
}

function Copy-ItemWithMerge {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (-not (Test-Path $Destination)) {
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    }

    Get-ChildItem -Path $Source | ForEach-Object {
        $destPath = Join-Path $Destination $_.Name
        if ($_.PSIsContainer) {
            Copy-ItemWithMerge -Source $_.FullName -Destination $destPath
        } else {
            Copy-Item -Path $_.FullName -Destination $destPath -Force
        }
    }
}

function Install-VisualStudioIntegration {
    param(
        [string]$CudaPath
    )

    # VS integration files need to be copied to MSBuild directories
    # The redistributable package structure varies by CUDA version, so check multiple paths
    $potentialPaths = @(
        # CUDA 12.x redistributable structure
        (Join-Path $CudaPath "visual_studio_integration\CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions"),
        (Join-Path $CudaPath "extras\visual_studio_integration\MSBuildExtensions"),
        # Some versions have it directly under visual_studio_integration
        (Join-Path $CudaPath "visual_studio_integration\MSBuildExtensions"),
        # CUDA toolkit installer structure
        (Join-Path $CudaPath "extras\visual_studio_integration\MSBuild")
    )

    $vsIntegrationSource = $null
    foreach ($path in $potentialPaths) {
        Write-Host "  Checking: $path"
        if (Test-Path $path) {
            $vsIntegrationSource = $path
            Write-Host "  Found VS integration at: $path"
            break
        }
    }

    if (-not $vsIntegrationSource) {
        # Search recursively for MSBuild extension files as fallback
        Write-Host "  Searching recursively for CUDA MSBuild extensions..."
        $cudaProps = Get-ChildItem -Path $CudaPath -Recurse -Filter "CUDA *.props" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cudaProps) {
            $vsIntegrationSource = $cudaProps.DirectoryName
            Write-Host "  Found VS integration at: $vsIntegrationSource"
        }
    }

    if ($vsIntegrationSource) {
        Write-Host "Installing Visual Studio integration from: $vsIntegrationSource"

        $installed = $false

        # Find VS installation paths using vswhere
        $vswherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vswherePath) {
            $vsInstalls = & $vswherePath -products * -requires Microsoft.Component.MSBuild -property installationPath
            foreach ($vsInstall in $vsInstalls) {
                # Try both v170 (VS 2022) and v160 (VS 2019) paths
                foreach ($vcVersion in @("v170", "v160")) {
                    $msbuildExtPath = Join-Path $vsInstall "MSBuild\Microsoft\VC\$vcVersion\BuildCustomizations"
                    if (Test-Path $msbuildExtPath) {
                        Write-Host "  Copying to: $msbuildExtPath"
                        Copy-Item -Path "$vsIntegrationSource\*" -Destination $msbuildExtPath -Recurse -Force
                        $installed = $true
                    }
                }
            }
        }

        # Also try common MSBuild paths for GitHub Actions runners
        $commonPaths = @(
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\MSBuild\Microsoft\VC\v170\BuildCustomizations",
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\MSBuild\Microsoft\VC\v170\BuildCustomizations",
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations",
            "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Enterprise\MSBuild\Microsoft\VC\v160\BuildCustomizations"
        )

        foreach ($msbuildPath in $commonPaths) {
            if ((Test-Path $msbuildPath) -and -not $installed) {
                Write-Host "  Copying to: $msbuildPath"
                Copy-Item -Path "$vsIntegrationSource\*" -Destination $msbuildPath -Recurse -Force
                $installed = $true
            }
        }

        if ($installed) {
            Write-Host "  Visual Studio integration installed successfully"
        } else {
            Write-Host "Warning: Could not find Visual Studio MSBuild directory to install CUDA integration"
        }
    } else {
        Write-Host "Warning: Visual Studio integration files not found"
        Write-Host "  Searched in: $($potentialPaths -join ', ')"
        Write-Host "  Listing contents of $CudaPath for debugging:"
        Get-ChildItem -Path $CudaPath -Depth 2 | ForEach-Object { Write-Host "    $($_.FullName)" }
    }
}

# Main execution
Write-Host "=========================================="
Write-Host "CUDA Redistributable Installer for Windows"
Write-Host "=========================================="
Write-Host "CUDA Version: $CudaVersion"
Write-Host "Install Path: $InstallPath"
Write-Host "Platform: $platform"

# Download redistrib manifest
$manifestUrl = "$redistBaseUrl/redistrib_$CudaVersion.json"
$tempDir = Join-Path $env:TEMP "cuda_install_$([guid]::NewGuid().ToString('N').Substring(0,8))"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

$manifestFile = Join-Path $tempDir "redistrib.json"
Write-Host "`nDownloading manifest: $manifestUrl"
Download-File -Url $manifestUrl -OutFile $manifestFile

$manifest = Get-Content $manifestFile | ConvertFrom-Json

# Create install directory
if (-not (Test-Path $InstallPath)) {
    New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
}

# Get required components
$components = Get-RequiredComponents -ComponentsFile $ComponentsFile -CudaVersion $CudaVersion
Write-Host "`nComponents to install: $($components -join ', ')"

# Install each component
$installedCount = 0
foreach ($componentName in $components) {
    $componentInfo = $manifest.$componentName
    if ($componentInfo) {
        $result = Install-CudaComponent -ComponentName $componentName `
                                        -ComponentInfo $componentInfo `
                                        -InstallPath $InstallPath `
                                        -TempDir $tempDir
        if ($result) {
            $installedCount++
        }
    } else {
        Write-Host "Warning: Component '$componentName' not found in manifest"
    }
}

# Install VS integration
Install-VisualStudioIntegration -CudaPath $InstallPath

# Cleanup temp directory
Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "`n=========================================="
Write-Host "Installation complete!"
Write-Host "Installed $installedCount components to: $InstallPath"
Write-Host "=========================================="

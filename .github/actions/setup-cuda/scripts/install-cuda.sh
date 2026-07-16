#!/bin/bash
# CUDA Installation Script for Linux using Redistributables
# Downloads and installs CUDA components from NVIDIA redistributable archives
# with SHA256 checksum verification.

set -euo pipefail

usage() {
    echo "Usage: $0 -v VERSION -p INSTALL_PATH [-c COMPONENTS_FILE]"
    echo ""
    echo "Options:"
    echo "  -v VERSION        CUDA version (e.g., 12.4.1, 13.0.2)"
    echo "  -p INSTALL_PATH   Installation directory (e.g., ~/cuda-12.4)"
    echo "  -c COMPONENTS     Path to components.json (optional)"
    exit 1
}

# Parse arguments
CUDA_VERSION=""
INSTALL_PATH=""
COMPONENTS_FILE=""

while getopts "v:p:c:h" opt; do
    case $opt in
        v) CUDA_VERSION="$OPTARG" ;;
        p) INSTALL_PATH="$OPTARG" ;;
        c) COMPONENTS_FILE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ -z "$CUDA_VERSION" || -z "$INSTALL_PATH" ]]; then
    echo "Error: VERSION and INSTALL_PATH are required"
    usage
fi

# Configuration
REDIST_BASE_URL="https://developer.download.nvidia.com/compute/cuda/redist"
PLATFORM="linux-x86_64"
TEMP_DIR=$(mktemp -d)

# Default required components
DEFAULT_COMPONENTS=(
    "cuda_nvcc"
    "cuda_cudart"
    "cuda_cccl"
    "cuda_nvml_dev"
)

# Cleanup on exit
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Compare semantic versions: returns 0 if v1 >= v2, 1 otherwise
version_gte() {
    local v1="$1"
    local v2="$2"

    # Split into arrays
    IFS='.' read -ra v1_parts <<< "$v1"
    IFS='.' read -ra v2_parts <<< "$v2"

    local max_len=${#v1_parts[@]}
    [[ ${#v2_parts[@]} -gt $max_len ]] && max_len=${#v2_parts[@]}

    for ((i = 0; i < max_len; i++)); do
        local p1=${v1_parts[i]:-0}
        local p2=${v2_parts[i]:-0}
        if ((p1 > p2)); then
            return 0
        elif ((p1 < p2)); then
            return 1
        fi
    done
    return 0
}

# Get required components from config file or use defaults
get_required_components() {
    if [[ -n "$COMPONENTS_FILE" && -f "$COMPONENTS_FILE" ]]; then
        if ! command -v jq &> /dev/null; then
            echo "Warning: jq not available, using default components" >&2
            printf '%s\n' "${DEFAULT_COMPONENTS[@]}"
            return
        fi

        # Process each component from the config
        jq -r '.components | to_entries[] | "\(.key)|\(.value.required)|\(.value.min_version // "")|\(.value.platforms // [] | join(","))"' "$COMPONENTS_FILE" | \
        while IFS='|' read -r name required min_version platforms; do
            # Skip if not for our platform
            if [[ -n "$platforms" && "$platforms" != *"$PLATFORM"* ]]; then
                continue
            fi

            # Check min_version requirement
            if [[ -n "$min_version" ]]; then
                if ! version_gte "$CUDA_VERSION" "$min_version"; then
                    echo "Skipping $name (requires CUDA >= $min_version)" >&2
                    continue
                fi
            fi

            # Include if required OR if min_version is satisfied (for optional components)
            if [[ "$required" == "true" ]] || [[ -n "$min_version" ]]; then
                echo "$name"
            fi
        done
    else
        printf '%s\n' "${DEFAULT_COMPONENTS[@]}"
    fi
}

# Download file with checksum verification
download_file() {
    local url="$1"
    local output="$2"
    local expected_hash="${3:-}"

    echo "Downloading: $url"
    if command -v curl &> /dev/null; then
        curl -fsSL "$url" -o "$output"
    else
        wget -q "$url" -O "$output"
    fi

    if [[ -n "$expected_hash" ]]; then
        local actual_hash
        actual_hash=$(sha256sum "$output" | cut -d' ' -f1)
        if [[ "${actual_hash,,}" != "${expected_hash,,}" ]]; then
            rm -f "$output"
            echo "Error: Checksum mismatch for $output"
            echo "  Expected: $expected_hash"
            echo "  Actual:   $actual_hash"
            exit 1
        fi
        echo "  Checksum verified: $actual_hash"
    fi
}

# Install a single component
install_component() {
    local component_name="$1"
    local manifest_file="$2"

    # Extract component info using jq
    local relative_path sha256

    if ! command -v jq &> /dev/null; then
        echo "Error: jq is required for parsing JSON manifests"
        exit 1
    fi

    # Check if component exists in manifest
    if ! jq -e ".$component_name" "$manifest_file" > /dev/null 2>&1; then
        echo "Warning: Component '$component_name' not found in manifest"
        return 1
    fi

    # Check if component is available for our platform
    if ! jq -e ".$component_name.\"$PLATFORM\"" "$manifest_file" > /dev/null 2>&1; then
        echo "  Component $component_name not available for $PLATFORM, skipping"
        return 0
    fi

    relative_path=$(jq -r ".$component_name.\"$PLATFORM\".relative_path" "$manifest_file")
    sha256=$(jq -r ".$component_name.\"$PLATFORM\".sha256" "$manifest_file")

    local download_url="$REDIST_BASE_URL/$relative_path"
    local archive_file="$TEMP_DIR/$component_name.tar.xz"
    local extract_dir="$TEMP_DIR/${component_name}_extract"

    echo "Installing component: $component_name"
    download_file "$download_url" "$archive_file" "$sha256"

    # Extract archive
    echo "  Extracting..."
    mkdir -p "$extract_dir"
    tar -xJf "$archive_file" -C "$extract_dir"

    # Find the content directory (usually named like cuda_nvcc-version)
    local content_dir
    content_dir=$(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d | head -1)
    if [[ -z "$content_dir" ]]; then
        content_dir="$extract_dir"
    fi

    # Merge contents to install path
    echo "  Merging to: $INSTALL_PATH"
    cp -a "$content_dir"/* "$INSTALL_PATH"/

    # Cleanup
    rm -f "$archive_file"
    rm -rf "$extract_dir"

    return 0
}

# Main execution
echo "=========================================="
echo "CUDA Redistributable Installer for Linux"
echo "=========================================="
echo "CUDA Version: $CUDA_VERSION"
echo "Install Path: $INSTALL_PATH"
echo "Platform: $PLATFORM"

# Download redistrib manifest
MANIFEST_URL="$REDIST_BASE_URL/redistrib_$CUDA_VERSION.json"
MANIFEST_FILE="$TEMP_DIR/redistrib.json"

echo ""
echo "Downloading manifest: $MANIFEST_URL"
download_file "$MANIFEST_URL" "$MANIFEST_FILE"

# Create install directory
mkdir -p "$INSTALL_PATH"

# Get components to install
echo ""
echo "Determining components to install..."
mapfile -t COMPONENTS < <(get_required_components)
echo "Components to install: ${COMPONENTS[*]}"

# Install each component
INSTALLED_COUNT=0
for component in "${COMPONENTS[@]}"; do
    if install_component "$component" "$MANIFEST_FILE"; then
        ((INSTALLED_COUNT++)) || true
    fi
done

# Create lib64 symlink if needed (nvcc expects lib64/ but redistributables use lib/)
if [[ -d "$INSTALL_PATH/lib" && ! -e "$INSTALL_PATH/lib64" ]]; then
    echo ""
    echo "Creating symlink: lib64 -> lib"
    ln -s lib "$INSTALL_PATH/lib64"
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "Installed $INSTALLED_COUNT components to: $INSTALL_PATH"
echo "=========================================="

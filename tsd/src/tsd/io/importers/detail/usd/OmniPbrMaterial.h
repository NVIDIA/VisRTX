// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if TSD_USE_USD

#include "MaterialCommon.h"

namespace tsd::io::materials {

/// Import an OmniPBR material as a physicallyBased TSD material
///
/// @param scene Scene to create material in
/// @param usdMat USD material to import from
/// @param basePath Base directory path for resolving relative texture paths
/// @param textureCache Cache for reusing loaded textures
/// @return Imported material reference
MaterialRef importOmniPBRMaterial(Scene &scene,
    const pxr::UsdShadeMaterial &usdMat,
    const pxr::UsdShadeShader &usdShader,
    const std::string &basePath,
    TextureCache &textureCache);

} // namespace tsd::io::materials

#endif

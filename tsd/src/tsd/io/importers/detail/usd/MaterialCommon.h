// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if TSD_USE_USD

#include "tsd/core/scene/Scene.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"

// pxr
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/base/gf/vec3f.h>

// std
#include <string>

namespace tsd::io::materials {

using namespace tsd::core;

// Helper functions for extracting shader parameters

/// Get a float input from a USD shader
bool getShaderFloatInput(const pxr::UsdShadeShader &shader,
    const char *inputName,
    float &outValue);

/// Get a bool input from a USD shader
bool getShaderBoolInput(const pxr::UsdShadeShader &shader,
    const char *inputName,
    bool &outValue);

/// Get a color3f input from a USD shader
bool getShaderColorInput(const pxr::UsdShadeShader &shader,
    const char *inputName,
    pxr::GfVec3f &outValue);

/// Get a texture file path from a USD shader input
/// Handles both connected texture reader nodes and direct asset path inputs
bool getShaderTextureInput(const pxr::UsdShadeShader &shader,
    const char *inputName,
    std::string &outFilePath);

} // namespace tsd::io::materials

#endif

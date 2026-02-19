// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/scene/Scene.hpp"
// std
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>
#if TSD_USE_VTK
// vtk
#include <vtkDataArray.h>
#endif

namespace tsd::io {

std::string pathOf(const std::string &filepath);
std::string fileOf(const std::string &filepath);
std::string extensionOf(const std::string &filepath);
std::vector<std::string> splitString(const std::string &s, char delim);

tsd::core::ArrayRef readArray(
    tsd::core::Scene &scene, anari::DataType elementType, std::FILE *fp);

using TextureCache = std::unordered_map<std::string, tsd::core::ArrayRef>;
tsd::core::SamplerRef importTexture(tsd::core::Scene &scene,
    std::string filepath,
    TextureCache &cache,
    bool isLinear = false);

tsd::core::SamplerRef makeDefaultColorMapSampler(
    tsd::core::Scene &scene, const tsd::math::float2 &range);

// Transfer function import functions
tsd::core::TransferFunction importTransferFunction(const std::string &filepath);

bool calcTangentsForTriangleMesh(const tsd::math::uint3 *indices,
    const tsd::math::float3 *vertexPositions,
    const tsd::math::float3 *vertexNormals,
    const tsd::math::float2 *texCoords,
    tsd::math::float4 *tangents,
    size_t numIndices,
    size_t numVertices);

#if TSD_USE_VTK
anari::DataType vtkTypeToANARIType(
    int vtkType, int numComps, const char *errorIdentifier = "");
tsd::core::ArrayRef makeArray1DFromVTK(tsd::core::Scene &scene,
    vtkDataArray *array,
    const char *errorIdentifier = "");
tsd::core::ArrayRef makeArray3DFromVTK(tsd::core::Scene &scene,
    vtkDataArray *array,
    size_t w,
    size_t h,
    size_t d,
    const char *errorIdentifier = "");
#endif

} // namespace tsd::io

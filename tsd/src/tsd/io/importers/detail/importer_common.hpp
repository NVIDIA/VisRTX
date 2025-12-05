// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Scene.hpp"
#include "tsd/core/ColorMapUtil.hpp"
// std
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

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
core::TransferFunction importTransferFunction(const std::string &filepath);

bool calcTangentsForTriangleMesh(const tsd::math::uint3 *indices,
    const tsd::math::float3 *vertexPositions,
    const tsd::math::float3 *vertexNormals,
    const tsd::math::float2 *texCoords,
    tsd::math::float4 *tangents,
    size_t numIndices,
    size_t numVertices);

} // namespace tsd::io

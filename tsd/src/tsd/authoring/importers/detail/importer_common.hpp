// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Context.hpp"
// std
#include <string>
#include <unordered_map>
#include <vector>

namespace tsd {

std::string pathOf(const std::string &filepath);
std::string fileOf(const std::string &filepath);
std::string extensionOf(const std::string &filepath);
std::vector<std::string> splitString(const std::string &s, char delim);

using TextureCache = std::unordered_map<std::string, ArrayRef>;
SamplerRef importTexture(Context &ctx,
    std::string filepath,
    TextureCache &cache,
    bool isLinear = false);

SamplerRef makeDefaultColorMapSampler(Context &ctx, const float2 &range);

bool calcTangentsForTriangleMesh(const uint3 *indices,
    const float3 *vertexPositions,
    const float3 *vertexNormals,
    const float3 *texCoords,
    float4 *tangents,
    size_t numIndices,
    size_t numVertices);

} // namespace tsd

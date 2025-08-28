// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Context.hpp"
// std
#include <string>
#include <unordered_map>
#include <vector>

namespace tsd::io {

std::string pathOf(const std::string &filepath);
std::string fileOf(const std::string &filepath);
std::string extensionOf(const std::string &filepath);
std::vector<std::string> splitString(const std::string &s, char delim);

using TextureCache = std::unordered_map<std::string, tsd::core::ArrayRef>;
tsd::core::SamplerRef importTexture(tsd::core::Context &ctx,
    std::string filepath,
    TextureCache &cache,
    bool isLinear = false);

tsd::core::SamplerRef makeDefaultColorMapSampler(
    tsd::core::Context &ctx, const tsd::math::float2 &range);

bool calcTangentsForTriangleMesh(const tsd::math::uint3 *indices,
    const tsd::math::float3 *vertexPositions,
    const tsd::math::float3 *vertexNormals,
    const tsd::math::float2 *texCoords,
    tsd::math::float4 *tangents,
    size_t numIndices,
    size_t numVertices);

} // namespace tsd::io

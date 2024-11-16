// Copyright 2024 NVIDIA Corporation
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

using TextureCache = std::unordered_map<std::string, SamplerRef>;
SamplerRef importTexture(
    Context &ctx, std::string filepath, TextureCache &cache);

} // namespace tsd

/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <nonstd/span.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace visrtx::materialx {

std::vector<std::string> enumerateRenderableMaterials(
    const std::filesystem::path &mtlxFile,
    nonstd::span<const std::filesystem::path> librarySearchPaths);

struct ParamMapping
{
  std::string cleanName;  // MaterialX input name (origin tail), e.g. "base_color"
  std::string mdlArgName; // generated MDL arg name, e.g. "srf_base_color"
};

struct TranscodeResult
{
  std::string mdlSource;
  std::string materialName;
  std::vector<std::string> available;
  std::vector<ParamMapping> paramMap; // clean MaterialX name -> generated MDL arg
  std::string error;
};

TranscodeResult transcodeMaterialXToMdl(
    const std::filesystem::path &mtlxFile,
    std::optional<std::string> selected,
    nonstd::span<const std::filesystem::path> librarySearchPaths);

} // namespace visrtx::materialx

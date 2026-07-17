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

struct DocumentSource
{
  enum class Kind { File, Inline };
  Kind kind;
  std::string value; // File: filesystem path; Inline: .mtlx XML text

  static DocumentSource file(const std::filesystem::path &p)
  {
    return {Kind::File, p.string()};
  }
  static DocumentSource inlineText(std::string xml)
  {
    return {Kind::Inline, std::move(xml)};
  }
};

// The MaterialX distribution resolved at runtime (ADR 0008). `root` is the
// directory containing the distribution's "libraries" folder; empty when no
// step of the chain resolved. `source` names the chain step that won, so a
// caller can tell an explicit-parameter win from a fallback. `trace` records
// each miss for diagnostics.
struct DistributionRoot
{
  enum class Source { None, Explicit, Environment, SelfDiscovery, Baked };
  std::filesystem::path root;
  Source source{Source::None};
  std::string trace;
};

const char *sourceName(DistributionRoot::Source source);

// Search chain, first hit wins: explicit roots (the materialxSearchPaths
// device parameter) -> MATERIALX_SEARCH_PATH -> MaterialX self-discovery
// (mx::getDefaultDataSearchPath) -> the compile-time last resort.
DistributionRoot resolveDistributionRoot(
    nonstd::span<const std::filesystem::path> explicitRoots);

std::vector<std::string> enumerateRenderableMaterials(
    const DocumentSource &source,
    nonstd::span<const std::filesystem::path> librarySearchPaths);

struct ParamMapping
{
  std::string cleanName;   // MaterialX input name (origin tail), e.g. "base_color"
  std::string originPath;  // full origin path, e.g. "srf/base_color" (getPath())
  std::string type;        // MaterialX input type: color3/color4/float/vector2..4
  std::string valueArg;    // generated constant MDL arg (was mdlArgName)
  std::string textureArg;  // generated texture MDL arg when spliced; else empty
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
    const DocumentSource &source,
    std::optional<std::string> selected,
    nonstd::span<const std::filesystem::path> librarySearchPaths,
    nonstd::span<const std::string> texturedOriginPaths);

} // namespace visrtx::materialx

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

#include "materialx/Transcoder.h"
#include <cstdio>
#include <filesystem>
#include <vector>

int main()
{
  const std::filesystem::path data =
      std::filesystem::path(MATERIALX_TEST_DATA_DIR) / "two_materials.mtlx";
  std::vector<std::filesystem::path> libs = {MATERIALX_LIBRARIES_DIR};

  auto names = visrtx::materialx::enumerateRenderableMaterials(data, libs);

  bool hasRed = false, hasBlue = false;
  for (const auto &n : names) {
    if (n.find("Mat_Red") != std::string::npos) hasRed = true;
    if (n.find("Mat_Blue") != std::string::npos) hasBlue = true;
  }
  if (names.size() != 2 || !hasRed || !hasBlue) {
    std::printf("FAIL: expected Mat_Red+Mat_Blue, got %zu names\n", names.size());
    return 1;
  }
  {
    const std::filesystem::path red =
        std::filesystem::path(MATERIALX_TEST_DATA_DIR) / "red_surface.mtlx";
    auto r = visrtx::materialx::transcodeMaterialXToMdl(red, std::nullopt, libs);
    if (!r.error.empty() || r.mdlSource.empty() || r.materialName.empty()) {
      std::printf("FAIL: red transcode error='%s'\n", r.error.c_str());
      return 1;
    }
    if (r.mdlSource.find("mdl 1.7") == std::string::npos) {
      std::printf("FAIL: generated MDL not pinned to 1.7\n");
      return 1;
    }
  }
  {
    auto r = visrtx::materialx::transcodeMaterialXToMdl(
        "/no/such/file.mtlx", std::nullopt, libs);
    if (r.error.empty() || !r.mdlSource.empty()) {
      std::printf("FAIL: missing-file should set error and empty source\n");
      return 1;
    }
  }
  {
    const std::filesystem::path red =
        std::filesystem::path(MATERIALX_TEST_DATA_DIR) / "red_surface.mtlx";
    auto r = visrtx::materialx::transcodeMaterialXToMdl(red, std::nullopt, libs);
    bool foundBaseColor = false;
    for (const auto &m : r.paramMap) {
      if (m.cleanName == "base_color") {
        foundBaseColor = true;
        // mdlArgName is the generated entry-material arg; for red_surface its
        // surface node is "srf", so the arg is "srf_base_color".
        if (m.mdlArgName != "srf_base_color") {
          std::printf("FAIL: base_color maps to '%s', expected 'srf_base_color'\n",
              m.mdlArgName.c_str());
          return 1;
        }
      }
    }
    if (!foundBaseColor) {
      std::printf("FAIL: paramMap missing base_color (%zu entries)\n",
          r.paramMap.size());
      return 1;
    }
  }
  std::printf("PASS\n");
  return 0;
}

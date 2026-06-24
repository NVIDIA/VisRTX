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

#include "MDL.h"
#include "materialx/Transcoder.h" // materialx::ParamMapping

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace visrtx {

struct MaterialX : public MDL
{
  MaterialX(DeviceGlobalState *d);

  void commitParameters() override;
  bool getProperty(const std::string_view &name, ANARIDataType type,
      void *ptr, uint64_t size, uint32_t flags) override;

 private:
  // True if the user's .mtlx path, selection, or the file's mtime changed since
  // the last successful transcode. Updates the cached user inputs as a side
  // effect. Must distinguish the user's `source`/`materialName` from the
  // generated MDL we write back into those same params (see commitParameters).
  bool needsRetranscode();

  // Rename any ANARI param stored under a clean MaterialX input name to its
  // generated MDL arg-block name before MDL::commitParameters() snapshots them.
  void remapParameters();

  std::string m_userPath;                    // last .mtlx path the app set
  std::optional<std::string> m_userSelected; // last materialName the app set
  std::filesystem::file_time_type m_userPathWrite{};

  std::string m_generatedSource; // MDL written into the `source` param
  std::string m_generatedName;   // name written into the `materialName` param
  std::vector<std::string> m_materialNames;
  std::vector<const char *> m_materialNamePtrs;
  std::vector<materialx::ParamMapping> m_paramMap; // clean name -> MDL arg name
};

} // namespace visrtx

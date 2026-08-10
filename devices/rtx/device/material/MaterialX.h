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
  ~MaterialX() override;

  void commitParameters() override;
  bool getProperty(const std::string_view &name, ANARIDataType type,
      void *ptr, uint64_t size, uint32_t flags) override;

 private:
  // True if the user's source, sourceType, selection, or (for documentFile) the
  // file's mtime changed since the last successful transcode. Updates the cached
  // user inputs as a side effect. The app's `source`/`sourceType`/`materialName`
  // params are never overwritten (the generated MDL travels via the
  // MDL::SourceHandoff — see commitParameters), so the live params read here are
  // authoritative user input; the "recognize our own write" branches remain only
  // as tolerance for apps staging 'code'/generated values directly.
  bool needsRetranscode();

  // Invoke the transcoder and update all generated state.
  void transcode(const std::vector<std::string> &texturedOrigins);

  // The textured-origin set the next transcode should target. Starts from the
  // persisted m_texturedOrigins so an absent (already-routed) clean param keeps
  // its current state; only a present clean param adds (sampler) or drops
  // (non-sampler value) its origin. Read BEFORE routing.
  std::vector<std::string> desiredTexturedOrigins() const;

  // Route each ParamMapping's clean param to its value or texture arg by type.
  void routeParameters(); // replaces remapParameters

  std::string m_userPath; // last source the app set: .mtlx path OR inline XML
  std::optional<std::string> m_userSelected; // last materialName the app set
  std::string m_userSourceType{"documentFile"}; // documentFile | documentInline
  std::filesystem::file_time_type m_userPathWrite{};

  // Distribution generation the last transcode ran against (see
  // DeviceGlobalState::MaterialXDistribution). A device commit that re-resolves
  // to a different root bumps it, forcing a retranscode; the sentinel makes the
  // first commit always mismatch.
  uint64_t m_distributionGeneration{~uint64_t(0)};

  std::string m_generatedSource; // MDL handed to the base via SourceHandoff
  std::string m_generatedName; // material name handed via SourceHandoff
  std::vector<std::string> m_materialNames;
  std::vector<const char *> m_materialNamePtrs;
  std::vector<std::string> m_textureInputNames;
  std::vector<const char *> m_textureInputPtrs;
  std::vector<materialx::ParamMapping> m_paramMap; // clean name -> MDL arg name

  // Origin paths of inputs currently bound to a sampler (persisted across
  // commits so a topology change is detected without recomputing from the
  // params that routing removes).
  std::vector<std::string> m_texturedOrigins;

  // MDL arg params set by the last routeParameters. Lets a topology change tear
  // down args the freshly compiled module no longer declares, so no orphan
  // param survives to mismatch the arg block (and leak its sampler).
  std::vector<std::string> m_routedArgs;
};

} // namespace visrtx

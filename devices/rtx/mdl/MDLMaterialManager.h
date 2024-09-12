/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "MDLSDK.h"
#include "mdl/MDLMaterialInfo.h"
#include "optix_visrtx.h"
#include "renderer/MaterialSbtData.cuh"
#include "utility/DeviceBuffer.h"

#include <anari/anari.h>
#include <anari/frontend/anari_enums.h>

#include <mi/base/uuid.h>

#include <stdint.h>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace visrtx {

struct MDLParamDesc
{
  enum class Semantic
  {
    Scalar, // Treat inputs as singular values
    Range, // The set of inputs is to be considered as a range ()
    Color,
  };
  std::string name;
  std::string displayName;
  ANARIDataType type;
  Semantic semantic;
};

class MDLMaterialManager
{
 public:
  MDLMaterialManager(MDLSDK *mdlSdk);
  ~MDLMaterialManager();

  using Uuid = mi::base::Uuid;
  using Id = uint64_t;
  static const constexpr auto INVALID_ID = std::numeric_limits<Id>::max();

  Uuid acquireModule(const char *modulePath);
  void releaseModule(Uuid moduleUuid);
  Id getModuleIndex(Uuid moduleUuid)
  {
    auto it = m_uuidToIndex.find(moduleUuid);
    if (it == m_uuidToIndex.cend()) {
      return INVALID_ID;
    }
    return it->second;
  }

  std::vector<MDLParamDesc> getModuleParameters(Uuid moduleUuid);

  // - getOrLoadModule(module) -> Hash
  // - releaseModule(moduleHash)
  // - getCompiledResult(moduleHash) -> CompiledResult

  std::vector<ptx_blob> getPTXBlobs();
  std::vector<MaterialSbtData> getMaterialSbtEntries();

 private:
  MDLSDK *m_mdlSdk;

  struct MaterialImplementation
  {
    MDLMaterialInfo materialInfo;
    std::vector<unsigned char> ptxBlob;
  };
  std::vector<MaterialImplementation> m_materialImplementations;

  struct UuidHasher
  {
    std::size_t operator()(const mi::base::Uuid &uuid) const noexcept
    {
      return mi::base::uuid_hash32(uuid);
    }
  };
  std::unordered_map<Uuid, uint64_t, UuidHasher> m_uuidToIndex;

  DeviceBuffer m_argblocksBuffer;
};

} // namespace visrtx

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

#include <helium/utility/AnariAny.h>
#include "Material.h"
#include "gpu/gpu_objects.h"
#include "mdl/MaterialRegistry.h"
#include "optix_visrtx.h"

#include "libmdl/ArgumentBlockInstance.h"
#include "sampler/Sampler.h"

#include <optional>
#include <unordered_map>

namespace visrtx {

struct MDL : public Material
{
  MDL(DeviceGlobalState *d);
  ~MDL() override;

  void commitParameters() override;
  void finalize() override;

  // Handle source changes
  void syncSource();
  // Update actual implementation index to use for the material.
  void syncImplementationIndex();
  // Handle argument block update
  void syncParameters();

 private:
  MaterialGPUData gpuData() const override;
  std::map<std::string, helium::AnariAny> m_parameterMap;

  void clearSamplers();

  DeviceBuffer m_argBlockBuffer;

  std::string m_source;
  std::string m_sourceType;
  struct SamplerDesc {
    Sampler* sampler = nullptr;
    std::string name;
    bool isFromRegistry = false;
    bool operator==(const SamplerDesc &other) const {
      return sampler == other.sampler && name == other.name &&
             isFromRegistry == other.isFromRegistry;
    }
  };
  std::vector<SamplerDesc> m_samplers;

  libmdl::Uuid m_uuid{};
  mdl::MaterialRegistry::ImplementationIndex m_implementationIndex{};
  std::optional<libmdl::ArgumentBlockInstance> m_argumentBlockInstance;
};

} // namespace visrtx

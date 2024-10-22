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

#include "MDL.h"
#include <cuda_runtime_api.h>

#include "gpu/gpu_objects.h"
#include "mdl/MDLCompiler.h"
#include "scene/surface/material/Material.h"

namespace visrtx {

MDL::MDL(DeviceGlobalState *d) : Material(d) {
}

MaterialGPUData MDL::gpuData() const
{
  auto self = const_cast<MDL*>(this);
  auto sourceType = self->getParamString("sourceType", "module");
  auto source = self->getParamString("source", "::visrtx::default::simple");
  
  auto mdlCompiler = MDLCompiler::getMDLCompiler(deviceState());

  if (sourceType == "module") {
    if (source != m_source) {
      self->m_implementationId = mdlCompiler->acquireModule(source.c_str());
      self->m_implementationIndex = mdlCompiler->getModuleIndex(m_implementationId);
      self->m_samplers = mdlCompiler->getModuleSamplers(m_implementationId);

      self->m_source = source;
    }
  } else if (sourceType == "code") {
    reportMessage(
        ANARI_SEVERITY_ERROR, "MDL::commit(): sourceType 'code' not supported");
  } else {
    reportMessage(ANARI_SEVERITY_ERROR,
        "MDL::commit(): sourceType must be either 'module' or 'code'");
  }

  MaterialGPUData retval;
  retval.materialType = MaterialType::MDL;
  retval.mdl.implementationId = m_implementationIndex;
  retval.mdl.numTextures = std::min(std::size(retval.mdl.samplers), size(m_samplers));

  using std::begin, std::end;

  std::fill(begin(retval.mdl.samplers), end(retval.mdl.samplers), 0);
  std::generate_n(begin(retval.mdl.samplers), retval.mdl.numTextures, [it= cbegin(m_samplers)]() mutable {
    auto idx = (*it)->index();
    ++it;
    return idx;
  });

  return retval;
}

void MDL::markCommitted()
{
  Object::markCommitted();
  deviceState()->objectUpdates.lastMDLMaterialChange = helium::newTimeStamp();
}

} // namespace visrtx

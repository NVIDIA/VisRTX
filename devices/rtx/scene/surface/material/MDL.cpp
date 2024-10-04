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

#include <anari/frontend/anari_enums.h>

#include "gpu/gpu_objects.h"
#include "mdl/MDLCompiler.h"
#include "scene/surface/material/Material.h"

namespace visrtx {

MDL::MDL(DeviceGlobalState *d) : Material(d) {
}

void MDL::commit()
{
  auto sourceType = getParamString("sourceType", "module");
  auto source = getParamString("source", "::visrtx::default::simple");
  
  auto mdlCompiler = MDLCompiler::getMDLCompiler(deviceState());

  if (sourceType == "module") {
    if (source != m_source) {
      m_implementationId = mdlCompiler->acquireModule(source.c_str());
      m_implementationIndex = mdlCompiler->getModuleIndex(m_implementationId);

      m_source = source;
    }
  } else if (sourceType == "code") {
    reportMessage(
        ANARI_SEVERITY_ERROR, "MDL::commit(): sourceType 'code' not supported");
  } else {
    reportMessage(ANARI_SEVERITY_ERROR,
        "MDL::commit(): sourceType must be either 'module' or 'code'");
  }

  upload();
}

MaterialGPUData MDL::gpuData() const
{
  MaterialGPUData retval;
  retval.materialType = MaterialType::MDL;
  retval.mdl.implementationId = m_implementationIndex;

  return retval;
}

} // namespace visrtx

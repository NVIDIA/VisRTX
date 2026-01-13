/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Debug.h"
// ptx
#include "Debug_ptx.h"

namespace visrtx {

static DebugMethod methodFromString(const std::string &name)
{
  if (name == "primitiveId")
    return DebugMethod::PRIM_ID;
  else if (name == "objectId")
    return DebugMethod::OBJ_ID;
  else if (name == "instanceId")
    return DebugMethod::INST_ID;
  else if (name == "primIndex")
    return DebugMethod::PRIM_INDEX;
  else if (name == "objIndex")
    return DebugMethod::OBJ_INDEX;
  else if (name == "instIndex")
    return DebugMethod::INST_INDEX;
  else if (name == "materialId")
    return DebugMethod::MAT_ID;
  else if (name == "Ng")
    return DebugMethod::NG;
  else if (name == "Ng.abs")
    return DebugMethod::NG_ABS;
  else if (name == "Ns")
    return DebugMethod::NS;
  else if (name == "Ns.abs")
    return DebugMethod::NS_ABS;
  else if (name == "tU")
    return DebugMethod::TANGENT_U;
  else if (name == "tU.handedness")
    return DebugMethod::TANGENT_HANDEDNESS;
  else if (name == "tV")
    return DebugMethod::TANGENT_V;
  else if (name == "uvw")
    return DebugMethod::RAY_UVW;
  else if (name == "backface")
    return DebugMethod::BACKFACE;
  else if (name == "istri")
    return DebugMethod::IS_TRIANGLE;
  else if (name == "isvol")
    return DebugMethod::IS_VOLUME;
  else if (name == "hasMaterial")
    return DebugMethod::HAS_MATERIAL;
  else if (name == "geometry.attribute0")
    return DebugMethod::GEOMETRY_ATTRIBUTE_0;
  else if (name == "geometry.attribute1")
    return DebugMethod::GEOMETRY_ATTRIBUTE_1;
  else if (name == "geometry.attribute2")
    return DebugMethod::GEOMETRY_ATTRIBUTE_2;
  else if (name == "geometry.attribute3")
    return DebugMethod::GEOMETRY_ATTRIBUTE_3;
  else if (name == "geometry.color")
    return DebugMethod::GEOMETRY_ATTRIBUTE_COLOR;
  else
    return DebugMethod::PRIM_ID; // match default value
}

Debug::Debug(DeviceGlobalState *s) : Renderer(s) {}

void Debug::commitParameters()
{
  Renderer::commitParameters();
  m_method = methodFromString(getParamString("method", "primID"));
  m_sampleLimit = 1; // single-shot renderer
  m_denoise = false; // never denoise
}

void Debug::populateFrameData(FrameGPUData &fd) const
{
  Renderer::populateFrameData(fd);
  fd.renderer.params.debug.method = static_cast<int>(m_method);
}

OptixModule Debug::optixModule() const
{
  return deviceState()->rendererModules.debug;
}

ptx_blob Debug::ptx()
{
  return {Debug_ptx, sizeof(Debug_ptx)};
}

} // namespace visrtx

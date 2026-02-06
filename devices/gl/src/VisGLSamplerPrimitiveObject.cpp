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

#include "VisGLSpecializations.h"
#include "shader_blocks.h"
#include "anari2gl_types.h"

namespace visgl {

Object<SamplerPrimitive>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{}

void Object<SamplerPrimitive>::commit()
{
  DefaultObject::commit();
  array = acquire<DataArray1D *>(current.array);
}

void Object<SamplerPrimitive>::update()
{
  DefaultObject::update();
  uint64_t offset = 0;
  current.inOffset.get(ANARI_UINT64, &offset);

  meta[0] = offset;
}

void Object<SamplerPrimitive>::allocateResources(
    SurfaceObjectBase *surf, int slot)
{
  if (array) {
    surf->allocateStorageBuffer(slot, array->getBuffer());
  }
  surf->addAttributeFlags(ATTRIBUTE_PRIMITIVE_ID, ATTRIBUTE_FLAG_USED);
}

void Object<SamplerPrimitive>::drawCommand(int index, DrawCommand &command)
{
  if (array) {
    // these sevens are magic numbers for now
    array->drawCommand(index, command);
  }
}

void Object<SamplerPrimitive>::declare(int index, AppendableShader &shader)
{
  if (array) {
    array->declare(index, shader);
  }
}

void Object<SamplerPrimitive>::sample(
    int index, AppendableShader &shader, const char *meta)
{
  if (array) {
    array->sample(index, shader);
    shader.append("primitiveId+floatBitsToUint(");
    shader.append(meta);
    shader.append(".x));\n");
  } else {
    shader.append("vec4(1,0,1,1);\n");
  }
}

std::array<uint32_t, 4> Object<SamplerPrimitive>::metadata()
{
  return meta;
}

Object<SamplerPrimitive>::~Object() {}

} // namespace visgl

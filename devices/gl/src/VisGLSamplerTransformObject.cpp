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

Object<SamplerTransform>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  transform_index = thisDevice->transforms.allocate(1);
}

void Object<SamplerTransform>::commit()
{
  DefaultObject::commit();
}

void Object<SamplerTransform>::update()
{
  DefaultObject::update();
  std::array<float, 16> transform;

  if (current.outTransform.get(ANARI_FLOAT32_MAT4, transform.data())) {
    thisDevice->transforms.set(transform_index, transform);
  }
}

void Object<SamplerTransform>::allocateResources(
    SurfaceObjectBase *surf, int slot)
{
  surf->allocateTransform(slot);
  surf->addAttributeFlags(
      attribIndex(current.inAttribute.getStringEnum()), ATTRIBUTE_FLAG_USED);
}

void Object<SamplerTransform>::drawCommand(int index, DrawCommand &command) {}

#define DECLARE_SAMPLER(I)                                                     \
  "vec4 sampleTransform" #I                                                    \
  "(vec4 attrib, vec4 meta) {\n"                                               \
  "  uint idx = floatBitsToUint(meta.x);\n"                                    \
  "  mat4 transform = transforms[idx];\n"                                      \
  "  return transform*attrib;\n"                                               \
  "}\n"

static const char *declareSampler1D[] = {
    DECLARE_SAMPLER(0),
    DECLARE_SAMPLER(1),
    DECLARE_SAMPLER(2),
    DECLARE_SAMPLER(3),
    DECLARE_SAMPLER(4),
    DECLARE_SAMPLER(5),
    DECLARE_SAMPLER(6),
    DECLARE_SAMPLER(7),
    DECLARE_SAMPLER(8),
    DECLARE_SAMPLER(9),
    DECLARE_SAMPLER(10),
    DECLARE_SAMPLER(11),
    DECLARE_SAMPLER(12),
    DECLARE_SAMPLER(13),
    DECLARE_SAMPLER(14),
    DECLARE_SAMPLER(15),
};

void Object<SamplerTransform>::declare(int index, AppendableShader &shader)
{
  shader.append(declareSampler1D[index]);
}

static const char *textureSample[] = {
    "sampleTransform0(",
    "sampleTransform1(",
    "sampleTransform2(",
    "sampleTransform3(",
    "sampleTransform4(",
    "sampleTransform5(",
    "sampleTransform6(",
    "sampleTransform7(",
    "sampleTransform8(",
    "sampleTransform9(",
    "sampleTransform10(",
    "sampleTransform11(",
    "sampleTransform12(",
    "sampleTransform13(",
    "sampleTransform14(",
    "sampleTransform15(",
};

void Object<SamplerTransform>::sample(
    int index, AppendableShader &shader, const char *meta)
{
  shader.append(textureSample[index]);
  shader.append(current.inAttribute.getString());
  shader.append(", ");
  shader.append(meta);
  shader.append(");\n");
}

std::array<uint32_t, 4> Object<SamplerTransform>::metadata()
{
  return std::array<uint32_t, 4>{uint32_t(transform_index), 0, 0, 0};
}

Object<SamplerTransform>::~Object() {}

} // namespace visgl

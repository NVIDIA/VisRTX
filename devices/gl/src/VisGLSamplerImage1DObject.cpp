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

Object<SamplerImage1D>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  transform_index = thisDevice->transforms.allocate(3);
}

void image1d_init_objects(
    ObjectRef<SamplerImage1D> samplerObj, int filter, GLenum wrapS)
{
  auto &gl = samplerObj->thisDevice->gl;
  if (samplerObj->sampler == 0) {
    gl.GenSamplers(1, &samplerObj->sampler);
  }
  gl.SamplerParameteri(
      samplerObj->sampler, GL_TEXTURE_MAG_FILTER, gl_mag_filter(filter));
  gl.SamplerParameteri(
      samplerObj->sampler, GL_TEXTURE_MIN_FILTER, gl_min_filter(filter));
  gl.SamplerParameteri(samplerObj->sampler, GL_TEXTURE_WRAP_S, wrapS);
  if (gl.ES_VERSION_3_2) {
    gl.SamplerParameteri(
        samplerObj->sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }
}

void Object<SamplerImage1D>::commit()
{
  DefaultObject::commit();
  image = acquire<DataArray1D *>(current.image);
}

void Object<SamplerImage1D>::update()
{
  DefaultObject::update();
  std::array<float, 16> inTransform;
  std::array<float, 16> outTransform;
  std::array<float, 16> offsets;

  if (current.inTransform.get(ANARI_FLOAT32_MAT4, inTransform.data())) {
    thisDevice->transforms.set(transform_index, inTransform);
  }

  if (current.outTransform.get(ANARI_FLOAT32_MAT4, outTransform.data())) {
    thisDevice->transforms.set(transform_index + 1, outTransform);
  }

  current.inOffset.get(ANARI_FLOAT32_VEC4, offsets.data());
  current.outOffset.get(ANARI_FLOAT32_VEC4, offsets.data()+4);
  thisDevice->transforms.set(transform_index+2, offsets);

  int filter = current.filter.getStringEnum();
  GLenum wrapS = gl_wrap(current.wrapMode1.getStringEnum());

  thisDevice->queue.enqueue(image1d_init_objects, this, filter, wrapS);
}

void Object<SamplerImage1D>::allocateResources(
    SurfaceObjectBase *surf, int slot)
{
  GLenum target;
  if (thisDevice->gl.ES_VERSION_3_2) {
    target = GL_TEXTURE_2D;
  } else {
    target = GL_TEXTURE_1D;
  }
  if (image) {
    surf->allocateTexture(slot, target, image->getTexture1D(), sampler);
  }
  surf->addAttributeFlags(attribIndex(current.inAttribute.getStringEnum()),
      ATTRIBUTE_FLAG_USED | ATTRIBUTE_FLAG_SAMPLED);
}

void Object<SamplerImage1D>::drawCommand(int index, DrawCommand &command)
{
  if (image) {
    auto &tex = command.textures[command.texcount];
    tex.index = index;
    if (thisDevice->gl.ES_VERSION_3_2) {
      tex.target = GL_TEXTURE_2D;
    } else {
      tex.target = GL_TEXTURE_1D;
    }
    tex.texture = image->getTexture1D();
    tex.sampler = sampler;
    command.texcount += 1;
  }
}
// clang-format off

#define DECLARE_SAMPLER(I)\
  "layout(binding = " #I ") uniform sampler1D sampler" #I ";\n"\
  "vec4 sampleImage1D" #I "(vec4 attrib, vec4 meta) {\n"\
  "  uint idx = floatBitsToUint(meta.x);\n"\
  "  mat4 inTransform = transforms[idx];\n"\
  "  mat4 outTransform = transforms[idx+1u];\n"\
  "  mat4 offsets = transforms[idx+2u];\n"\
  "  return outTransform*texture(sampler" #I ", (inTransform*attrib+offsets[0]).x)+offsets[1];\n"\
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

#define DECLARE_SAMPLER_GLES(I)\
  "layout(binding = " #I ") uniform sampler2D sampler" #I ";\n"\
  "vec4 sampleImage1D" #I "(vec4 attrib, vec4 meta) {\n"\
  "  uint idx = floatBitsToUint(meta.x);\n"\
  "  mat4 inTransform = transforms[idx];\n"\
  "  mat4 outTransform = transforms[idx+1u];\n"\
  "  return outTransform*texture(sampler" #I ", vec2((inTransform*attrib).x, 0.0));\n"\
  "}\n"

static const char *declareSampler1D_gles[] = {
  DECLARE_SAMPLER_GLES(0),
  DECLARE_SAMPLER_GLES(1),
  DECLARE_SAMPLER_GLES(2),
  DECLARE_SAMPLER_GLES(3),
  DECLARE_SAMPLER_GLES(4),
  DECLARE_SAMPLER_GLES(5),
  DECLARE_SAMPLER_GLES(6),
  DECLARE_SAMPLER_GLES(7),
  DECLARE_SAMPLER_GLES(8),
  DECLARE_SAMPLER_GLES(9),
  DECLARE_SAMPLER_GLES(10),
  DECLARE_SAMPLER_GLES(11),
  DECLARE_SAMPLER_GLES(12),
  DECLARE_SAMPLER_GLES(13),
  DECLARE_SAMPLER_GLES(14),
  DECLARE_SAMPLER_GLES(15),
};
// clang-format on

void Object<SamplerImage1D>::declare(int index, AppendableShader &shader)
{
  if (thisDevice->gl.ES_VERSION_3_2) {
    shader.append(declareSampler1D_gles[index]);
  } else {
    shader.append(declareSampler1D[index]);
  }
}

static const char *textureSample[] = {
    "sampleImage1D0(",
    "sampleImage1D1(",
    "sampleImage1D2(",
    "sampleImage1D3(",
    "sampleImage1D4(",
    "sampleImage1D5(",
    "sampleImage1D6(",
    "sampleImage1D7(",
    "sampleImage1D8(",
    "sampleImage1D9(",
    "sampleImage1D10(",
    "sampleImage1D11(",
    "sampleImage1D12(",
    "sampleImage1D13(",
    "sampleImage1D14(",
    "sampleImage1D15(",
};

void Object<SamplerImage1D>::sample(
    int index, AppendableShader &shader, const char *meta)
{
  shader.append(textureSample[index]);
  shader.append(current.inAttribute.getString());
  shader.append(", ");
  shader.append(meta);
  shader.append(");\n");
}

std::array<uint32_t, 4> Object<SamplerImage1D>::metadata()
{
  return std::array<uint32_t, 4>{uint32_t(transform_index), 0, 0, 0};
}

static void image2d_delete_objects(Object<Device> *deviceObj, GLuint sampler)
{
  deviceObj->gl.DeleteSamplers(1, &sampler);
}

Object<SamplerImage1D>::~Object()
{
  thisDevice->queue.enqueue(image2d_delete_objects, thisDevice, sampler);
}

} // namespace visgl

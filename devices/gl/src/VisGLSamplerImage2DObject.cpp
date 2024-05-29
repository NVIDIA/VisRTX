/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Object<SamplerImage2D>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  transform_index = thisDevice->transforms.allocate(3);
}

void image2d_init_objects(ObjectRef<SamplerImage2D> samplerObj,
    int filter,
    GLenum wrapS,
    GLenum wrapT)
{
  auto &gl = samplerObj->thisDevice->gl;
  if (samplerObj->sampler == 0) {
    gl.GenSamplers(1, &samplerObj->sampler);
  }
  if (samplerObj->image
      && gl_mipmappable(
          gl_internal_format(samplerObj->image->getElementType()))) {
    gl.SamplerParameteri(
        samplerObj->sampler, GL_TEXTURE_MAG_FILTER, gl_mag_filter(filter));
    gl.SamplerParameteri(
        samplerObj->sampler, GL_TEXTURE_MIN_FILTER, gl_min_filter_mip(filter));
    if (gl.EXT_texture_filter_anisotropic) {
      float anisotropy = 2.0f;
      gl.GetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &anisotropy);
      gl.SamplerParameterf(
          samplerObj->sampler, GL_TEXTURE_MAX_ANISOTROPY, anisotropy);
    }
  } else {
    gl.SamplerParameteri(
        samplerObj->sampler, GL_TEXTURE_MAG_FILTER, gl_mag_filter(filter));
    gl.SamplerParameteri(
        samplerObj->sampler, GL_TEXTURE_MIN_FILTER, gl_min_filter(filter));
  }
  gl.SamplerParameteri(samplerObj->sampler, GL_TEXTURE_WRAP_S, wrapS);
  gl.SamplerParameteri(samplerObj->sampler, GL_TEXTURE_WRAP_T, wrapT);
}

void Object<SamplerImage2D>::commit()
{
  DefaultObject::commit();
  image = acquire<Object<Array2D> *>(current.image);
}

void Object<SamplerImage2D>::update()
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
  GLenum wrapT = gl_wrap(current.wrapMode2.getStringEnum());

  thisDevice->queue.enqueue(image2d_init_objects, this, filter, wrapS, wrapT);
}

void Object<SamplerImage2D>::allocateResources(
    SurfaceObjectBase *surf, int slot)
{
  if (image) {
    surf->allocateTexture(slot, GL_TEXTURE_2D, image->getTexture2D(), sampler);
  }
  surf->addAttributeFlags(attribIndex(current.inAttribute.getStringEnum()),
      ATTRIBUTE_FLAG_USED | ATTRIBUTE_FLAG_SAMPLED);
}

void Object<SamplerImage2D>::drawCommand(int index, DrawCommand &command)
{
  if (image) {
    auto &tex = command.textures[command.texcount];
    tex.index = index;
    tex.target = GL_TEXTURE_2D;
    tex.texture = image->getTexture2D();
    tex.sampler = sampler;
    command.texcount += 1;
  }
}

// clang-format off
#define DECLARE_SAMPLER(I)\
  "layout(binding = " #I ") uniform sampler2D sampler" #I ";\n"\
  "vec4 sampleImage2D" #I "(vec4 attrib, vec4 meta) {\n"\
  "  uint idx = floatBitsToUint(meta.x);\n"\
  "  mat4 inTransform = transforms[idx];\n"\
  "  mat4 outTransform = transforms[idx+1u];\n"\
  "  mat4 offsets = transforms[idx+2u];\n"\
  "  return outTransform*texture(sampler" #I ", (inTransform*attrib+offsets[0]).xy)+offsets[1];\n"\
  "}\n"

static const char *declareSampler2D[] = {
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
// clang-format on

void Object<SamplerImage2D>::declare(int index, AppendableShader &shader)
{
  shader.append(declareSampler2D[index]);
}

static const char *textureSample[] = {
    "sampleImage2D0(",
    "sampleImage2D1(",
    "sampleImage2D2(",
    "sampleImage2D3(",
    "sampleImage2D4(",
    "sampleImage2D5(",
    "sampleImage2D6(",
    "sampleImage2D7(",
    "sampleImage2D8(",
    "sampleImage2D9(",
    "sampleImage2D10(",
    "sampleImage2D11(",
    "sampleImage2D12(",
    "sampleImage2D13(",
    "sampleImage2D14(",
    "sampleImage2D15(",
};

void Object<SamplerImage2D>::sample(
    int index, AppendableShader &shader, const char *meta)
{
  shader.append(textureSample[index]);
  shader.append(current.inAttribute.getString());
  shader.append(", ");
  shader.append(meta);
  shader.append(");\n");
}

std::array<uint32_t, 4> Object<SamplerImage2D>::metadata()
{
  return std::array<uint32_t, 4>{uint32_t(transform_index), 0, 0, 0};
}

static void image2d_delete_objects(Object<Device> *deviceObj, GLuint sampler)
{
  deviceObj->gl.DeleteSamplers(1, &sampler);
}

Object<SamplerImage2D>::~Object()
{
  thisDevice->queue.enqueue(image2d_delete_objects, thisDevice, sampler);
}

} // namespace visgl

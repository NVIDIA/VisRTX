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


#include "anari2gl_types.h"
#include "AppendableShader.h"
#include "shader_compile_segmented.h"
#include "shader_blocks.h"

#include <cstdlib>
#include <cstring>

namespace visgl {

#define LUT_RESOLUTION 1024

Object<VolumeTransferFunction1D>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle), lutData(LUT_RESOLUTION)
{
  material_index = thisDevice->materials.allocate(1);
}

template <typename A, typename B>
static bool compare_and_assign(A &a, const B &b)
{
  bool cmp = (a == b);
  a = b;
  return cmp;
}

void Object<VolumeTransferFunction1D>::commit()
{
  DefaultObject::commit();

  dirty |= compare_and_assign(
      field, acquire<SpatialFieldObjectBase *>(current.value));
  dirty |= compare_and_assign(color, acquire<DataArray1D *>(current.color));
  dirty |= compare_and_assign(opacity, acquire<DataArray1D *>(current.opacity));
}

const char *transfer_sampler1d = R"GLSL(
layout(binding = 1) uniform highp sampler1D transferSampler;
vec4 transferSample(vec4 coord) {
  vec4 meta = materials[instanceIndices.y];
  float s = (coord.x-meta.x)/(meta.y-meta.x);
  vec4 c = texture(transferSampler, s);
  c.w *= meta.z;
  return c;
}
)GLSL";

const char *transfer_sampler2d = R"GLSL(
layout(binding = 1) uniform highp sampler2D transferSampler;
vec4 transferSample(vec4 coord) {
  vec4 meta = materials[instanceIndices.y];
  float s = (coord.x-meta.x)/(meta.y-meta.x);
  vec4 c = texture(transferSampler, vec2(s, 0));
  c.w *= meta.z;
  return c;
}
)GLSL";

void scivis_init_objects(ObjectRef<VolumeTransferFunction1D> scivisObj)
{
  auto &gl = scivisObj->thisDevice->gl;

  if (scivisObj->lut == 0) {
    gl.GenTextures(1, &scivisObj->lut);
  }

  if (gl.ES_VERSION_3_2) {
    gl.BindTexture(GL_TEXTURE_2D, scivisObj->lut);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1);
    gl.TexImage2D(GL_TEXTURE_2D,
        0,
        GL_RGBA32F,
        LUT_RESOLUTION,
        1,
        0,
        GL_RGBA,
        GL_FLOAT,
        scivisObj->lutData.data());
  } else {
    gl.BindTexture(GL_TEXTURE_1D, scivisObj->lut);
    gl.TexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl.TexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl.TexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1);
    gl.TexImage1D(GL_TEXTURE_1D,
        0,
        GL_RGBA32F,
        LUT_RESOLUTION,
        0,
        GL_RGBA,
        GL_FLOAT,
        scivisObj->lutData.data());
  }
}

static void volume_compile_shader(ObjectRef<VolumeTransferFunction1D> scivisObj,
    StaticAppendableShader<SHADER_SEGMENTS> vs,
    StaticAppendableShader<SHADER_SEGMENTS> fs)
{
  if (scivisObj->shader == 0) {
    scivisObj->shader = scivisObj->thisDevice->shaders.get(vs, fs);
  }
}

void Object<VolumeTransferFunction1D>::update()
{
  DefaultObject::update();

  std::array<float, 4> data;
  current.valueRange.get(ANARI_FLOAT32_BOX1, data.data());
  current.unitDistance.get(ANARI_FLOAT32, data.data() + 2);
  thisDevice->materials.set(material_index, data);

  if (dirty) {
    if (color) {
      uint64_t N = color->size();
      for (uint64_t i = 0; i < N - 1; ++i) {
        uint64_t begin = i * LUT_RESOLUTION / (N - 1);
        uint64_t end = (i + 1u) * LUT_RESOLUTION / (N - 1);
        std::array<float, 4> a = color->at(i);
        std::array<float, 4> b = color->at(i + 1u);
        for (uint64_t j = begin; j < end; ++j) {
          float s = float(j - begin) / float(end - begin);
          lutData[j][0] = a[0] + s * (b[0] - a[0]);
          lutData[j][1] = a[1] + s * (b[1] - a[1]);
          lutData[j][2] = a[2] + s * (b[2] - a[2]);
          lutData[j][3] = a[3] + s * (b[3] - a[3]);
        }
      }
    }

    if (opacity) {
      uint64_t N = opacity->size();
      for (uint64_t i = 0; i < N - 1; ++i) {
        uint64_t begin = i * LUT_RESOLUTION / (N - 1);
        uint64_t end = (i + 1u) * LUT_RESOLUTION / (N - 1);
        std::array<float, 4> a = opacity->at(i);
        std::array<float, 4> b = opacity->at(i + 1u);
        for (uint64_t j = begin; j < end; ++j) {
          float s = float(j - begin) / float(end - begin);
          lutData[j][3] = a[0] + s * (b[0] - a[0]);
        }
      }
    }

    thisDevice->queue.enqueue(scivis_init_objects, this);
  }

  if (shader == 0 && field) {
    StaticAppendableShader<SHADER_SEGMENTS> vs;
    if (thisDevice->gl.VERSION_4_3) {
      vs.append(version_430);
    } else {
      vs.append(version_320_es);
    }
    vs.append(shader_preamble);
    field->vertexShaderMain(this, vs);

    StaticAppendableShader<SHADER_SEGMENTS> fs;
    if (thisDevice->gl.VERSION_4_3) {
      fs.append(version_430);
    } else {
      fs.append(version_320_es);
    }
    fs.append(shader_preamble);
    fs.append(shader_conversions);
    if (thisDevice->gl.ES_VERSION_3_2) {
      fs.append(transfer_sampler2d);
    } else {
      fs.append(transfer_sampler1d);
    }
    field->fragmentShaderMain(this, fs);

    thisDevice->queue.enqueue(volume_compile_shader, this, vs, fs).wait();
  }

  dirty = false;
}

void Object<VolumeTransferFunction1D>::drawCommand(DrawCommand &command)
{
  command.shader = shader;

  auto &tex = command.textures[command.texcount];
  tex.index = 1;
  tex.target = thisDevice->gl.ES_VERSION_3_2 ? GL_TEXTURE_2D : GL_TEXTURE_1D;
  tex.texture = lut;
  tex.sampler = 0;
  command.texcount += 1;
}

uint32_t Object<VolumeTransferFunction1D>::index()
{
  return material_index;
}

static void scivis_delete_objects(Object<Device> *deviceObj, GLuint lut)
{
  auto &gl = deviceObj->gl;
  gl.DeleteTextures(1, &lut);
};

Object<VolumeTransferFunction1D>::~Object()
{
  thisDevice->queue.enqueue(scivis_delete_objects, thisDevice, lut);
}

} // namespace visgl

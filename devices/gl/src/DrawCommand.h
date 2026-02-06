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

#include "ogl.h"

namespace visgl {

struct DrawCommand
{
  static const int maxTextures = 10;
  static const int maxSSBOs = 10;

  struct TextureInfo
  {
    int index;
    GLenum target;
    GLuint texture;
    GLuint sampler;
  };

  struct SSBOInfo
  {
    int index;
    GLuint buffer;
  };

  // binding setup
  GLuint shader = 0;
  GLuint shadow_shader = 0;
  GLuint occlusion_resolve_shader = 0;
  GLuint vao = 0;
  GLuint occlusion_resolve_vao = 0;

  GLuint texcount = 0;
  TextureInfo textures[maxTextures];

  GLuint ssbocount = 0;
  SSBOInfo ssbos[maxSSBOs];

  GLuint uniform[4];

  // draw parameters
  GLenum prim;
  GLuint count;
  GLuint instanceCount;
  GLenum indexType;

  GLenum cullMode = GL_NONE;

  uint32_t vertex_count = 0;

  template <typename G>
  void operator()(G &gl, int mode)
  {
    GLuint current_shader = 0;
    GLuint current_vao = vao;
    if (mode == 0) {
      current_shader = shader;
    } else if (mode == 1) {
      current_shader = shadow_shader;
    } else if (mode == 2) {
      current_shader = occlusion_resolve_shader;
      current_vao = occlusion_resolve_vao;
    }

    if (!current_vao || !current_shader) {
      return;
    }

    gl.UseProgram(current_shader);
    gl.BindVertexArray(current_vao);
    gl.Uniform4uiv(0, 1, uniform);
    for (int i = 0; i < texcount; ++i) {
      if (textures[i].texture) {
        gl.ActiveTexture(GL_TEXTURE0 + textures[i].index);
        gl.BindTexture(textures[i].target, textures[i].texture);
        if (textures[i].sampler) {
          gl.BindSampler(textures[i].index, textures[i].sampler);
        }
      }
    }
    for (int i = 0; i < ssbocount; ++i) {
      if (ssbos[i].buffer) {
        gl.BindBufferBase(
            GL_SHADER_STORAGE_BUFFER, ssbos[i].index, ssbos[i].buffer);
      }
    }
    if (cullMode) {
      gl.Enable(GL_CULL_FACE);
      gl.CullFace(cullMode);
    } else {
      gl.Disable(GL_CULL_FACE);
    }
    if (mode == 2) {
      gl.DrawArrays(GL_POINTS, 0, vertex_count);
    } else {
      if (indexType) {
        gl.DrawElementsInstanced(prim, count, indexType, 0, instanceCount);
      } else {
        gl.DrawArraysInstanced(prim, 0, count, instanceCount);
      }
    }
  }
};

} // namespace visgl

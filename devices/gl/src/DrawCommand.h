// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "glad/gl.h"

namespace visgl{

struct DrawCommand {
  static const int maxTextures = 10;
  static const int maxSSBOs = 10;

  struct TextureInfo {
    int index;
    GLenum target;
    GLuint texture;
    GLuint sampler;
  };

  struct SSBOInfo {
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

  template<typename G>
  void operator()(G &gl, int mode) {
    GLuint current_shader = 0;
    GLuint current_vao = vao;
    if(mode == 0) {
      current_shader = shader;
    } else if(mode == 1) {
      current_shader = shadow_shader;
    } else if(mode == 2) {
      current_shader = occlusion_resolve_shader;
      current_vao = occlusion_resolve_vao;
    }

    if(!current_vao || !current_shader) {
      return;
    }

    gl.UseProgram(current_shader);
    gl.BindVertexArray(current_vao);
    gl.Uniform4uiv(0, 1, uniform);
    for(int i = 0;i<texcount;++i) {
      if(textures[i].texture) {
        gl.ActiveTexture(GL_TEXTURE0+textures[i].index);
        gl.BindTexture(textures[i].target, textures[i].texture);
        if(textures[i].sampler) {
          gl.BindSampler(textures[i].index, textures[i].sampler);
        }
      }
    }
    for(int i = 0;i<ssbocount;++i) {
      if(ssbos[i].buffer) {
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbos[i].index, ssbos[i].buffer);
      }
    }
    if(cullMode) {
      gl.Enable(GL_CULL_FACE);
      gl.CullFace(cullMode);
    } else {
      gl.Disable(GL_CULL_FACE);
    }
    if(mode == 2) {
      gl.DrawArrays(GL_POINTS, 0, vertex_count);
    } else {
      if(indexType) {
        gl.DrawElementsInstanced(prim, count, indexType, 0, instanceCount);
      } else {
        gl.DrawArraysInstanced(prim, 0, count, instanceCount);
      }
    }
  }
};


} //namespace visgl


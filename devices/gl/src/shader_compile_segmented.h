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

#include <cstdio>
#include <stdint.h>
#include <vector>

static uint32_t FNV1a(const char *str)
{
  uint32_t hash = 0x811c9dc5u;
  for (int i = 0; str[i] != 0; ++i) {
    hash ^= (uint32_t)str[i];
    hash *= 0x01000193u;
  }
  return hash;
}

static uint32_t FNV1a_seg(const char *const *arr)
{
  uint32_t hash = 0x811c9dc5u;
  for (int j = 0; arr[j] != 0; ++j) {
    const char *str = arr[j];
    for (int i = 0; str[i] != 0; ++i) {
      hash ^= (uint32_t)str[i];
      hash *= 0x01000193u;
    }
  }
  return hash;
}

static int streq_seg(const char *const *a, const char *const *b)
{
  // capture some trivial cases
  if (a == b) { // obviously
    return 1;
  }
  // check if the segments are equal
  for (int i = 0; a[i] == b[i]; ++i) {
    if (a[i] == 0 && b[i] == 0) {
      return 1;
    } // if only one of them is 0 the loop condition will handle it
  }
  // do full comparison
  int i = 0;
  int j = 0;
  for (;;) {
    if (a == 0 && b == 0) {
      return 1;
    } else if (a == 0 || b == 0) {
      return 0;
    } else if ((*a)[i] != (*b)[j]) {
      return 0;
    }

    i += 1;
    j += 1;
    if ((*a)[i] == 0) {
      a += 1;
      i = 0;
    }
    if ((*b)[j] == 0) {
      b += 1;
      j = 0;
    }
  }
}

static const char *shader_name(GLenum type)
{
  switch (type) {
  case GL_VERTEX_SHADER: return "vertex";
  case GL_TESS_CONTROL_SHADER: return "tess control";
  case GL_TESS_EVALUATION_SHADER: return "tess evaluation";
  case GL_GEOMETRY_SHADER: return "geometry";
  case GL_FRAGMENT_SHADER: return "fragment";
  case GL_COMPUTE_SHADER: return "compute";
  default: return "unkonwn";
  }
}

template <typename T>
static GLuint link_shader(T &gl, GLuint *shaders, int N)
{
  GLint status;
  GLuint shader_program = gl.CreateProgram();
  for (int i = 0; i < N; ++i) {
    gl.AttachShader(shader_program, shaders[i]);
  }
  gl.LinkProgram(shader_program);
  gl.GetProgramiv(shader_program, GL_LINK_STATUS, &status);
  if (status == GL_FALSE) {
    GLint length;
    gl.GetProgramiv(shader_program, GL_INFO_LOG_LENGTH, &length);
    std::vector<char> log(length);
    gl.GetProgramInfoLog(shader_program, length, &length, log.data());
    std::fprintf(stderr, "[GLSL] link error: %s", log.data());
    return 0;
  }

  return shader_program;
}

inline static void print_source(const char *const *source)
{
  if (source == NULL) {
    return;
  }
  int linenumber = 1;
  std::fprintf(stderr, "%3d: ", linenumber++);
  for (int i = 0; source[i] != 0; ++i) {
    const char *line = source[i];
    const char *next = std::strchr(source[i], '\n');
    while (next) {
      int length = next - line + 1;
      std::fprintf(stderr, "%.*s%3d: ", length, line, linenumber++);
      line = next + 1;
      next = std::strchr(line, '\n');
    }
    std::fprintf(stderr, "%s", line);
  }
  std::fprintf(stderr, "\n");
}

template <typename T>
inline static GLuint compile_shader_segmented(
    T &gl, GLenum type, const char *const *source)
{
  GLint status;
  GLuint shader = gl.CreateShader(type);
  GLsizei N = 0;
  while (source[N] != nullptr) {
    N += 1;
  }
  gl.ShaderSource(shader, N, source, 0);
  gl.CompileShader(shader);
  gl.GetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (status == GL_FALSE) {
    GLint length;
    gl.GetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    std::vector<char> log(length);
    gl.GetShaderInfoLog(shader, length, &length, log.data());
    std::fprintf(stderr,
        "[GLSL] %s shader compilation error %s\n",
        shader_name(type),
        log.data());

    print_source(source);

    return 0;
  }

  return shader;
}

template <typename T>
inline static GLuint shader_build_graphics_segmented(T &gl,
    const char *const *vertex_source,
    const char *const *control_source,
    const char *const *eval_source,
    const char *const *geometry_source,
    const char *const *fragment_source)
{
  GLuint shaders[5];
  int count = 0;
  if (vertex_source != nullptr)
    shaders[count++] =
        compile_shader_segmented(gl, GL_VERTEX_SHADER, vertex_source);
  if (control_source != nullptr)
    shaders[count++] =
        compile_shader_segmented(gl, GL_TESS_CONTROL_SHADER, control_source);
  if (eval_source != nullptr)
    shaders[count++] =
        compile_shader_segmented(gl, GL_TESS_EVALUATION_SHADER, eval_source);
  if (geometry_source != nullptr)
    shaders[count++] =
        compile_shader_segmented(gl, GL_GEOMETRY_SHADER, geometry_source);
  if (fragment_source != nullptr)
    shaders[count++] =
        compile_shader_segmented(gl, GL_FRAGMENT_SHADER, fragment_source);

  GLuint program = link_shader(gl, shaders, count);
  if (program == 0) {
    print_source(vertex_source);
    print_source(control_source);
    print_source(eval_source);
    print_source(geometry_source);
    print_source(fragment_source);
  }
  for (int i = 0; i < count; ++i) {
    gl.DeleteShader(shaders[i]);
  }
  return program;
}

template <typename T>
inline static GLuint shader_build_compute_segmented(
    T &gl, const char *const *compute_source)
{
  GLuint shaders[1];
  int count = 0;
  if (compute_source != nullptr)
    shaders[count++] =
        compile_shader_segmented(gl, GL_COMPUTE_SHADER, compute_source);

  GLuint program = link_shader(gl, shaders, count);
  for (int i = 0; i < count; ++i) {
    gl.DeleteShader(shaders[i]);
  }
  return program;
}

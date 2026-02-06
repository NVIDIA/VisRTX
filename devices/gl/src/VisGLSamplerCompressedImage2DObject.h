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

#include "VisGLDevice.h"
#include "AppendableShader.h"

#include <array>

namespace visgl {

template <>
class Object<SamplerCompressedImage2D>
    : public DefaultObject<SamplerCompressedImage2D, SamplerObjectBase>
{
  GLuint texture = 0;
  GLuint sampler = 0;
  size_t transform_index;
  ObjectRef<DataArray1D> image;

  friend void compressed_image2d_init_objects(ObjectRef<SamplerCompressedImage2D> samplerObj,
    int filter,
    GLenum wrapS,
    GLenum wrapT,
    GLenum internalformat,
    GLsizei width,
    GLsizei height,
    GLsizei byteSize,
    GLuint buffer);

 public:
  Object(ANARIDevice d, ANARIObject handle);

  void commit() override;
  void update() override;

  void allocateResources(SurfaceObjectBase *, int) override;
  void drawCommand(int index, DrawCommand &command) override;
  void declare(int index, AppendableShader &shader) override;
  void sample(int index, AppendableShader &shader, const char *meta) override;
  std::array<uint32_t, 4> metadata() override;

  ~Object();
};

} // namespace visgl

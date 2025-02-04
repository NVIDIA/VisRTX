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

#pragma once

#include "VisGLDevice.h"
#include "AppendableShader.h"

#include <array>

namespace visgl {

template <>
class Object<Spatial_FieldStructuredRegular>
    : public DefaultObject<Spatial_FieldStructuredRegular,
          SpatialFieldObjectBase>
{
  GLuint sampler = 0;
  size_t transform_index;
  ObjectRef<Array3D> data;
  std::array<float, 4> origin{0.0f, 0.0f, 0.0f, 1.0f};
  std::array<float, 4> spacing{1.0f, 1.0f, 1.0f, 0.0f};

  GLuint vao = 0;
  GLuint box_position = 0;
  GLuint box_index = 0;

  friend void field_init_objects(
      ObjectRef<Spatial_FieldStructuredRegular> samplerObj, int filter);

 public:
  Object(ANARIDevice d, ANARIObject handle);

  void commit() override;
  void update() override;

  void drawCommand(VolumeObjectBase *, DrawCommand &) override;
  void vertexShaderMain(VolumeObjectBase *, AppendableShader &) override;
  void fragmentShaderMain(VolumeObjectBase *, AppendableShader &) override;
  uint32_t index() override;
  std::array<float, 6> bounds() override;

  ~Object();
};

} // namespace visgl

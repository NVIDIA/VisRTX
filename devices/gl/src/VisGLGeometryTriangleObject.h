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

#include <vector>

namespace visgl {

template <>
class Object<GeometryTriangle>
    : public DefaultObject<GeometryTriangle, GeometryObjectBase>
{
  ObjectRef<DataArray1D> position_array;
  ObjectRef<DataArray1D> color_array;
  ObjectRef<DataArray1D> normal_array;
  ObjectRef<DataArray1D> attribute0_array;
  ObjectRef<DataArray1D> attribute1_array;
  ObjectRef<DataArray1D> attribute2_array;
  ObjectRef<DataArray1D> attribute3_array;

  ObjectRef<DataArray1D> primitive_color_array;
  ObjectRef<DataArray1D> primitive_attribute0_array;
  ObjectRef<DataArray1D> primitive_attribute1_array;
  ObjectRef<DataArray1D> primitive_attribute2_array;
  ObjectRef<DataArray1D> primitive_attribute3_array;
  ObjectRef<DataArray1D> primitive_id_array;

  ObjectRef<DataArray1D> index_array;

  bool dirty = true;

  friend void triangles_init_objects(ObjectRef<GeometryTriangle> triangleObj);

  void interfaceBlock(SurfaceObjectBase *, AppendableShader &);

 public:
  GLuint vao = 0;

  Object(ANARIDevice d, ANARIObject handle);
  ~Object();

  void commit() override;
  void update() override;
  void allocateResources(SurfaceObjectBase *) override;
  void drawCommand(SurfaceObjectBase *, DrawCommand &) override;
  void vertexShader(SurfaceObjectBase *, AppendableShader &) override;
  void fragmentShaderMain(SurfaceObjectBase *, AppendableShader &) override;

  void vertexShaderShadow(SurfaceObjectBase *, AppendableShader &) override;
  void geometryShaderShadow(SurfaceObjectBase *, AppendableShader &) override;
  void fragmentShaderShadowMain(
      SurfaceObjectBase *, AppendableShader &) override;

  void vertexShaderOcclusion(SurfaceObjectBase *, AppendableShader &) override;

  std::array<float, 6> bounds() override;
  uint32_t index() override;
};

} // namespace visgl

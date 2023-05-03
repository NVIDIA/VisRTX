// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "VisGLDevice.h"

#include <vector>

namespace visgl{


template <>
class Object<GeometrySphere> : public DefaultObject<GeometrySphere, GeometryObjectBase>
{
  ObjectRef<DataArray1D> position_array;
  ObjectRef<DataArray1D> radius_array;
  ObjectRef<DataArray1D> color_array;
  ObjectRef<DataArray1D> attribute0_array;
  ObjectRef<DataArray1D> attribute1_array;
  ObjectRef<DataArray1D> attribute2_array;
  ObjectRef<DataArray1D> attribute3_array;

  ObjectRef<DataArray1D> primitive_id_array;

  ObjectRef<DataArray1D> index_array;

  float radius;

  size_t geometry_index;

  bool dirty = true;

  friend void sphere_init_objects(ObjectRef<GeometrySphere> sphereObj);
 public:
  GLuint vao = 0;
  GLuint occlusion_resolve_vao = 0;
  GLuint ico_position = 0;
  GLuint ico_index = 0;

  Object(ANARIDevice d, ANARIObject handle);
  ~Object();

  void commit() override;
  void update() override;
  void allocateResources(SurfaceObjectBase*) override;
  void drawCommand(SurfaceObjectBase*, DrawCommand&) override;
  void vertexShader(SurfaceObjectBase*, AppendableShader&) override;
  void fragmentShaderMain(SurfaceObjectBase*, AppendableShader&) override;

  void vertexShaderShadow(SurfaceObjectBase*, AppendableShader&) override;
  void geometryShaderShadow(SurfaceObjectBase*, AppendableShader&) override;
  void fragmentShaderShadowMain(SurfaceObjectBase*, AppendableShader&) override;

  void vertexShaderOcclusion(SurfaceObjectBase*, AppendableShader&) override;

  std::array<float, 6> bounds() override;
  uint32_t index() override;
};

} //namespace visgl


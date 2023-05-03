// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "VisGLDevice.h"

#include <vector>

namespace visgl{


template <>
class Object<GeometryCylinder> : public DefaultObject<GeometryCylinder, GeometryObjectBase>
{
  ObjectRef<DataArray1D> position_array;
  ObjectRef<DataArray1D> cap_array;
  ObjectRef<DataArray1D> color_array;
  ObjectRef<DataArray1D> attribute0_array;
  ObjectRef<DataArray1D> attribute1_array;
  ObjectRef<DataArray1D> attribute2_array;
  ObjectRef<DataArray1D> attribute3_array;

  ObjectRef<DataArray1D> primitive_color_array;
  ObjectRef<DataArray1D> primitive_radius_array;
  ObjectRef<DataArray1D> primitive_attribute0_array;
  ObjectRef<DataArray1D> primitive_attribute1_array;
  ObjectRef<DataArray1D> primitive_attribute2_array;
  ObjectRef<DataArray1D> primitive_attribute3_array;
  ObjectRef<DataArray1D> primitive_id_array;

  ObjectRef<DataArray1D> index_array;

  size_t geometry_index;

  float radius;
  int caps;

  bool dirty = true;

  friend void cylinder_init_objects(ObjectRef<GeometryCylinder> cylinderObj);
 public:
  GLuint vao = 0;
  GLuint occlusion_resolve_vao = 0;
  GLuint occlusion_resolve_shader = 0;
  GLuint cyl_position = 0;
  GLuint cyl_index = 0;
  uint32_t index_count;
  uint32_t body_index_count;

  Object(ANARIDevice d, ANARIObject handle);
  ~Object();

  void commit() override;
  void update() override;
  void allocateResources(SurfaceObjectBase*) override;
  void declarations(SurfaceObjectBase*, AppendableShader&);
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


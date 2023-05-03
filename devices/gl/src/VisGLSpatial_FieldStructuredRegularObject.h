// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "VisGLDevice.h"
#include "AppendableShader.h"

#include <array>

namespace visgl{


template <>
class Object<Spatial_FieldStructuredRegular> : public DefaultObject<Spatial_FieldStructuredRegular, SpatialFieldObjectBase>
{
  GLuint sampler = 0;
  size_t transform_index;
  ObjectRef<Array3D> data;
  std::array<float, 4> origin{0.0f, 0.0f, 0.0f, 1.0f};
  std::array<float, 4> spacing{1.0f, 1.0f, 1.0f, 0.0f};

  GLuint vao = 0;
  GLuint box_position = 0;
  GLuint box_index = 0;

  friend void field_init_objects(ObjectRef<Spatial_FieldStructuredRegular> samplerObj, int filter);
public:

  Object(ANARIDevice d, ANARIObject handle);

  void commit() override;
  void update() override;

  void drawCommand(VolumeObjectBase*, DrawCommand&) override;
  void vertexShaderMain(VolumeObjectBase*, AppendableShader&) override;
  void fragmentShaderMain(VolumeObjectBase*, AppendableShader&) override;
  uint32_t index() override;
  std::array<float, 6> bounds() override;

  ~Object();

};

} //namespace visgl


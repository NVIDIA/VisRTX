// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "VisGLDevice.h"
#include "AppendableShader.h"

#include <array>

namespace visgl{


template <>
class Object<SamplerImage3D> : public DefaultObject<SamplerImage3D, SamplerObjectBase>
{
  GLuint sampler = 0;
  size_t transform_index;
  ObjectRef<Array3D> image;

  friend void image3d_init_objects(ObjectRef<SamplerImage3D> samplerObj, int filter, GLenum wrapS, GLenum wrapT, GLenum wrapR);
public:

  Object(ANARIDevice d, ANARIObject handle);

  void commit() override;
  void update() override;

  void allocateResources(SurfaceObjectBase*, int) override;
  void drawCommand(int index, DrawCommand &command) override;
  void declare(int index, AppendableShader &shader) override;
  void sample(int index, AppendableShader &shader, const char *meta) override;
  std::array<uint32_t, 4> metadata() override;

  ~Object();

};

} //namespace visgl


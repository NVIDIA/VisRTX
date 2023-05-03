// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "VisGLDevice.h"

namespace visgl{


template <>
class Object<World> : public DefaultObject<World>
{
public:
  //occlusion
  GLuint occlusionbuffer = 0;
  uint32_t occlusioncapacity = 0;
  uint32_t occlusionsamples = 0;
  //occlusion

  //shadowmaps
  GLuint shadowtex = 0;
  GLuint shadowfbo = 0;
  uint32_t shadow_map_size = 0;
  int shadow_map_count = 0;
  //shadowmaps


  uint64_t worldEpoch = 0;
  uint64_t lightEpoch = 0;
  uint64_t geometryEpoch = 0;

  Object(ANARIDevice d, ANARIObject handle);

  ~Object();

  int getProperty(const char *propname,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    ANARIWaitMask mask) override;
};

} //namespace visgl


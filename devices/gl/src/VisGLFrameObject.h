/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace visgl{

class CollectScene;

template <>
class Object<Frame> : public DefaultObject<Frame, FrameObjectBase>
{
  std::array<uint32_t, 2> size{0, 0};
  ANARIDataType colorType = ANARI_UNKNOWN;
  ANARIDataType depthType = ANARI_UNKNOWN;

  GLuint colortarget = 0;
  GLuint colorbuffer = 0;
  GLuint depthtarget = 0;
  GLuint depthbuffer = 0;
  GLuint fbo = 0;

  GLuint multicolortarget = 0;
  GLuint multidepthtarget = 0;
  GLuint multifbo = 0;

  GLuint shadowubo= 0;
  bool shadow_dirty = true;
  uint32_t shadow_map_size = 4096;

  int occlusionMode = STRING_ENUM_none;

  GLuint sceneubo = 0;

  GLuint duration_query = 0;
  uint64_t duration = 1;

  bool configuration_changed = true;

  size_t camera_index = 0;

  std::unique_ptr<CollectScene> collector;
  
  friend void frame_allocate_objects(ObjectRef<Frame> frameObj);
  friend void frame_map_color(ObjectRef<Frame> frameObj, uint64_t size, void **ptr);
  friend void frame_map_depth(ObjectRef<Frame> frameObj, uint64_t size, void **ptr);
  friend void frame_unmap_color(ObjectRef<Frame> frameObj);
  friend void frame_unmap_depth(ObjectRef<Frame> frameObj);
  friend void frame_render(ObjectRef<Frame> frameObj, uint32_t width, uint32_t height, uint32_t camera_index, uint32_t ambient_index, std::array<float, 4> clearColor);
public:
  Object(ANARIDevice d, ANARIObject handle);
  ~Object();

  void commit() override;
  void update() override;

  int getProperty(const char *propname,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    ANARIWaitMask mask) override;

  void *mapFrame(
      const char *, uint32_t *, uint32_t *, ANARIDataType *) override;
  void unmapFrame(const char *) override;
  void renderFrame() override;
  void discardFrame() override;
  int frameReady(ANARIWaitMask mask) override;
};

} //namespace visgl


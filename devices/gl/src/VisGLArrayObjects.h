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
#include "anari2gl_types.h"
#include "shader_blocks.h"

#include <vector>

namespace visgl {

template <>
struct ObjectAllocator<Array1D>
{
  static ArrayObjectBase *allocate(ANARIDevice d,
      ANARIObject handle,
      const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType type,
      uint64_t numItems1);
};

template <>
class Object<Array2D> : public DefaultObject<Array2D, ArrayObjectBase>
{
  const void *appMemory;
  ANARIMemoryDeleter deleter;
  const void *userdata;
  ANARIDataType elementType;
  uint64_t numItems1;
  uint64_t numItems2;

  GLuint texture = 0;
  std::future<void> future;

  friend void array2d_allocate_objects(ObjectRef<Array2D> arrayObj);
  friend void array2d_unmap_to_buffer(ObjectRef<Array2D> arrayObj);

 public:
  Object(ANARIDevice d,
      ANARIObject handle,
      const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType type,
      uint64_t numItems1,
      uint64_t numItems2);

  void init() override;
  void *map() override;
  void unmap() override;
  void releasePublic() override;
  ANARIDataType getElementType() const override;
  uint64_t size() const override
  {
    return numItems1 * numItems2;
  }
  int dims(uint64_t *d) const override;
  GLuint getTexture2D();

  ~Object();
};

template <>
class Object<Array3D> : public DefaultObject<Array3D, ArrayObjectBase>
{
  const void *appMemory;
  ANARIMemoryDeleter deleter;
  const void *userdata;
  ANARIDataType elementType;
  uint64_t numItems1;
  uint64_t numItems2;
  uint64_t numItems3;

  GLuint texture = 0;
  std::future<void> future;

  friend void array3d_allocate_objects(ObjectRef<Array3D> arrayObj);
  friend void array3d_unmap_to_buffer(ObjectRef<Array3D> arrayObj);

 public:
  Object(ANARIDevice d,
      ANARIObject handle,
      const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType type,
      uint64_t numItems1,
      uint64_t numItems2,
      uint64_t numItems3);

  void init() override;
  void *map() override;
  void unmap() override;
  void releasePublic() override;
  ANARIDataType getElementType() const override;
  uint64_t size() const override
  {
    return numItems1 * numItems2 * numItems3;
  }
  int dims(uint64_t *d) const override;
  GLuint getTexture3D();

  ~Object();
};

} // namespace visgl

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

#include "VisGLObject.h"
#include "VisGLObjects.h"
#include "queue_thread.h"

#include "glContextInterface.h"
#include "ogl.h"
#include "shader_compile_segmented.h"
#include "AppendableShader.h"

#include <memory>
#include <iostream>
#include <array>
#include <thread>
#include <unordered_map>
#include <atomic>
#include <vector>

namespace visgl {

template <typename T, typename G>
class StorageBuffer
{
  static const int N = 1;
  G *gl = nullptr;
  std::vector<T> data;
  GLuint ssbo[N] = {};
  size_t ssbo_capacity = 0;
  bool dirty = false;
  std::mutex mutex;
  std::condition_variable condition;
  bool busy = false;

 public:
  void init(G *gl)
  {
    this->gl = gl;
    gl->GenBuffers(N, ssbo);
  }
  size_t allocate(size_t n)
  {
    std::unique_lock<std::mutex> guard(mutex);
    condition.wait(guard, [&] { return !busy; });

    size_t index = data.size();
    data.resize(index + n);
    dirty = true;
    return index;
  }
  void set(size_t index, const T &value)
  {
    std::unique_lock<std::mutex> guard(mutex);
    condition.wait(guard, [&] { return !busy; });

    data[index] = value;
    dirty = true;
  }
  template <typename U>
  void setMem(size_t index, const U *mem)
  {
    static_assert(sizeof(T) == sizeof(U), "setMem size mismatch");
    std::unique_lock<std::mutex> guard(mutex);
    condition.wait(guard, [&] { return !busy; });

    std::memcpy(&data[index], mem, sizeof(T));
    dirty = true;
  }
  void lock()
  {
    std::unique_lock<std::mutex> lock(mutex);
    busy = true;
  }
  GLuint consume()
  {
    std::unique_lock<std::mutex> lock(mutex);
    if (dirty) {
      gl->BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[0]);
      // this will either resize or orphan the buffer
      gl->BufferData(GL_SHADER_STORAGE_BUFFER,
          data.capacity() * sizeof(T),
          0,
          GL_DYNAMIC_DRAW);
      gl->BufferSubData(
          GL_SHADER_STORAGE_BUFFER, 0, data.size() * sizeof(T), data.data());

      dirty = false;
    }
    busy = false;
    condition.notify_all();
    return ssbo[0];
  }
  void release()
  {
    if(gl) {
      gl->DeleteBuffers(N, ssbo);
    }
  }
};

struct OcclusionResources
{
  GLuint fbo = 0;
  GLuint tex = 0;
  int size = 1024;
  GLuint clear_shader = 0;
};

template <>
class Object<Device> : public DefaultObject<Device>
{
  std::atomic<uint64_t> epochCounter{0u};
  friend uint64_t anariIncrementEpoch(Object<Device> *, ObjectBase *);

  OcclusionResources occlusion;

 public:
  std::unique_ptr<glContextInterface> context;
  int clientapi;
  GladGLContext gl{};
  std::vector<const char*> extensions;

  queue_thread queue;
  StorageBuffer<std::array<float, 16>, GladGLContext> transforms;
  StorageBuffer<std::array<float, 4>, GladGLContext> materials;
  StorageBuffer<std::array<float, 4>, GladGLContext> lights;
  ShaderCache<SHADER_SEGMENTS, GladGLContext> shaders;

  OcclusionResources *getOcclusionResources();

  Object(ANARIDevice d);
  int getProperty(const char *propname,
      ANARIDataType type,
      void *mem,
      uint64_t size,
      ANARIWaitMask mask) override;

  void commit() override;
  void update() override;

  uint64_t globalEpoch() const;

  ~Object();
};

} // namespace visgl

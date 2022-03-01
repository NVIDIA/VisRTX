/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// anari
#include "anari/detail/IntrusivePtr.h"
#include "anari/detail/ParameterizedObject.h"
// anari_cpp
#include "anari/anari_cpp/ext/glm.h"
// glm
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"
// std
#include <map>
#include <memory>
#include <string_view>
#include <utility>
// cuda/optix
#include "optix_visrtx.h"

// clang-format off
#define VISRTX_COMMIT_PRIORITY_DEFAULT  0
#define VISRTX_COMMIT_PRIORITY_MATERIAL 1
#define VISRTX_COMMIT_PRIORITY_SURFACE  2
#define VISRTX_COMMIT_PRIORITY_VOLUME   2
#define VISRTX_COMMIT_PRIORITY_GROUP    3
#define VISRTX_COMMIT_PRIORITY_WORLD    4
// clang-format on

namespace visrtx {

struct Object : public anari::RefCounted, public anari::ParameterizedObject
{
  Object() = default;
  virtual ~Object() = default;

  template <typename T>
  T *getParamObject(const std::string &name, T *valIfNotFound = nullptr);

  virtual bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags);

  virtual void commit();
  virtual void upload();

  virtual void *deviceData() const;

  void setObjectType(ANARIDataType type);
  ANARIDataType type() const;

  DeviceGlobalState *deviceState() const;
  void setDeviceState(DeviceGlobalState *d);

  TimeStamp lastUpdated() const;
  void markUpdated();

  TimeStamp lastCommitted() const;
  virtual void markCommitted();

  int commitPriority() const;

  template <typename... Args>
  void reportMessage(
      ANARIStatusSeverity, const char *fmt, Args &&...args) const;

 protected:
  void setCommitPriority(int priority);

 private:
  int m_commitPriority{VISRTX_COMMIT_PRIORITY_DEFAULT};
  DeviceGlobalState *m_deviceState{nullptr};
  TimeStamp m_lastUpdated{0};
  TimeStamp m_lastCommitted{0};
  ANARIDataType m_type{ANARI_OBJECT};
};

std::string string_printf(const char *fmt, ...);

// Inlined defintions /////////////////////////////////////////////////////////

template <typename T>
inline T *Object::getParamObject(const std::string &name, T *valIfNotFound)
{
  if (!hasParam(name))
    return valIfNotFound;

  try {
    using OBJECT_T = typename std::remove_pointer<T>::type;
    using PTR_T = anari::IntrusivePtr<OBJECT_T>;

    PTR_T val = getParam<PTR_T>(name, PTR_T());
    return val.ptr;
  } catch (...) {
    return valIfNotFound;
  }
}

template <typename... Args>
inline void Object::reportMessage(
    ANARIStatusSeverity severity, const char *fmt, Args &&...args) const
{
  auto *state = deviceState();

  if (!state)
    throw std::runtime_error("malformed object created without globals ptr");

  auto msg = string_printf(fmt, std::forward<Args>(args)...);
  state->messageFunction(severity, msg, this);
}

} // namespace visrtx

// anari::ParameterizedObject specializations /////////////////////////////////

namespace anari {

template <>
inline visrtx::box1 ParameterizedObject::getParam<visrtx::box1>(
    const std::string &name, visrtx::box1 nfv)
{
  auto tmp = getParam<visrtx::vec2>(name, {nfv.lower, nfv.upper});
  return visrtx::make_box1(tmp);
}

} // namespace anari

#define VISRTX_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define VISRTX_ANARI_TYPEFOR_DEFINITION(type)                                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Object *, ANARI_OBJECT);

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

#include "AnariAny.h"
#include "anari/anari_cpp/Traits.h"
// anari
#include "anari/backend/utilities/IntrusivePtr.h"
// stl
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace visrtx {

struct ParameterizedObject
{
  ParameterizedObject() = default;
  virtual ~ParameterizedObject() = default;

  bool hasParam(const std::string &name);

  void setParam(const std::string &name, ANARIDataType type, const void *v);

  template <typename T>
  T getParam(const std::string &name, T valIfNotFound);

  template <typename T>
  T *getParamObject(const std::string &name);

  std::string getParamString(
      const std::string &name, const std::string &valIfNotFound);

  AnariAny getParamDirect(const std::string &name);
  void setParamDirect(const std::string &name, const AnariAny &v);

  void removeParam(const std::string &name);

 private:
  // Data members //

  using Param = std::pair<std::string, AnariAny>;

  Param *findParam(const std::string &name, bool addIfNotExist = false);

  std::vector<Param> m_params;
};

// Inlined ParameterizedObject definitions ////////////////////////////////////

template <typename T>
inline T ParameterizedObject::getParam(const std::string &name, T valIfNotFound)
{
  constexpr ANARIDataType type = anari::ANARITypeFor<T>::value;
  static_assert(!anari::isObject(type),
      "use ParameterizedObect::getParamObject() for getting objects");
  static_assert(type != ANARI_STRING && !std::is_same_v<T, std::string>,
      "use ParameterizedObject::getParamString() for getting strings");
  auto *p = findParam(name);
  return p && p->second.type() == type? p->second.get<T>() : valIfNotFound;
}

template <typename T>
inline T *ParameterizedObject::getParamObject(const std::string &name)
{
  auto *p = findParam(name);
  return p ? p->second.getObject<T>() : nullptr;
}

} // namespace visrtx

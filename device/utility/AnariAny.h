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

#include <anari/backend/utilities/IntrusivePtr.h>
#include <anari/type_utility.h>
#include <anari/anari_cpp.hpp>
// std
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

namespace visrtx {

struct AnariAny
{
  AnariAny();
  AnariAny(const AnariAny &copy);

  template <typename T>
  AnariAny(T value);

  AnariAny(ANARIDataType type, const void *v);

  ~AnariAny();

  AnariAny &operator=(const AnariAny &rhs);

  template <typename T>
  AnariAny &operator=(T rhs);

  bool operator==(const AnariAny &rhs) const;
  bool operator!=(const AnariAny &rhs) const;

  template <typename T>
  T get() const;

  template <typename T>
  anari::IntrusivePtr<T> getObject() const;

  std::string getString() const;

  template <typename T>
  bool is() const;

  bool is(ANARIDataType t) const;

  ANARIDataType type() const;

  bool valid() const;
  void reset();

 private:
  template <typename T>
  T storageAs() const;

  void refIncObject() const;
  void refDecObject() const;

  constexpr static int MAX_LOCAL_STORAGE = 16 * sizeof(float);

  std::array<uint8_t, MAX_LOCAL_STORAGE> m_storage;
  std::string m_string;
  ANARIDataType m_type{ANARI_UNKNOWN};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline AnariAny::AnariAny()
{
  reset();
}

inline AnariAny::AnariAny(const AnariAny &copy)
{
  reset();
  std::memcpy(m_storage.data(), copy.m_storage.data(), m_storage.size());
  m_string = copy.m_string;
  m_type = copy.m_type;
  refIncObject();
}

template <typename T>
inline AnariAny::AnariAny(T value)
{
  constexpr auto type = anari::ANARITypeFor<T>::value;
  static_assert(
      type != ANARI_UNKNOWN, "unknown type used initialize visrtx::AnariAny");

  if constexpr (type == ANARI_STRING)
    m_string = value;
  else
    std::memcpy(m_storage.data(), &value, sizeof(value));

  m_type = type;
  refIncObject();
}

inline AnariAny::AnariAny(ANARIDataType type, const void *v)
{
  m_type = type;
  if (type == ANARI_STRING)
    m_string = (const char *)v;
  else
    std::memcpy(m_storage.data(), v, anari::sizeOf(type));
  refIncObject();
}

inline AnariAny::~AnariAny()
{
  reset();
}

inline AnariAny &AnariAny::operator=(const AnariAny &rhs)
{
  reset();
  std::memcpy(m_storage.data(), rhs.m_storage.data(), m_storage.size());
  m_string = rhs.m_string;
  m_type = rhs.m_type;
  refIncObject();
  return *this;
}

template <typename T>
inline AnariAny &AnariAny::operator=(T rhs)
{
  return *this = AnariAny(rhs);
}

inline bool AnariAny::operator==(const AnariAny &rhs) const
{
  if (!valid() || !rhs.valid())
    return false;
  if (type() != rhs.type())
    return false;
  if (type() == ANARI_BOOL)
    return get<bool>() == rhs.get<bool>();
  else if (type() == ANARI_STRING)
    return m_string == rhs.m_string;
  else {
    return std::equal(m_storage.data(),
        m_storage.data() + ::anari::sizeOf(type()),
        rhs.m_storage.data());
  }
}

inline bool AnariAny::operator!=(const AnariAny &rhs) const
{
  return !(*this == rhs);
}

template <typename T>
inline T AnariAny::get() const
{
  constexpr ANARIDataType type = anari::ANARITypeFor<T>::value;
  static_assert(
      !anari::isObject(type), "use AnariAny::getObject() for getting objects");
  static_assert(
      type != ANARI_STRING, "use AnariAny::getString() for getting strings");

  if (!valid())
    throw std::runtime_error("get() called on empty visrtx::AnariAny");
  if (!is<T>()) {
    throw std::runtime_error(
        "get() called with invalid type on visrtx::AnariAny");
  }

  return storageAs<T>();
}

template <typename T>
inline anari::IntrusivePtr<T> AnariAny::getObject() const
{
  constexpr ANARIDataType type = anari::ANARITypeFor<T>::value;
  static_assert(
      anari::isObject(type), "use AnariAny::get() for getting non-objects");
  return anari::IntrusivePtr<T>(storageAs<T *>());
}

template <typename T>
inline bool AnariAny::is() const
{
  return is(::anari::ANARITypeFor<T>::value);
}

inline bool AnariAny::is(ANARIDataType t) const
{
  return type() == t;
}

inline ANARIDataType AnariAny::type() const
{
  return m_type;
}

inline bool AnariAny::valid() const
{
  return type() != ANARI_UNKNOWN;
}

inline void AnariAny::reset()
{
  refDecObject();
  std::fill(m_storage.begin(), m_storage.end(), 0);
  m_string.clear();
  m_type = ANARI_UNKNOWN;
}

template <typename T>
inline T AnariAny::storageAs() const
{
  static_assert(sizeof(T) <= MAX_LOCAL_STORAGE, "AnariAny: not enough storage");
  T retval;
  std::memcpy(&retval, m_storage.data(), sizeof(retval));
  return retval;
}

inline std::string AnariAny::getString() const
{
  return type() == ANARI_STRING ? m_string : "";
}

inline void AnariAny::refIncObject() const
{
  if (anari::isObject(type())) {
    auto *o = storageAs<anari::RefCounted *>();
    if (o)
      o->refInc(anari::RefType::INTERNAL);
  }
}

inline void AnariAny::refDecObject() const
{
  if (anari::isObject(type())) {
    auto *o = storageAs<anari::RefCounted *>();
    if (o)
      o->refDec(anari::RefType::INTERNAL);
  }
}

} // namespace visrtx

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

#include "anari/anari_cpp.hpp"
#include "anari/detail/Helpers.h"
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

  ~AnariAny() = default;

  AnariAny &operator=(const AnariAny &rhs);

  template <typename T>
  AnariAny &operator=(T rhs);

  bool operator==(const AnariAny &rhs) const;
  bool operator!=(const AnariAny &rhs) const;

  template <typename T>
  T get() const;

  const void *data() const;

  template <typename T>
  bool is() const;

  bool is(ANARIDataType t) const;

  ANARIDataType type() const;

  bool valid() const;
  void reset();

 private:
  constexpr static int MAX_LOCAL_STORAGE = 16 * sizeof(float);

  std::array<uint8_t, MAX_LOCAL_STORAGE> m_storage;
  ANARIDataType m_type{ANARI_UNKNOWN};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline AnariAny::AnariAny()
{
  reset();
}

inline AnariAny::AnariAny(const AnariAny &copy)
{
  std::memcpy(m_storage.data(), copy.m_storage.data(), m_storage.size());
  m_type = copy.m_type;
}

template <typename T>
inline AnariAny::AnariAny(T value)
{
  constexpr auto type = ::anari::ANARITypeFor<T>::value;
  static_assert(
      type != ANARI_UNKNOWN, "unknown type used initialize visrtx::AnariAny");

  std::memcpy(m_storage.data(), &value, sizeof(value));
  m_type = type;
}

inline AnariAny &AnariAny::operator=(const AnariAny &rhs)
{
  std::memcpy(m_storage.data(), rhs.m_storage.data(), m_storage.size());
  m_type = rhs.m_type;
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
  return std::equal(m_storage.data(),
      m_storage.data() + ::anari::sizeOfDataType(type()),
      rhs.m_storage.data());
}

inline bool AnariAny::operator!=(const AnariAny &rhs) const
{
  return !(*this == rhs);
}

template <typename T>
inline T AnariAny::get() const
{
  if (!valid())
    throw std::runtime_error("get() called on empty visrtx::AnariAny");
  if (!is<T>()) {
    throw std::runtime_error(
        "get() called with invalid type on visrtx::AnariAny");
  }
  T retval;
  std::memcpy(&retval, data(), sizeof(T));
  return retval;
}

inline const void *AnariAny::data() const
{
  return m_storage.data();
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
  std::fill(m_storage.begin(), m_storage.end(), 0);
  m_type = ANARI_UNKNOWN;
}

} // namespace visrtx

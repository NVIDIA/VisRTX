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

#include "utility/AnariAny.h"
// std
#include <string>

namespace visrtx {

struct ParameterInfo
{
  template <typename T>
  ParameterInfo(bool required, std::string_view description, T defaultValue);

  template <typename T>
  ParameterInfo(bool required,
      std::string_view description,
      T defaultValue,
      T min,
      T max);

  bool required() const;
  const char *description() const;
  const void *defaultValue() const;
  const void *minValue() const;
  const void *maxValue() const;

  bool hasDescription() const;
  bool hasDefaultValue() const;
  bool hasMin() const;
  bool hasMax() const;

  const void *fromString(
      std::string_view name, ANARIDataType expectedType) const;

  ANARIDataType type() const;

 private:
  ParameterInfo(bool required, std::string_view description);

  bool m_required{false};
  std::string m_description;

  AnariAny m_default;
  AnariAny m_min;
  AnariAny m_max;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline ParameterInfo::ParameterInfo(bool required, std::string_view description)
    : m_required(required), m_description(description)
{}

template <typename T>
inline ParameterInfo::ParameterInfo(
    bool required, std::string_view description, T defaultValue)
    : ParameterInfo(required, description)
{
  m_default = defaultValue;
}

template <typename T>
inline ParameterInfo::ParameterInfo(
    bool required, std::string_view description, T defaultValue, T min, T max)
    : ParameterInfo(required, description, defaultValue)
{
  m_min = min;
  m_max = max;
}

inline bool ParameterInfo::required() const
{
  return m_required;
}

inline const char *ParameterInfo::description() const
{
  return m_description.c_str();
}

inline const void *ParameterInfo::defaultValue() const
{
  return m_default.data();
}

inline const void *ParameterInfo::minValue() const
{
  return m_min.data();
}

inline const void *ParameterInfo::maxValue() const
{
  return m_max.data();
}

inline bool ParameterInfo::hasDescription() const
{
  return !m_description.empty();
}

inline bool ParameterInfo::hasDefaultValue() const
{
  return m_default.valid();
}

inline bool ParameterInfo::hasMin() const
{
  return m_min.valid();
}

inline bool ParameterInfo::hasMax() const
{
  return m_max.valid();
}

inline const void *ParameterInfo::fromString(
    std::string_view name, ANARIDataType expectedType) const
{
  if (name == "required")
    return &m_required;
  else if (hasDescription() && name == "description")
    return description();
  else if (hasDefaultValue() && expectedType == m_default.type()
      && name == "default")
    return defaultValue();
  else if (hasMin() && expectedType == m_min.type() && name == "minimum")
    return minValue();
  else if (hasMax() && expectedType == m_max.type() && name == "maximum")
    return maxValue();

  return nullptr;
}

inline ANARIDataType ParameterInfo::type() const
{
  if (m_default.valid())
    return m_default.type();
  return ANARI_UNKNOWN;
}

} // namespace visrtx

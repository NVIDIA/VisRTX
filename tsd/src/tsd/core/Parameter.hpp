// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Any.hpp"
#include "Token.hpp"
// std
#include <string>
#include <string_view>
#include <vector>

namespace tsd {

using Any = tsd::utility::Any;

enum ParameterUsageHint
{
  NONE = 0,
  COLOR = (1 << 0),
  DIRECTION = (1 << 1),
  FILE = (1 << 2)
};

struct Parameter;
struct ParameterObserver
{
  virtual void parameterChanged(const Parameter *p) = 0;
};

struct Parameter
{
  template <typename T>
  Parameter(ParameterObserver *object,
      Token name,
      T value,
      std::string description = "",
      ParameterUsageHint usage = ParameterUsageHint::NONE,
      Any min = {},
      Any max = {});

  Parameter(ParameterObserver *object,
      Token name,
      ANARIDataType type,
      const void *v,
      ParameterUsageHint usage = ParameterUsageHint::NONE);

  Token name() const;
  const std::string &description() const;
  void setDescription(const char *d);

  // Value //

  const void *data() const;

  const Any &value() const;
  void setValue(const Any &newValue);
  ANARIDataType type() const;

  template <typename T>
  void operator=(T newValue);

  ParameterUsageHint usage() const;
  void setUsage(ParameterUsageHint u);

  // Value min/max bounds //

  void setMin(const Any &newMin);
  void setMax(const Any &newMax);
  const Any &min() const;
  const Any &max() const;
  bool hasMin() const;
  bool hasMax() const;

  const std::vector<std::string> &stringValues() const;
  void setStringValues(const std::vector<std::string> &sv);
  int stringSelection() const;
  void setStringSelection(int s);

  ////////////////////////////////////////

  Parameter() = default;
  ~Parameter() = default;

  Parameter(const Parameter &) = default;
  Parameter(Parameter &&) = default;

  Parameter &operator=(const Parameter &) = default;
  Parameter &operator=(Parameter &&) = default;

 private:
  friend struct Object;

  void setObserver(ParameterObserver *o);

  ParameterObserver *m_observer{nullptr};
  Token m_name;
  std::string m_description;
  ParameterUsageHint m_usageHint{ParameterUsageHint::NONE};
  Any m_value;
  Any m_min;
  Any m_max;
  std::vector<std::string> m_stringValues;
  int m_stringSelection{0};
};

template <typename T>
constexpr ANARIDataType anariType()
{
  return anari::ANARITypeFor<T>::value;
}

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline Parameter::Parameter(ParameterObserver *object,
    Token name,
    T value,
    std::string description,
    ParameterUsageHint usage,
    Any minValue,
    Any maxValue)
    : m_observer(object),
      m_name(name),
      m_description(description),
      m_usageHint(usage)
{
  setValue(value);
  setMin(minValue);
  setMax(maxValue);
}

template <typename T>
inline void Parameter::operator=(T newValue)
{
  setValue(newValue);
}

} // namespace tsd
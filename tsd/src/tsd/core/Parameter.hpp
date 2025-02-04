// Copyright 2024-2025 NVIDIA Corporation
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
  Parameter(ParameterObserver *object, Token name);

  Token name() const;
  const std::string &description() const;

  bool isEnabled() const;

  // Builder pattern methods for progressive construction //

  Parameter &setDescription(const char *d);
  Parameter &setValue(const Any &newValue);
  Parameter &setMin(const Any &newMin);
  Parameter &setMax(const Any &newMax);
  Parameter &setStringValues(const std::vector<std::string> &sv);
  Parameter &setStringSelection(int s);
  Parameter &setUsage(ParameterUsageHint u);
  Parameter &setEnabled(bool enabled);

  // Value access //

  const Any &value() const;

  template <typename T>
  void operator=(T newValue);

  ParameterUsageHint usage() const;

  // Value min/max bounds //

  const Any &min() const;
  const Any &max() const;
  bool hasMin() const;
  bool hasMax() const;

  // Methods when holding multi-string values //

  const std::vector<std::string> &stringValues() const;
  int stringSelection() const;

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
  bool m_enabled{true};
};

template <typename T>
constexpr ANARIDataType anariType()
{
  return anari::ANARITypeFor<T>::value;
}

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline void Parameter::operator=(T newValue)
{
  setValue(newValue);
}

} // namespace tsd
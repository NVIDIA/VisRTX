// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/Any.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/core/TypeMacros.hpp"
// std
#include <string>
#include <string_view>
#include <vector>

namespace tsd::scene {

using namespace tsd::core;

enum ParameterUsageHint
{
  NONE = 0,
  COLOR = (1 << 0),
  DIRECTION = (1 << 1),
  FILE = (1 << 2),
  VALUE_RANGE_TRANSFORM = (1 << 3)
};

// clang-format off
/*
 * Observer interface implemented by Object; receives notifications when a
 * Parameter's value changes or when a Parameter is removed from its parent.
 *
 * Example:
 *   struct MyObj : ParameterObserver {
 *     void parameterChanged(const Parameter *p, const Any &old) override { ... }
 *     void removeParameter(const Parameter *p) override { ... }
 *   };
 */
// clang-format on
struct Parameter;
struct ParameterObserver
{
  virtual void parameterChanged(const Parameter *p, const Any &oldValue) = 0;
  virtual void removeParameter(const Parameter *p) = 0;
};

/*
 * Named, typed value slot on an Object; supports optional min/max bounds,
 * usage hints, string enumerations, and a builder-pattern setter API.
 *
 * Example:
 *   obj.setParameter("radius", 0.5f);
 *   auto *p = obj.parameter("radius");
 *   p->setMin(0.f).setMax(10.f);
 *   float r = p->value().get<float>();
 */
struct Parameter
{
  Parameter(ParameterObserver *object, Token name);

  void remove(); // remove (and delete) parameter from parent object

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

  // Convenience methods for setting to a particular kind of parameter //

  void setToAttribute();

  ////////////////////////////////////////

  Parameter() = default;
  ~Parameter() = default;

  TSD_DEFAULT_MOVEABLE(Parameter)
  TSD_DEFAULT_COPYABLE(Parameter)

 private:
  friend struct Object;
  friend struct Scene;

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
constexpr anari::DataType anariType()
{
  return anari::ANARITypeFor<T>::value;
}

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline void Parameter::operator=(T newValue)
{
  setValue(newValue);
}

} // namespace tsd::scene

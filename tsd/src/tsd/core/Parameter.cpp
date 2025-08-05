// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Parameter.hpp"

namespace tsd::core {

Parameter::Parameter(ParameterObserver *object, Token name)
    : m_observer(object), m_name(name)
{}

void Parameter::remove()
{
  if (m_observer)
    m_observer->removeParameter(this);
}

Token Parameter::name() const
{
  return m_name;
}

const std::string &Parameter::description() const
{
  return m_description;
}

bool Parameter::isEnabled() const
{
  return m_enabled;
}

Parameter &Parameter::setDescription(const char *d)
{
  m_description = d;
  return *this;
}

Parameter &Parameter::setValue(const Any &newValue)
{
  m_value = newValue;
  if (m_observer)
    m_observer->parameterChanged(this);
  return *this;
}

Parameter &Parameter::setMin(const Any &newMin)
{
  m_min = newMin;
  return *this;
}

Parameter &Parameter::setMax(const Any &newMax)
{
  m_max = newMax;
  return *this;
}

Parameter &Parameter::setStringValues(const std::vector<std::string> &sv)
{
  m_stringValues = sv;
  return *this;
}

Parameter &Parameter::setStringSelection(int s)
{
  m_stringSelection = s;
  return *this;
}

Parameter &Parameter::setUsage(ParameterUsageHint u)
{
  m_usageHint = u;
  return *this;
}

Parameter &Parameter::setEnabled(bool enabled)
{
  m_enabled = enabled;
  if (m_observer)
    m_observer->parameterChanged(this);
  return *this;
}

const Any &Parameter::value() const
{
  return m_value;
}

ParameterUsageHint Parameter::usage() const
{
  return m_usageHint;
}

const Any &Parameter::min() const
{
  return m_min;
}

const Any &Parameter::max() const
{
  return m_max;
}

bool Parameter::hasMin() const
{
  return min().type() == value().type();
}

bool Parameter::hasMax() const
{
  return max().type() == value().type();
}

const std::vector<std::string> &Parameter::stringValues() const
{
  return m_stringValues;
}

int Parameter::stringSelection() const
{
  return m_stringSelection;
}

void Parameter::setObserver(ParameterObserver *o)
{
  m_observer = o;
}

} // namespace tsd::core

// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Parameter.hpp"

namespace tsd {

Parameter::Parameter(ParameterObserver *object,
    Token name,
    ANARIDataType type,
    const void *v,
    ParameterUsageHint usage)
    : m_observer(object), m_name(name), m_usageHint(usage)
{
  setValue(Any(type, v));
}

Token Parameter::name() const
{
  return m_name;
}

const std::string &Parameter::description() const
{
  return m_description;
}

void Parameter::setDescription(const char *d)
{
  m_description = d;
}

const void *Parameter::data() const
{
  return m_value.data();
}

const Any &Parameter::value() const
{
  return m_value;
}

void Parameter::setValue(const Any &newValue)
{
  m_value = newValue;
  if (m_observer)
    m_observer->parameterChanged(this);
}

ANARIDataType Parameter::type() const
{
  return m_value.type();
}

ParameterUsageHint Parameter::usage() const
{
  return m_usageHint;
}

void Parameter::setUsage(ParameterUsageHint u)
{
  m_usageHint = u;
}

void Parameter::setMin(const Any &newMin)
{
  m_min = newMin;
}

void Parameter::setMax(const Any &newMax)
{
  m_max = newMax;
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
  return m_min.type() == type();
}

bool Parameter::hasMax() const
{
  return m_max.type() == type();
}

const std::vector<std::string> &Parameter::stringValues() const
{
  return m_stringValues;
}

void Parameter::setStringValues(const std::vector<std::string> &sv)
{
  m_stringValues = sv;
}

int Parameter::stringSelection() const
{
  return m_stringSelection;
}

void Parameter::setStringSelection(int s)
{
  m_stringSelection = s;
}

void Parameter::setObserver(ParameterObserver *o)
{
  m_observer = o;
}

} // namespace tsd

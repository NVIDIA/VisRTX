// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Object.hpp"
// std
#include <iomanip>

namespace tsd {

namespace tokens {

Token none = "none";
Token unknown = "unknown";

} // namespace tokens

// Object definitions /////////////////////////////////////////////////////////

Object::Object(anari::DataType type, Token stype)
    : m_type(type), m_subtype(stype)
{}

Object::Object(Object &&o)
{
  m_parameters = std::move(o.m_parameters);
  m_type = std::move(o.m_type);
  m_subtype = std::move(o.m_subtype);
  m_name = std::move(o.m_name);
  m_index = std::move(o.m_index);
  m_updateDelegate = std::move(o.m_updateDelegate);
  for (auto &p : m_parameters)
    p.second.setObserver(this);
}

Object &Object::operator=(Object &&o)
{
  m_parameters = std::move(o.m_parameters);
  m_type = std::move(o.m_type);
  m_subtype = std::move(o.m_subtype);
  m_name = std::move(o.m_name);
  m_index = std::move(o.m_index);
  m_updateDelegate = std::move(o.m_updateDelegate);
  for (auto &p : m_parameters)
    p.second.setObserver(this);
  return *this;
}

anari::DataType Object::type() const
{
  return m_type;
}

Token Object::subtype() const
{
  return m_subtype;
}

size_t Object::index() const
{
  return m_index;
}

const std::string &Object::name() const
{
  return m_name;
}

void Object::setName(const char *n)
{
  m_name = n;
}

Parameter &Object::addParameter(Token name)
{
  m_parameters.set(name, Parameter(this, name.c_str()));
  return *parameter(name);
}

void Object::setParameter(Token name, ANARIDataType type, const void *v)
{
  if (anari::isObject(type))
    return;

  auto *p = parameter(name);
  if (p)
    p->setValue({type, v});
  else {
    auto &np = addParameter(name);
    np.setValue({type, v});
  }
}

void Object::setParameterObject(Token name, const Object &obj)
{
  auto *p = parameter(name);
  if (p)
    p->setValue({obj.type(), obj.index()});
  else
    addParameter(name).setValue({obj.type(), obj.index()});
}

Parameter *Object::parameter(Token name)
{
  return m_parameters.at(name);
}

void Object::removeParameter(Token name)
{
  if (auto *p = parameter(name); p && m_updateDelegate)
    m_updateDelegate->signalParameterRemoved(this, p);
  m_parameters.erase(name);
}

void Object::removeAllParameters()
{
  for (size_t i = 0; i < numParameters(); i++) {
    if (m_updateDelegate)
      m_updateDelegate->signalParameterRemoved(this, &parameterAt(i));
  }
  m_parameters.clear();
}

size_t Object::numParameters() const
{
  return m_parameters.size();
}

const Parameter &Object::parameterAt(size_t i) const
{
  return m_parameters.at_index(i).second;
}

Parameter &Object::parameterAt(size_t i)
{
  return m_parameters.at_index(i).second;
}

const char *Object::parameterNameAt(size_t i) const
{
  return m_parameters.at_index(i).first.c_str();
}

anari::Object Object::makeANARIObject(anari::Device) const
{
  return {};
}

void Object::updateANARIParameter(anari::Device d,
    anari::Object o,
    const Parameter &p,
    const char *n,
    AnariObjectCache *cache) const
{
  if (cache && p.value().holdsObject()) {
    auto objType = p.value().type();
    auto objHandle = cache->getHandle(objType, p.value().getAsObjectIndex());
    anari::setParameter(d, o, n, objType, &objHandle);
  } else if (!p.value().holdsObject()) {
    if (p.value().type() == ANARI_FLOAT32_VEC2
        && p.usage() & ParameterUsageHint::DIRECTION) {
      const auto azel = p.value().get<float2>();
      const float az = math::radians(azel.x);
      const float el = math::radians(azel.y);
      anari::setParameter(d,
          o,
          n,
          float3(std::sin(az) * std::cos(el),
              std::sin(el),
              std::cos(az) * std::cos(el)));
    } else {
      anari::setParameter(d, o, n, p.value().type(), p.value().data());
    }
  }
}

void Object::updateAllANARIParameters(
    anari::Device d, anari::Object o, AnariObjectCache *cache) const
{
  for (size_t i = 0; i < numParameters(); i++)
    updateANARIParameter(d, o, parameterAt(i), parameterNameAt(i), cache);
}

void Object::setUpdateDelegate(BaseUpdateDelegate *ud)
{
  m_updateDelegate = ud;
}

void Object::parameterChanged(const Parameter *p)
{
  if (m_updateDelegate)
    m_updateDelegate->signalParameterUpdated(this, p);
}

BaseUpdateDelegate *Object::updateDelegate() const
{
  return m_updateDelegate;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void print(const Object &obj, std::ostream &out)
{
  out << "Object -- '" << obj.name() << "'\n";
  out << "     type : " << anari::toString(obj.type()) << '\n';
  if (!obj.subtype().empty())
    out << "  subtype : " << obj.subtype().c_str() << '\n';

  out << "\nparameters(" << obj.numParameters() << "):\n";
  for (int i = 0; i < obj.numParameters(); i++) {
    auto &p = obj.parameterAt(i);
    auto *name = obj.parameterNameAt(i);
    out << std::setw(20) << name << "\t| " << anari::toString(p.value().type())
        << '\n';
  }
}

} // namespace tsd

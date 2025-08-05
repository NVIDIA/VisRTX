// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/Object.hpp"
#include "tsd/core/AnariObjectCache.hpp"
// std
#include <iomanip>

namespace tsd::core {

namespace tokens {

Token none = "none";
Token unknown = "unknown";

} // namespace tokens

// Helper functions ///////////////////////////////////////////////////////////

static Any parseValue(ANARIDataType type, const void *mem)
{
  if (type == ANARI_STRING)
    return Any(ANARI_STRING, "");
  else if (anari::isObject(type)) {
    ANARIObject nullHandle = ANARI_INVALID_HANDLE;
    return Any(type, &nullHandle);
  } else if (mem)
    return Any(type, mem);
  else
    return {};
}

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
  m_context = std::move(o.m_context);
  m_index = std::move(o.m_index);
  m_updateDelegate = std::move(o.m_updateDelegate);
  m_metadata = std::move(o.m_metadata);
  for (auto &p : m_parameters)
    p.second.setObserver(this);
}

Object &Object::operator=(Object &&o)
{
  m_parameters = std::move(o.m_parameters);
  m_type = std::move(o.m_type);
  m_subtype = std::move(o.m_subtype);
  m_name = std::move(o.m_name);
  m_context = std::move(o.m_context);
  m_index = std::move(o.m_index);
  m_updateDelegate = std::move(o.m_updateDelegate);
  m_metadata = std::move(o.m_metadata);
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

Context *Object::context() const
{
  return m_context;
}

const std::string &Object::name() const
{
  return m_name;
}

void Object::setName(const char *n)
{
  m_name = n;
}

Any Object::getMetadataValue(const std::string &name) const
{
  if (!m_metadata)
    return {};
  else if (const auto *c = m_metadata->root().child(name); c != nullptr)
    return c->getValue();
  else
    return {};
}

void Object::getMetadataArray(const std::string &name,
    anari::DataType *type,
    const void **ptr,
    size_t *size) const
{
  *type = ANARI_UNKNOWN;
  *ptr = nullptr;
  *size = 0;
  if (!m_metadata)
    return;
  if (const auto *c = m_metadata->root().child(name); c != nullptr)
    c->getValueAsArray(type, ptr, size);
}

void Object::setMetadataValue(const std::string &name, Any v)
{
  initMetadata();
  m_metadata->root().append(name) = v;
}

void Object::setMetadataArray(const std::string &name,
    anari::DataType type,
    const void *v,
    size_t numElements)
{
  initMetadata();
  m_metadata->root().append(name).setValueAsArray(type, v, numElements);
}

void Object::removeMetadata(const std::string &name)
{
  if (!m_metadata)
    return;
  m_metadata->root().remove(name);
}

size_t Object::numMetadata() const
{
  if (!m_metadata)
    return 0;
  return m_metadata->root().numChildren();
}

const char *Object::getMetadataName(size_t i) const
{
  if (!m_metadata)
    return "";
  if (const auto *c = m_metadata->root().child(i); c != nullptr)
    return c->name().c_str();
  else
    return "";
}

Parameter &Object::addParameter(Token name)
{
  m_parameters.set(name, Parameter(this, name.c_str()));
  return *parameter(name);
}

Parameter *Object::setParameter(Token name, ANARIDataType type, const void *v)
{
  if (anari::isObject(type))
    return nullptr;

  auto *p = parameter(name);
  if (p)
    p->setValue({type, v});
  else {
    p = &(addParameter(name));
    p->setValue({type, v});
  }
  return p;
}

Parameter *Object::setParameterObject(Token name, const Object &obj)
{
  auto *p = parameter(name);
  if (p)
    p->setValue({obj.type(), obj.index()});
  else {
    p = &(addParameter(name));
    p->setValue({obj.type(), obj.index()});
  }
  return p;
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
  if (!o)
    return;

  if (cache && !p.isEnabled()) {
    anari::unsetParameter(d, o, n);
  } else if (cache && p.value().holdsObject()) {
    auto objType = p.value().type();
    auto objHandle =
        cache->getHandle(objType, p.value().getAsObjectIndex(), true);
    if (objHandle)
      anari::setParameter(d, o, n, objType, &objHandle);
    else
      anari::unsetParameter(d, o, n);
  } else if (!p.value().holdsObject()) {
    if (p.value().type() == ANARI_FLOAT32_VEC2
        && p.usage() & ParameterUsageHint::DIRECTION) {
      anari::setParameter(d, o, n, math::azelToDir(p.value().get<float2>()));
    } else if (p.value().type() == ANARI_FLOAT32_VEC2
        && p.usage() & ParameterUsageHint::VALUE_RANGE_TRANSFORM) {
      anari::setParameter(
          d, o, n, math::makeValueRangeTransform(p.value().get<float2>()));
    } else {
      anari::setParameter(d, o, n, p.value().type(), p.value().data());
    }
  }
}

void Object::updateAllANARIParameters(
    anari::Device d, anari::Object o, AnariObjectCache *cache) const
{
  if (!o)
    return;

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

void Object::removeParameter(const Parameter *p)
{
  removeParameter(p->name());
}

BaseUpdateDelegate *Object::updateDelegate() const
{
  return m_updateDelegate;
}

void Object::initMetadata() const
{
  if (!m_metadata)
    m_metadata = std::make_unique<core::DataTree>();
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

std::vector<std::string> getANARIObjectSubtypes(
    anari::Device d, anari::DataType type)
{
  if (!anari::isObject(type))
    return {};

  const char **r_subtypes = anariGetObjectSubtypes(d, type);

  std::vector<std::string> retval;
  if (r_subtypes != nullptr) {
    for (int i = 0; r_subtypes[i] != nullptr; i++)
      retval.push_back(r_subtypes[i]);
  } else if (type == ANARI_RENDERER)
    retval.emplace_back("default");

  return retval;
}

Object parseANARIObjectInfo(
    anari::Device d, ANARIDataType objectType, const char *subtype)
{
  Object retval(objectType, subtype);

  if (objectType == ANARI_RENDERER) {
    retval.addParameter("background")
        .setValue(float4(0.05f, 0.05f, 0.05f, 1.f))
        .setDescription("background color")
        .setUsage(ParameterUsageHint::COLOR);
    retval.addParameter("ambientRadiance")
        .setValue(0.25f)
        .setDescription("intensity of ambient light")
        .setMin(0.f);
    retval.addParameter("ambientColor")
        .setValue(float3(1.f))
        .setDescription("color of ambient light")
        .setUsage(ParameterUsageHint::COLOR);
  }

  auto *parameter = (const ANARIParameter *)anariGetObjectInfo(
      d, objectType, subtype, "parameter", ANARI_PARAMETER_LIST);

  for (; parameter && parameter->name != nullptr; parameter++) {
    tsd::core::Token name(parameter->name);
    if (retval.parameter(name))
      continue;

    auto *description = (const char *)anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "description",
        ANARI_STRING);

    const void *defaultValue = anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "default",
        parameter->type);

    const void *minValue = anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "minimum",
        parameter->type);

    const void *maxValue = anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "maximum",
        parameter->type);

    const auto **stringValues = (const char **)anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "value",
        ANARI_STRING_LIST);

    auto &p = retval.addParameter(name);
    p.setValue(Any(parameter->type, nullptr));
    p.setDescription(description ? description : "");
    p.setValue(parseValue(parameter->type, defaultValue));
    if (minValue)
      p.setMin(parseValue(parameter->type, minValue));
    if (maxValue)
      p.setMax(parseValue(parameter->type, maxValue));

    std::vector<std::string> svs;
    for (; stringValues && *stringValues; stringValues++)
      svs.push_back(*stringValues);
    if (!svs.empty())
      p.setStringValues(svs);
  }

  return retval;
}

} // namespace tsd::core

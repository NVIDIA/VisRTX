// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Object.h"
// std
#include <algorithm>

namespace tsd_device {

Object::Object(anari::DataType type, DeviceGlobalState *s, tsd::Token subtype)
    : helium::BaseObject(type, s)
{
  switch (type) {
  case ANARI_SURFACE:
    m_object = s->ctx.createObject<tsd::Surface>().data();
    break;
  case ANARI_GEOMETRY:
    m_object = s->ctx.createObject<tsd::Geometry>(subtype).data();
    break;
  case ANARI_MATERIAL:
    m_object = s->ctx.createObject<tsd::Material>(subtype).data();
    break;
  case ANARI_SAMPLER:
    m_object = s->ctx.createObject<tsd::Sampler>(subtype).data();
    break;
  case ANARI_VOLUME:
    m_object = s->ctx.createObject<tsd::Volume>(subtype).data();
    break;
  case ANARI_SPATIAL_FIELD:
    m_object = s->ctx.createObject<tsd::SpatialField>(subtype).data();
    break;
  case ANARI_LIGHT:
    m_object = s->ctx.createObject<tsd::Light>(subtype).data();
    break;
  default:
    // no-op
    break;
  }
}

void Object::commitParameters()
{
  if (!m_object) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "no equivalent TSD object present during commit() for %s",
        anari::toString(type()));
    return;
  }
#if 0 // for now let removed parameters persist
  m_object->removeAllParameters();
#endif
  std::for_each(params_begin(), params_end(), [&](auto &p) {
    if (anari::isObject(p.second.type())) {
      auto *obj = p.second.template getObject<Object>();
      if (obj)
        m_object->setParameterObject(tsd::Token(p.first), *obj->tsdObject());
    } else {
      m_object->setParameter(
          tsd::Token(p.first), p.second.type(), p.second.data());
    }
  });
}

void Object::finalize()
{
  // no-op
}

bool Object::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  return false;
}

bool Object::isValid() const
{
  return true;
}

DeviceGlobalState *Object::deviceState() const
{
  return (DeviceGlobalState *)helium::BaseObject::m_state;
}

tsd::Object *Object::tsdObject() const
{
  return m_object;
}

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_DEFINITION(tsd_device::Object *);

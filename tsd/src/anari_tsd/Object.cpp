// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Object.h"
#include "Array.h"
// std
#include <algorithm>

namespace tsd_device {

// Object definitions /////////////////////////////////////////////////////////

Object::Object(anari::DataType type, DeviceGlobalState *s)
    : helium::BaseObject(type, s)
{
  // no-op
}

bool Object::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  return false;
}

void Object::commitParameters()
{
  // no-op
}

void Object::finalize()
{
  // no-op
}

bool Object::isValid() const
{
  return true;
}

DeviceGlobalState *Object::deviceState() const
{
  return (DeviceGlobalState *)helium::BaseObject::m_state;
}

// TSDObject definitions //////////////////////////////////////////////////////

TSDObject::TSDObject(
    anari::DataType type, DeviceGlobalState *s, tsd::core::Token subtype)
    : Object(type, s)
{
  tsd::core::Object *obj = nullptr;
  std::string name;
  switch (type) {
  case ANARI_CAMERA:
    obj = s->scene.createObject<tsd::core::Camera>(subtype).data();
    name = "camera" + std::to_string(s->cameraCount++);
    break;
  case ANARI_SURFACE:
    obj = s->scene.createObject<tsd::core::Surface>().data();
    name = "surface" + std::to_string(s->surfaceCount++);
    break;
  case ANARI_GEOMETRY:
    obj = s->scene.createObject<tsd::core::Geometry>(subtype).data();
    name = "geometry" + std::to_string(s->geometryCount++);
    break;
  case ANARI_MATERIAL:
    obj = s->scene.createObject<tsd::core::Material>(subtype).data();
    name = "material" + std::to_string(s->materialCount++);
    break;
  case ANARI_SAMPLER:
    obj = s->scene.createObject<tsd::core::Sampler>(subtype).data();
    name = "sampler" + std::to_string(s->samplerCount++);
    break;
  case ANARI_VOLUME:
    obj = s->scene.createObject<tsd::core::Volume>(subtype).data();
    name = "volume" + std::to_string(s->volumeCount++);
    break;
  case ANARI_SPATIAL_FIELD:
    obj = s->scene.createObject<tsd::core::SpatialField>(subtype).data();
    name = "field" + std::to_string(s->fieldCount++);
    break;
  case ANARI_LIGHT:
    obj = s->scene.createObject<tsd::core::Light>(subtype).data();
    name = "light" + std::to_string(s->lightCount++);
    break;
  default:
    break;
  }

  if (!obj) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "failed to create equivalent TSD object for %s",
        anari::toString(type));
  } else {
    m_object = tsd::core::Any(obj->type(), obj->index());
    obj->setName(name.c_str());
  }
}

void TSDObject::commitParameters()
{
  auto *object = tsdObject();
  if (!object) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no equivalent TSD object present during commit() for %s",
        anari::toString(type()));
    return;
  }
#if 0 // for now let removed parameters persist
  object->removeAllParameters();
#endif
  std::for_each(params_begin(), params_end(), [&](auto &p) {
    if (anari::isObject(p.second.type())) {
      if (anari::isArray(p.second.type())) {
        auto *arr = p.second.template getObject<Array>();
        if (arr) {
          object->setParameterObject(
              tsd::core::Token(p.first), *arr->tsdObject());
        }
      } else {
        auto *obj = p.second.template getObject<TSDObject>();
        if (obj && obj->tsdObject() != nullptr) {
          object->setParameterObject(
              tsd::core::Token(p.first), *obj->tsdObject());
        }
      }
    } else if (p.second.type() != ANARI_UNKNOWN) {
      object->setParameter(
          tsd::core::Token(p.first), p.second.type(), p.second.data());
    } else {
      reportMessage(ANARI_SEVERITY_WARNING,
          "skip setting parameter '%s' of unknown type",
          p.first.c_str());
    }
  });
}

tsd::core::Object *TSDObject::tsdObject() const
{
  return deviceState()->scene.getObject(m_object);
}

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_DEFINITION(tsd_device::TSDObject *);

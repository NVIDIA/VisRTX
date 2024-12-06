// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Array.h"
#include "Object.h"

namespace tsd_device {

Array::Array(DeviceGlobalState *state, Array1DMemoryDescriptor desc)
    : helium::BaseArray(ANARI_ARRAY1D, state),
      m_appMemory(desc.appMemory),
      m_deleter(desc.deleter),
      m_deleterPtr(desc.deleterPtr)
{
  m_array = state->scene.createArray(desc.elementType, desc.numItems);
  syncSharedData();
}

Array::Array(DeviceGlobalState *state, Array2DMemoryDescriptor desc)
    : helium::BaseArray(ANARI_ARRAY2D, state),
      m_appMemory(desc.appMemory),
      m_deleter(desc.deleter),
      m_deleterPtr(desc.deleterPtr)
{
  m_array = state->scene.createArray(
      desc.elementType, desc.numItems1, desc.numItems2);
  syncSharedData();
}

Array::Array(DeviceGlobalState *state, Array3DMemoryDescriptor desc)
    : helium::BaseArray(ANARI_ARRAY3D, state),
      m_appMemory(desc.appMemory),
      m_deleter(desc.deleter),
      m_deleterPtr(desc.deleterPtr)
{
  m_array = state->scene.createArray(
      desc.elementType, desc.numItems1, desc.numItems2, desc.numItems3);
  syncSharedData();
}

Array::~Array()
{
  freeAppMemory();
}

bool Array::isShared() const
{
  return m_appMemory != nullptr;
}

void *Array::map()
{
  return const_cast<void *>(isShared() ? m_appMemory : m_array->map());
}

void Array::unmap()
{
  syncSharedData();
  if (!isShared())
    m_array->unmap();
}

void Array::privatize()
{
  freeAppMemory();
}

void Array::commitParameters()
{
  auto *object = tsdObject();
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

void Array::finalize()
{
  // no-op
}

bool Array::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  // no-op
  return 0;
}

tsd::core::Object *Array::tsdObject() const
{
  return m_array.data();
}

DeviceGlobalState *Array::deviceState() const
{
  return (DeviceGlobalState *)m_state;
}

void Array::syncSharedData()
{
  if (isShared())
    m_array->setData(m_appMemory);
}

void Array::freeAppMemory()
{
  if (isShared() && m_deleter != nullptr)
    m_deleter(m_deleterPtr, m_appMemory);

  m_appMemory = nullptr;
  m_deleter = nullptr;
  m_deleterPtr = nullptr;
}

} // namespace tsd_device

TSD_DEVICE_ANARI_TYPEFOR_DEFINITION(tsd_device::Array *);

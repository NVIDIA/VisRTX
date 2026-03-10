// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/TimeSamples.hpp"
#include <anari/frontend/anari_enums.h>

#include "tsd/core/Logging.hpp"

namespace tsd::animation {

using core::logWarning;

TimeSamples::TimeSamples(anari::DataType elementType, size_t items)
    : m_elementType(elementType), m_size(items)
{
  if (isEmpty()) {
    logWarning("TimeSamples of %s elements created with 0 size",
        anari::toString(this->elementType()));
    return;
  }

  m_data = std::malloc(size() * elementSize());
}

TimeSamples::~TimeSamples()
{
  freeMemory();
}

size_t TimeSamples::size() const
{
  return m_size;
}

size_t TimeSamples::elementSize() const
{
  return anari::sizeOf(m_elementType);
}

anari::DataType TimeSamples::elementType() const
{
  return m_elementType;
}

bool TimeSamples::isEmpty() const
{
  return size() == 0;
}

void *TimeSamples::map()
{
  decObjectUseCounts();
  m_mapped = true;
  return m_data;
}

const void *TimeSamples::data() const
{
  return m_data;
}

const void *TimeSamples::elementAt(size_t i) const
{
  if (i >= size()) {
    logWarning("TimeSamples::elementAt() - index out of bounds");
    return nullptr;
  }
  auto *bytes = static_cast<const uint8_t *>(m_data);
  return bytes + i * elementSize();
}

void TimeSamples::unmap()
{
  m_mapped = false;
  incObjectUseCounts();
}

void TimeSamples::setData(const void *data, size_t byteOffset)
{
  auto *bytes = (const uint8_t *)data;
  std::memcpy(map(), bytes + byteOffset, size() * elementSize());
  unmap();
}

size_t TimeSamples::setData(std::FILE *stream)
{
  if (!stream)
    return 0;

  auto r = std::fread(map(), elementSize(), size(), stream);
  unmap();
  return r;
}

TimeSamples::TimeSamples(const TimeSamples &o)
    : m_elementType(o.m_elementType), m_size(o.m_size)
{
  if (o.m_data && !o.isEmpty()) {
    size_t bytes = size() * elementSize();
    m_data = std::malloc(bytes);
    std::memcpy(m_data, o.m_data, bytes);
    incObjectUseCounts();
  }
}

TimeSamples &TimeSamples::operator=(const TimeSamples &o)
{
  if (this != &o) {
    freeMemory();
    m_elementType = o.m_elementType;
    m_size = o.m_size;
    m_mapped = false;
    m_data = nullptr;
    if (o.m_data && !o.isEmpty()) {
      size_t bytes = size() * elementSize();
      m_data = std::malloc(bytes);
      std::memcpy(m_data, o.m_data, bytes);
      incObjectUseCounts();
    }
  }
  return *this;
}

TimeSamples::TimeSamples(TimeSamples &&o)
{
  m_data = o.m_data;
  m_elementType = o.m_elementType;
  m_size = o.m_size;
  m_mapped = o.m_mapped;
  o.m_data = nullptr;
}

TimeSamples &TimeSamples::operator=(TimeSamples &&o)
{
  if (this != &o) {
    freeMemory();
    m_data = o.m_data;
    m_elementType = o.m_elementType;
    m_size = o.m_size;
    m_mapped = o.m_mapped;
    o.m_data = nullptr;
  }
  return *this;
}

bool TimeSamples::holdsObjects() const
{
  return m_data && !isEmpty() && anari::isObject(m_elementType);
}

void TimeSamples::incObjectUseCounts()
{
  if (!holdsObjects())
    return;
  auto **objects = static_cast<scene::Object **>(m_data);
  for (size_t i = 0; i < m_size; i++) {
    if (objects[i])
      objects[i]->incUseCount(scene::Object::UseKind::ANIM);
  }
}

void TimeSamples::decObjectUseCounts()
{
  if (!holdsObjects())
    return;
  auto **objects = static_cast<scene::Object **>(m_data);
  for (size_t i = 0; i < m_size; i++) {
    if (objects[i])
      objects[i]->decUseCount(scene::Object::UseKind::ANIM);
  }
}

void TimeSamples::freeMemory()
{
  decObjectUseCounts();
  std::free(m_data);
  m_data = nullptr;
}

} // namespace tsd::animation

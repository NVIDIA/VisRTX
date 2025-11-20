// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_CUDA
#define TSD_USE_CUDA 1
#endif

#include "tsd/core/scene/objects/Array.hpp"

#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/Scene.hpp"
// std
#include <stdexcept>
#if TSD_USE_CUDA
// cuda
#include <cuda_runtime.h>
#endif

static void noopANARIDeleter(const void *, const void *)
{
  // do nothing
}

namespace tsd::core {

Array::Array(anari::DataType type, size_t items0, Array::MemoryKind kind)
    : Array(ANARI_ARRAY1D, type, items0, 1, 1, kind)
{}

Array::Array(
    anari::DataType type, size_t items0, size_t items1, Array::MemoryKind kind)
    : Array(ANARI_ARRAY2D, type, items0, items1, 1, kind)
{}

Array::Array(anari::DataType type,
    size_t items0,
    size_t items1,
    size_t items2,
    Array::MemoryKind kind)
    : Array(ANARI_ARRAY3D, type, items0, items1, items2, kind)
{}

Array::~Array()
{
  if (m_data) {
#if TSD_USE_CUDA
    if (kind() == MemoryKind::CUDA)
      cudaMalloc(&m_data, size() * elementSize());
    else
#endif
      std::free(m_data);
  }
}

size_t Array::size() const
{
  return dim(0) * dim(1) * dim(2);
}

size_t Array::elementSize() const
{
  return anari::sizeOf(m_elementType);
}

bool Array::isEmpty() const
{
  return size() == 0;
}

Array::MemoryKind Array::kind() const
{
  return m_kind;
}

size_t Array::dim(size_t d) const
{
  if (d == 0)
    return m_dim0;
  else if (d == 1)
    return m_dim1;
  else if (d == 2)
    return m_dim2;

  return 0;
}

anari::DataType Array::elementType() const
{
  return m_elementType;
}

void *Array::map()
{
  m_mapped = true;
  if (auto *ud = updateDelegate(); ud != nullptr)
    ud->signalArrayMapped(this);
  return m_data;
}

const void *Array::data() const
{
  return m_data;
}

const void *Array::elementAt(size_t i) const
{
  if (i >= size()) {
    logWarning("Array::elementAt() - index out of bounds");
    return nullptr;
  }
  auto *bytes = static_cast<const uint8_t *>(m_data);
  return bytes + i * elementSize();
}

void Array::unmap()
{
  m_mapped = false;
  if (auto *ud = updateDelegate(); ud != nullptr)
    ud->signalArrayUnmapped(this);
}

void Array::setData(const void *data, size_t byteOffset)
{
  auto *bytes = (const uint8_t *)data;
  std::memcpy(map(), bytes + byteOffset, size() * elementSize());
  unmap();
}

size_t Array::setData(std::FILE *stream)
{
  if (!stream)
    return 0;
  auto r = std::fread(map(), elementSize(), size(), stream);
  unmap();
  return r;
}

IndexedVectorRef<Array> Array::self() const
{
  return scene() ? scene()->getObject<Array>(index())
                 : IndexedVectorRef<Array>{};
}

anari::Object Array::makeANARIObject(anari::Device d) const
{
  if (elementType() == ANARI_UNKNOWN || isEmpty())
    return nullptr;

  anari::Object retval = nullptr;

  const void *ptr = anari::isObject(elementType()) ? nullptr : m_data;
  switch (type()) {
  case ANARI_ARRAY1D:
    retval = anari::newArray1D(
        d, ptr, noopANARIDeleter, nullptr, elementType(), dim(0));
    break;
  case ANARI_ARRAY2D:
    retval = anari::newArray2D(
        d, ptr, noopANARIDeleter, nullptr, elementType(), dim(0), dim(1));
    break;
  case ANARI_ARRAY3D:
    retval = anari::newArray3D(d,
        ptr,
        noopANARIDeleter,
        nullptr,
        elementType(),
        dim(0),
        dim(1),
        dim(2));
    break;
  default:
    break;
  }

  assert(retval != nullptr);
  return retval;
}

Array::Array(Array &&o) : Object(std::move(static_cast<Object &&>(o)))
{
  m_data = o.m_data;
  m_kind = o.m_kind;
  m_elementType = o.m_elementType;
  m_dim0 = o.m_dim0;
  m_dim1 = o.m_dim1;
  m_dim2 = o.m_dim2;
  m_mapped = o.m_mapped;
  o.m_data = nullptr;
}

Array &Array::operator=(Array &&o)
{
  if (this != &o) {
    *static_cast<Object *>(this) = std::move(*static_cast<Object *>(&o));
    m_data = o.m_data;
    m_kind = o.m_kind;
    m_elementType = o.m_elementType;
    m_dim0 = o.m_dim0;
    m_dim1 = o.m_dim1;
    m_dim2 = o.m_dim2;
    m_mapped = o.m_mapped;
    o.m_data = nullptr;
  }
  return *this;
}

Array::Array(anari::DataType arrayType,
    anari::DataType type,
    size_t items0,
    size_t items1,
    size_t items2,
    MemoryKind kind)
    : Object(arrayType),
      m_kind(kind),
      m_elementType(type),
      m_dim0(items0),
      m_dim1(items1),
      m_dim2(items2)
{
  if (anari::isObject(type) && kind == MemoryKind::CUDA)
    throw std::runtime_error("cannot create CUDA arrays of objects!");

  if (isEmpty()) {
    logWarning("%s of %s elements created with 0 size",
        anari::toString(this->type()),
        anari::toString(this->elementType()));
    return;
  }

  if (kind == MemoryKind::CUDA) {
#if TSD_USE_CUDA
    cudaMalloc(&m_data, size() * elementSize());
#else
    throw std::runtime_error("CUDA support not enabled!");
#endif
  } else
    m_data = std::malloc(size() * elementSize());
}

} // namespace tsd::core

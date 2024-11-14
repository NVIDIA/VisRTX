// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/objects/Array.hpp"
// std
#include <exception>

namespace tsd {

Array::Array(anari::DataType type, size_t items0)
    : Array(ANARI_ARRAY1D, type, items0, 1, 1)
{
  m_shape = 1;
}

Array::Array(anari::DataType type, size_t items0, size_t items1)
    : Array(ANARI_ARRAY2D, type, items0, items1, 1)
{
  m_shape = 2;
}

Array::Array(anari::DataType type, size_t items0, size_t items1, size_t items2)
    : Array(ANARI_ARRAY3D, type, items0, items1, items2)
{
  m_shape = 3;
}

size_t Array::size() const
{
  return m_data.size() / elementSize();
}

size_t Array::elementSize() const
{
  return anari::sizeOf(m_elementType);
}

bool Array::isEmpty() const
{
  return size() == 0;
}

size_t Array::shape() const
{
  return m_shape;
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
  return m_data.data();
}

const void *Array::data() const
{
  return m_data.data();
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

anari::Object Array::makeANARIObject(anari::Device d) const
{
  if (elementType() == ANARI_UNKNOWN)
    return nullptr;

  const bool isObjectArray = anari::isObject(elementType());
  anari::Object retval = nullptr;

  switch (shape()) {
  case 1:
    retval = anariNewArray1D(d,
        isObjectArray ? nullptr : m_data.data(),
        nullptr,
        nullptr,
        elementType(),
        dim(0));
    break;
  case 2:
    retval = anariNewArray2D(d,
        isObjectArray ? nullptr : m_data.data(),
        nullptr,
        nullptr,
        elementType(),
        dim(0),
        dim(1));
    break;
  case 3:
    retval = anariNewArray3D(d,
        isObjectArray ? nullptr : m_data.data(),
        nullptr,
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

Array::Array(anari::DataType arrayType,
    anari::DataType type,
    size_t items0,
    size_t items1,
    size_t items2)
    : Object(arrayType),
      m_elementType(type),
      m_dim0(items0),
      m_dim1(items1),
      m_dim2(items2)
{
  if (anari::isObject(type))
    throw std::runtime_error("cannot create arrays of ANARI objects in TSD");
  m_data.resize(dim(0) * dim(1) * dim(2) * elementSize());
}

} // namespace tsd

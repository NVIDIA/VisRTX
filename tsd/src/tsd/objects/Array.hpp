// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Object.hpp"
// std
#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace tsd {

struct Array : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_ARRAY;

  Array(anari::DataType type, size_t items0);
  Array(anari::DataType type, size_t items0, size_t items1);
  Array(anari::DataType type, size_t items0, size_t items1, size_t items2);

  Array() = default;
  ~Array() override = default;

  size_t size() const;
  size_t elementSize() const;
  bool isEmpty() const;

  size_t shape() const;
  size_t dim(size_t d) const;

  anari::DataType elementType() const;

  void *map();
  const void *map() const;

  template <typename T>
  T *mapAs();
  template <typename T>
  const T *mapAs() const;

  void unmap();
  void unmap() const;

  template <typename T>
  void setData(const T *data, size_t size, size_t startOffset = 0);
  template <typename T>
  void setData(const std::vector<T> &data, size_t startOffset = 0);
  void setData(const void *data, size_t byteOffset = 0);

  anari::Object makeANARIObject(anari::Device d) const override;

  // Movable, not copyable
  Array(const Array &) = delete;
  Array &operator=(const Array &) = delete;
  Array(Array &&) = default;
  Array &operator=(Array &&) = default;

 private:
  Array(anari::DataType arrayType,
      anari::DataType type,
      size_t items0,
      size_t items1,
      size_t items2);

  std::vector<unsigned char> m_data;
  anari::DataType m_elementType{ANARI_UNKNOWN};
  size_t m_shape{0};
  size_t m_dim0{0};
  size_t m_dim1{0};
  size_t m_dim2{0};
  mutable bool m_mapped{false};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline T *Array::mapAs()
{
  assert(anari::ANARITypeFor<T>::value == elementType());
  return reinterpret_cast<T *>(map());
}

template <typename T>
inline const T *Array::mapAs() const
{
  assert(anari::ANARITypeFor<T>::value == elementType());
  return reinterpret_cast<const T *>(map());
}

template <typename T>
inline void Array::setData(const T *data, size_t size, size_t startOffset)
{
  auto *d = mapAs<T>();
  std::memcpy(d + startOffset, data, size * sizeof(T));
  unmap();
}

template <typename T>
inline void Array::setData(const std::vector<T> &data, size_t startOffset)
{
  setData(data.data(), data.size(), startOffset);
}

} // namespace tsd
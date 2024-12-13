// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Object.hpp"
// std
#include <cassert>
#include <cstddef>
#include <memory>

namespace tsd {

struct Array : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_ARRAY;
  // clang-format off
  enum class MemoryKind { HOST, CUDA };
  // clang-format on

  Array(
      anari::DataType type, size_t items0, MemoryKind kind = MemoryKind::HOST);
  Array(anari::DataType type,
      size_t items0,
      size_t items1,
      MemoryKind kind = MemoryKind::HOST);
  Array(anari::DataType type,
      size_t items0,
      size_t items1,
      size_t items2,
      MemoryKind kind = MemoryKind::HOST);

  Array() = default;
  ~Array() override;

  size_t size() const;
  size_t elementSize() const;
  bool isEmpty() const;

  MemoryKind kind() const;
  size_t shape() const;
  size_t dim(size_t d) const;

  anari::DataType elementType() const;

  void *map();
  template <typename T>
  T *mapAs();
  void unmap();

  const void *data() const;
  template <typename T>
  const T *dataAs() const;

  template <typename T>
  void setData(const T *data, size_t size, size_t startOffset = 0);
  template <typename T>
  void setData(const std::vector<T> &data, size_t startOffset = 0);
  void setData(const void *data, size_t byteOffset = 0);

  anari::Object makeANARIObject(anari::Device d) const override;

  // Movable, not copyable
  Array(const Array &) = delete;
  Array &operator=(const Array &) = delete;
  Array(Array &&);
  Array &operator=(Array &&);

 private:
  Array(anari::DataType arrayType,
      anari::DataType type,
      size_t items0,
      size_t items1,
      size_t items2,
      MemoryKind kind);

  void *m_data{nullptr};
  MemoryKind m_kind{MemoryKind::HOST};
  anari::DataType m_elementType{ANARI_UNKNOWN};
  size_t m_shape{0};
  size_t m_dim0{0};
  size_t m_dim1{0};
  size_t m_dim2{0};
  mutable bool m_mapped{false};
};

using ArrayRef = IndexedVectorRef<Array>;

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline T *Array::mapAs()
{
  assert(anari::ANARITypeFor<T>::value == elementType());
  return reinterpret_cast<T *>(map());
}

template <typename T>
inline const T *Array::dataAs() const
{
  assert(anari::ANARITypeFor<T>::value == elementType());
  return reinterpret_cast<const T *>(data());
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
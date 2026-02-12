// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Object.hpp"
// std
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <memory>

namespace tsd::core {

struct Array : public Object
{
  // clang-format off
  enum class MemoryKind {
    HOST, // Memory allocated on the host (main memory)
    CUDA, // Memory allocated on the GPU (device memory)
    PROXY // No memory allocated, only a placeholder object (not mappable)
  };
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
  anari::DataType elementType() const;
  size_t dim(size_t d) const;
  bool isEmpty() const;

  MemoryKind kind() const;
  bool isHost() const;
  bool isCUDA() const;
  bool isProxy() const;

  void convertProxyToHost();

  void *map();
  template <typename T>
  T *mapAs();
  void unmap();

  const void *data() const;
  template <typename T>
  const T *dataAs() const;

  const void *elementAt(size_t i) const;

  template <typename T>
  void setData(const T *data, size_t size, size_t startOffset = 0);
  template <typename T>
  void setData(const std::vector<T> &data, size_t startOffset = 0);
  void setData(const void *data, size_t byteOffset = 0);
  size_t setData(std::FILE *stream);

  ObjectPoolRef<Array> self() const;

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
  void freeMemory();

  void *m_data{nullptr};
  MemoryKind m_kind{MemoryKind::HOST};
  anari::DataType m_elementType{ANARI_UNKNOWN};
  size_t m_dim0{0};
  size_t m_dim1{0};
  size_t m_dim2{0};
  mutable bool m_mapped{false};
};

using ArrayRef = ObjectPoolRef<Array>;

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline T *Array::mapAs()
{
  assert(sizeof(T) == anari::sizeOf(elementType()));
  return reinterpret_cast<T *>(map());
}

template <typename T>
inline const T *Array::dataAs() const
{
  assert(sizeof(T) == anari::sizeOf(elementType()));
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

} // namespace tsd::core
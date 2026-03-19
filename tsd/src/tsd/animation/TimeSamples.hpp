// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include <anari/anari_cpp.hpp>
// std
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace tsd::animation {

/*
 * Owning, type-aware buffer of keyframe values backed by a raw allocation;
 * supports map/unmap access, deep copy, and move semantics.
 *
 * Example:
 *   TimeSamples ts(ANARI_FLOAT32, 4);
 *   float *p = ts.mapAs<float>();
 *   p[0] = 0.f; p[1] = 1.f; p[2] = 2.f; p[3] = 3.f;
 *   ts.unmap();
 */
struct TimeSamples
{
  TimeSamples(anari::DataType elementType, size_t items);

  TimeSamples() = default;
  ~TimeSamples();

  size_t size() const;
  size_t elementSize() const;
  anari::DataType elementType() const;
  bool isEmpty() const;

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

  // Deep-copyable and movable
  TimeSamples(const TimeSamples &);
  TimeSamples &operator=(const TimeSamples &);
  TimeSamples(TimeSamples &&);
  TimeSamples &operator=(TimeSamples &&);

 private:
  void *m_data{nullptr};
  anari::DataType m_elementType{ANARI_UNKNOWN};
  size_t m_size{0};
  mutable bool m_mapped{false};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline T *TimeSamples::mapAs()
{
  assert(sizeof(T) == anari::sizeOf(elementType()));
  return reinterpret_cast<T *>(map());
}

template <typename T>
inline const T *TimeSamples::dataAs() const
{
  assert(sizeof(T) == anari::sizeOf(elementType()));
  return reinterpret_cast<const T *>(data());
}

template <typename T>
inline void TimeSamples::setData(const T *data, size_t size, size_t startOffset)
{
  auto *d = mapAs<T>();
  std::memcpy(d + startOffset, data, size * sizeof(T));
  unmap();
}

template <typename T>
inline void TimeSamples::setData(const std::vector<T> &data, size_t startOffset)
{
  setData(data.data(), data.size(), startOffset);
}

} // namespace tsd::animation

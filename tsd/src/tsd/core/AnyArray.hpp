// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TypeMacros.hpp"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tsd::core {

/*
 * Type-erased owning array of identically-typed ANARI elements.
 *
 * Analogous to Any for single values, but for flat buffers.  Storage is a
 * std::vector<uint8_t> so copy/move are automatically deep-copying and cheap.
 *
 * Example:
 *   AnyArray a(ANARI_FLOAT32, 4);
 *   a.get<float>(0) = 1.f;
 *   float v = a.get<float>(0);
 *
 *   AnyArray b(ANARI_FLOAT32_VEC3, 2, ptr);
 *   auto *p = b.dataAs<anari::math::float3>();
 */
struct AnyArray
{
  // Construction
  AnyArray() = default;
  AnyArray(anari::DataType type, size_t count);
  AnyArray(anari::DataType type, const void *src, size_t count);

  TSD_DEFAULT_COPYABLE(AnyArray)
  TSD_DEFAULT_MOVEABLE(AnyArray)

  // Type / capacity query
  anari::DataType elementType() const;
  size_t size() const;        // element count
  size_t elementSize() const; // bytes per element (anari::sizeOf(type))
  size_t byteSize() const;    // total bytes
  bool empty() const;
  bool valid() const;         // elementType() != ANARI_UNKNOWN
  operator bool() const;

  template <typename T>
  bool is() const;
  bool is(anari::DataType t) const;

  // Raw buffer access
  const void *data() const;
  void *data();

  // Per-element raw access
  const void *elementAt(size_t i) const;
  void *elementAt(size_t i);

  // Typed element access — throws std::runtime_error on type mismatch
  template <typename T>
  const T &get(size_t i) const;
  template <typename T>
  T &get(size_t i);

  // Typed pointer to whole buffer — asserts sizeof(T) == elementSize()
  template <typename T>
  const T *dataAs() const;
  template <typename T>
  T *dataAs();

  // Mutation
  void resize(size_t count);  // grows/shrinks, zero-fills new elements
  void reserve(size_t count); // pre-allocate without resize
  void reset();               // clear buffer and type

  void setElement(size_t i, const void *value);
  template <typename T>
  void setElement(size_t i, const T &value);

 private:
  std::vector<uint8_t> m_storage;
  anari::DataType m_type{ANARI_UNKNOWN};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline AnyArray::AnyArray(anari::DataType type, size_t count)
    : m_type(type)
{
  m_storage.resize(count * anari::sizeOf(type), 0);
}

inline AnyArray::AnyArray(anari::DataType type, const void *src, size_t count)
    : m_type(type)
{
  const size_t bytes = count * anari::sizeOf(type);
  m_storage.resize(bytes);
  if (src)
    std::memcpy(m_storage.data(), src, bytes);
}

inline anari::DataType AnyArray::elementType() const
{
  return m_type;
}

inline size_t AnyArray::elementSize() const
{
  return anari::sizeOf(m_type);
}

inline size_t AnyArray::size() const
{
  const size_t es = elementSize();
  return es ? m_storage.size() / es : 0;
}

inline size_t AnyArray::byteSize() const
{
  return m_storage.size();
}

inline bool AnyArray::empty() const
{
  return m_storage.empty();
}

inline bool AnyArray::valid() const
{
  return m_type != ANARI_UNKNOWN;
}

inline AnyArray::operator bool() const
{
  return valid();
}

template <typename T>
inline bool AnyArray::is() const
{
  return is(anari::ANARITypeFor<T>::value);
}

inline bool AnyArray::is(anari::DataType t) const
{
  return m_type == t;
}

inline const void *AnyArray::data() const
{
  return m_storage.data();
}

inline void *AnyArray::data()
{
  return m_storage.data();
}

inline const void *AnyArray::elementAt(size_t i) const
{
  return m_storage.data() + i * elementSize();
}

inline void *AnyArray::elementAt(size_t i)
{
  return m_storage.data() + i * elementSize();
}

template <typename T>
inline const T &AnyArray::get(size_t i) const
{
  constexpr anari::DataType type = anari::ANARITypeFor<T>::value;
  static_assert(type != ANARI_STRING,
      "AnyArray::get<T>() does not support string element types");
  if (m_type != type)
    throw std::runtime_error(
        "AnyArray::get<T>() called with mismatched element type");
  return *reinterpret_cast<const T *>(elementAt(i));
}

template <typename T>
inline T &AnyArray::get(size_t i)
{
  constexpr anari::DataType type = anari::ANARITypeFor<T>::value;
  static_assert(type != ANARI_STRING,
      "AnyArray::get<T>() does not support string element types");
  if (m_type != type)
    throw std::runtime_error(
        "AnyArray::get<T>() called with mismatched element type");
  return *reinterpret_cast<T *>(elementAt(i));
}

template <typename T>
inline const T *AnyArray::dataAs() const
{
  static_assert(
      anari::ANARITypeFor<T>::value != ANARI_STRING,
      "AnyArray::dataAs<T>() does not support string element types");
  assert(sizeof(T) == elementSize());
  return reinterpret_cast<const T *>(data());
}

template <typename T>
inline T *AnyArray::dataAs()
{
  static_assert(
      anari::ANARITypeFor<T>::value != ANARI_STRING,
      "AnyArray::dataAs<T>() does not support string element types");
  assert(sizeof(T) == elementSize());
  return reinterpret_cast<T *>(data());
}

inline void AnyArray::resize(size_t count)
{
  m_storage.resize(count * anari::sizeOf(m_type), 0);
}

inline void AnyArray::reserve(size_t count)
{
  m_storage.reserve(count * anari::sizeOf(m_type));
}

inline void AnyArray::reset()
{
  m_storage.clear();
  m_type = ANARI_UNKNOWN;
}

inline void AnyArray::setElement(size_t i, const void *value)
{
  std::memcpy(elementAt(i), value, elementSize());
}

template <typename T>
inline void AnyArray::setElement(size_t i, const T &value)
{
  constexpr anari::DataType type = anari::ANARITypeFor<T>::value;
  static_assert(type != ANARI_STRING,
      "AnyArray::setElement<T>() does not support string element types");
  if (m_type != type)
    throw std::runtime_error(
        "AnyArray::setElement<T>() called with mismatched element type");
  std::memcpy(elementAt(i), &value, sizeof(T));
}

} // namespace tsd::core

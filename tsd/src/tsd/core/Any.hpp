// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include <anari/anari_cpp.hpp>
// std
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

namespace tsd::core {

struct Any
{
  static constexpr size_t INVALID_INDEX = ~size_t(0);

  Any();
  Any(const Any &copy);
  Any(Any &&tmp);

  template <typename T>
  Any(T value);
  Any(bool value);
  explicit Any(size_t value);

  Any(ANARIDataType type, const void *v);
  Any(ANARIDataType type, size_t v = INVALID_INDEX); // only use for objects

  ~Any();

  Any &operator=(const Any &rhs);
  Any &operator=(Any &&rhs);

  template <typename T>
  Any &operator=(T rhs);

  bool operator==(const Any &rhs) const;
  bool operator!=(const Any &rhs) const;

  // Raw data access, note that string values will be limited in storage size
  const void *data() const;
  void *data();

  template <typename T>
  T get() const;
  template <typename T>
  T getAs(anari::DataType expectedHeldAnariType = ANARI_UNKNOWN) const;

  size_t getAsObjectIndex() const; // when storing object indices only

  std::string getString() const;
  const char *getCStr() const;
  void reserveString(size_t size);
  void resizeString(size_t size);

  template <typename T>
  bool is() const;
  bool is(ANARIDataType t) const;

  ANARIDataType type() const;
  bool holdsObject() const;

  bool valid() const;
  operator bool() const;
  void reset();

 private:
  template <typename T>
  T storageAs() const;

  constexpr static int MAX_LOCAL_STORAGE = 16 * sizeof(float);

  std::array<uint8_t, MAX_LOCAL_STORAGE> m_storage;
  std::string m_string;
  ANARIDataType m_type{ANARI_UNKNOWN};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline Any::Any()
{
  reset();
}

inline Any::Any(const Any &copy)
{
  std::memcpy(m_storage.data(), copy.m_storage.data(), m_storage.size());
  m_string = copy.m_string;
  m_type = copy.m_type;
}

inline Any::Any(Any &&tmp)
{
  std::memcpy(m_storage.data(), tmp.m_storage.data(), m_storage.size());
  m_string = std::move(tmp.m_string);
  m_type = tmp.m_type;
  tmp.m_type = ANARI_UNKNOWN;
}

template <typename T>
inline Any::Any(T value) : Any()
{
  constexpr auto type = anari::ANARITypeFor<T>::value;
  static_assert(type != ANARI_UNKNOWN, "unknown type used initialize tsd::Any");

  if constexpr (type == ANARI_STRING)
    m_string = value;
  else
    std::memcpy(m_storage.data(), &value, sizeof(value));

  m_type = type;
}

inline Any::Any(bool value)
{
  uint32_t v = value;
  *this = Any(ANARI_BOOL, &v);
}

inline Any::Any(size_t value)
{
  uint64_t v = value;
  *this = Any(ANARI_UINT64, &v);
}

inline Any::Any(ANARIDataType type, const void *v) : Any()
{
  m_type = type;
  if (v != nullptr) {
    if (type == ANARI_STRING)
      m_string = (const char *)v;
    else if (type == ANARI_VOID_POINTER)
      std::memcpy(m_storage.data(), &v, anari::sizeOf(type));
    else
      std::memcpy(m_storage.data(), v, anari::sizeOf(type));
  }
}

inline Any::Any(ANARIDataType type, size_t v) : Any()
{
  if (anari::isObject(type)) {
    m_type = type;
    std::memcpy(m_storage.data(), &v, anari::sizeOf(type));
  }
}

inline Any::~Any()
{
  reset();
}

inline Any &Any::operator=(const Any &rhs)
{
  reset();
  std::memcpy(m_storage.data(), rhs.m_storage.data(), m_storage.size());
  m_string = rhs.m_string;
  m_type = rhs.m_type;
  return *this;
}

inline Any &Any::operator=(Any &&rhs)
{
  reset();
  std::memcpy(m_storage.data(), rhs.m_storage.data(), m_storage.size());
  m_string = std::move(rhs.m_string);
  m_type = rhs.m_type;
  rhs.m_type = ANARI_UNKNOWN;
  return *this;
}

template <typename T>
inline Any &Any::operator=(T rhs)
{
  return *this = Any(rhs);
}

inline bool Any::operator==(const Any &rhs) const
{
  if (!valid() || !rhs.valid())
    return false;
  if (type() != rhs.type())
    return false;
  if (type() == ANARI_BOOL)
    return get<bool>() == rhs.get<bool>();
  else if (type() == ANARI_STRING)
    return m_string == rhs.m_string;
  else {
    return std::equal(m_storage.data(),
        m_storage.data() + ::anari::sizeOf(type()),
        rhs.m_storage.data());
  }
}

inline bool Any::operator!=(const Any &rhs) const
{
  return !(*this == rhs);
}

template <typename T>
inline T Any::get() const
{
  if (!is<T>())
    throw std::runtime_error("get() called with invalid type on tsd::Any");
  return getAs<T>(type());
}

template <typename T>
inline T Any::getAs(anari::DataType expectedType) const
{
  constexpr ANARIDataType type = anari::ANARITypeFor<T>::value;
  static_assert(
      !anari::isObject(type), "use Any::getObject() for getting objects");
  static_assert(
      type != ANARI_STRING, "use Any::getString() for getting strings");

  if (!valid())
    throw std::runtime_error("Any::getAs<>() called on empty tsd::Any");
  else if (expectedType != ANARI_UNKNOWN && this->type() != expectedType) {
    throw std::runtime_error(
        "Any::getAs<>() given a type that disagress with what is held");
  }

  return storageAs<T>();
}

inline const void *Any::data() const
{
  return type() == ANARI_STRING ? (const void *)m_string.data()
                                : (const void *)m_storage.data();
}

inline void *Any::data()
{
  return type() == ANARI_STRING ? (void *)m_string.data()
                                : (void *)m_storage.data();
}

inline size_t Any::getAsObjectIndex() const
{
  return holdsObject() ? storageAs<size_t>() : ~size_t(0);
}

template <typename T>
inline bool Any::is() const
{
  return is(anari::ANARITypeFor<T>::value);
}

template <>
inline bool Any::is<bool>() const
{
  return is(ANARI_BOOL);
}

inline bool Any::is(ANARIDataType t) const
{
  return type() == t;
}

inline bool Any::holdsObject() const
{
  return anari::isObject(this->type());
}

inline ANARIDataType Any::type() const
{
  return m_type;
}

inline bool Any::valid() const
{
  return type() != ANARI_UNKNOWN;
}

inline Any::operator bool() const
{
  return valid();
}

inline void Any::reset()
{
  std::fill(m_storage.begin(), m_storage.end(), 0);
  m_string.clear();
  m_type = ANARI_UNKNOWN;
}

template <typename T>
inline T Any::storageAs() const
{
  static_assert(sizeof(T) <= MAX_LOCAL_STORAGE, "Any: not enough storage");
  T retval;
  std::memcpy(&retval, m_storage.data(), sizeof(retval));
  return retval;
}

inline std::string Any::getString() const
{
  return type() == ANARI_STRING ? m_string : "";
}

inline const char *Any::getCStr() const
{
  return type() == ANARI_STRING ? m_string.c_str() : "";
}

inline void Any::reserveString(size_t size)
{
  m_string.reserve(size);
}

inline void Any::resizeString(size_t size)
{
  m_string.resize(size);
}

} // namespace tsd::core

// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <stack>
#include <vector>

namespace tsd::core {

constexpr size_t INVALID_INDEX = ~size_t(0);
#define TSD_INVALID_INDEX tsd::core::INVALID_INDEX

template <typename T>
struct IndexedVectorRef;

template <typename T>
struct IndexedVector
{
  using element_t = T;
  using storage_t = std::vector<element_t>;
  using marker_t = std::vector<bool>;
  using index_pool_t = std::stack<size_t>;

  IndexedVector() = default;
  IndexedVector(size_t reserveSize);
  ~IndexedVector() = default;

  T &operator[](size_t i) const; // raw access
  IndexedVectorRef<T> at(size_t i) const; // safe access

  size_t size() const;
  size_t capacity() const;
  bool empty() const;

  bool slot_empty(size_t i) const;
  float density() const;

  IndexedVectorRef<T> insert(T &&v);
  template <typename... Args>
  IndexedVectorRef<T> emplace(Args &&...args);
  bool erase(size_t i);

  void clear();
  void reserve(size_t size);

  bool defragment();

  template <typename U>
  void sync_slots(const IndexedVector<U> &o);

 private:
  template <typename U>
  friend struct IndexedVector;

  mutable storage_t m_values;
  marker_t m_slots;
  index_pool_t m_freeIndices;
};

template <typename T>
struct IndexedVectorRef
{
  IndexedVectorRef() = default;
  IndexedVectorRef(const IndexedVector<T> &iv, size_t idx);

  size_t index() const;

  const IndexedVector<T> &storage() const;

  T value_or(T alt) const;

  T *data();
  const T *data() const;

  T &operator*();
  const T &operator*() const;
  T *operator->();
  const T *operator->() const;

  bool valid() const;
  operator bool() const;

  IndexedVectorRef(const IndexedVectorRef &) = default;
  IndexedVectorRef &operator=(const IndexedVectorRef &) = default;
  IndexedVectorRef(IndexedVectorRef &&) = default;
  IndexedVectorRef &operator=(IndexedVectorRef &&) = default;

 private:
  template <typename U>
  friend bool operator==(
      const IndexedVectorRef<U> &a, const IndexedVectorRef<U> &b);

  size_t m_idx{INVALID_INDEX};
  const IndexedVector<T> *m_iv{nullptr};
};

template <typename T>
bool operator==(const IndexedVectorRef<T> &a, const IndexedVectorRef<T> &b);
template <typename T>
bool operator!=(const IndexedVectorRef<T> &a, const IndexedVectorRef<T> &b);

template <typename T, typename FCN_T>
inline void foreach_item_ref(IndexedVector<T> &iv, FCN_T &&fcn)
{
  for (size_t i = 0; i < iv.capacity(); i++)
    fcn(iv.at(i));
}

template <typename T, typename FCN_T>
inline void foreach_item(IndexedVector<T> &iv, FCN_T &&fcn)
{
  for (size_t i = 0; i < iv.capacity(); i++)
    fcn(iv.slot_empty(i) ? nullptr : &iv[i]);
}

template <typename T, typename FCN_T>
inline void foreach_item_const(const IndexedVector<T> &iv, FCN_T &&fcn)
{
  for (size_t i = 0; i < iv.capacity(); i++)
    fcn(iv.slot_empty(i) ? nullptr : &iv[i]);
}

template <typename T, typename FCN_T>
inline IndexedVectorRef<T> find_item_if(const IndexedVector<T> &iv, FCN_T &&fcn)
{
  for (size_t i = 0; i < iv.capacity(); i++) {
    if(fcn(iv.slot_empty(i) ? nullptr : &iv[i]))
      return iv.at(i);
  }
  return {};
}

// Inlined definitions ////////////////////////////////////////////////////////

// IndexedVector //

template <typename T>
inline IndexedVector<T>::IndexedVector(size_t reserveSize)
{
  reserve(reserveSize);
}

template <typename T>
inline T &IndexedVector<T>::operator[](size_t i) const
{
  return m_values[i];
}

template <typename T>
inline IndexedVectorRef<T> IndexedVector<T>::at(size_t i) const
{
  return i >= capacity() || slot_empty(i) ? IndexedVectorRef<T>{}
                                          : IndexedVectorRef<T>(*this, i);
}

template <typename T>
inline size_t IndexedVector<T>::size() const
{
  return m_values.size() - m_freeIndices.size();
}

template <typename T>
inline size_t IndexedVector<T>::capacity() const
{
  return m_values.size();
}

template <typename T>
inline bool IndexedVector<T>::empty() const
{
  return size() == 0;
}

template <typename T>
inline bool IndexedVector<T>::slot_empty(size_t i) const
{
  return !m_slots[i];
}

template <typename T>
inline float IndexedVector<T>::density() const
{
  return capacity() == 0 ? 1.f : 1.f - float(m_freeIndices.size()) / capacity();
}

template <typename T>
inline IndexedVectorRef<T> IndexedVector<T>::insert(T &&v)
{
  if (m_freeIndices.empty()) {
    m_values.emplace_back(std::move(v));
    m_slots.push_back(true);
    return at(m_values.size() - 1);
  } else {
    size_t i = m_freeIndices.top();
    m_freeIndices.pop();
    m_values[i] = std::move(v);
    m_slots[i] = true;
    return at(i);
  }
}

template <typename T>
template <typename... Args>
IndexedVectorRef<T> IndexedVector<T>::emplace(Args &&...args)
{
  return insert(std::move(T(std::forward<Args>(args)...)));
}

template <typename T>
inline bool IndexedVector<T>::erase(size_t i)
{
  if (slot_empty(i))
    return false;

  m_values[i] = {};
  m_slots[i] = false;
  m_freeIndices.push(i);

  return true;
}

template <typename T>
inline void IndexedVector<T>::clear()
{
  m_values.clear();
  m_slots.clear();
  m_freeIndices = {};
}

template <typename T>
inline void IndexedVector<T>::reserve(size_t size)
{
  m_values.reserve(size);
  m_slots.reserve(size);
}

template <typename T>
inline bool IndexedVector<T>::defragment()
{
  if (density() == 1.f)
    return false;

  auto p =
      std::stable_partition(m_values.begin(), m_values.end(), [&](auto &v) {
        size_t i = std::distance(&m_values[0], &v);
        return m_slots[i];
      });
  m_values.erase(p, m_values.end());
  m_slots.resize(m_values.size());
  std::fill(m_slots.begin(), m_slots.end(), true);
  while (!m_freeIndices.empty())
    m_freeIndices.pop();

  return true;
}

template <typename T>
template <typename U>
inline void IndexedVector<T>::sync_slots(const IndexedVector<U> &o)
{
  m_slots = o.m_slots;
  m_freeIndices = o.m_freeIndices;
}

// IndexedVectorRef //

template <typename T>
inline IndexedVectorRef<T>::IndexedVectorRef(
    const IndexedVector<T> &iv, size_t idx)
    : m_iv(&iv), m_idx(idx)
{}

template <typename T>
inline size_t IndexedVectorRef<T>::index() const
{
  return m_idx;
}

template <typename T>
inline const IndexedVector<T> &IndexedVectorRef<T>::storage() const
{
  return *m_iv;
}

template <typename T>
inline T IndexedVectorRef<T>::value_or(T alt) const
{
  return valid() ? *data() : alt;
}

template <typename T>
inline T *IndexedVectorRef<T>::data()
{
  return valid() ? &storage()[index()] : nullptr;
}

template <typename T>
inline const T *IndexedVectorRef<T>::data() const
{
  return valid() ? &storage()[index()] : nullptr;
}

template <typename T>
inline T &IndexedVectorRef<T>::operator*()
{
  return *data();
}

template <typename T>
inline const T &IndexedVectorRef<T>::operator*() const
{
  return *data();
}

template <typename T>
inline T *IndexedVectorRef<T>::operator->()
{
  return &storage()[index()];
}

template <typename T>
inline const T *IndexedVectorRef<T>::operator->() const
{
  return &storage()[index()];
}

template <typename T>
inline bool IndexedVectorRef<T>::valid() const
{
  return m_idx != INVALID_INDEX && m_iv;
}

template <typename T>
inline IndexedVectorRef<T>::operator bool() const
{
  return valid();
}

template <typename T>
inline bool operator==(
    const IndexedVectorRef<T> &a, const IndexedVectorRef<T> &b)
{
  return a.m_iv == b.m_iv && a.m_idx == b.m_idx;
}

template <typename T>
inline bool operator!=(
    const IndexedVectorRef<T> &a, const IndexedVectorRef<T> &b)
{
  return !(a == b);
}

} // namespace tsd

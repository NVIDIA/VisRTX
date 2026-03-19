// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace tsd::core {

constexpr size_t INVALID_INDEX = ~size_t(0);
#define TSD_INVALID_INDEX tsd::core::INVALID_INDEX

template <typename T>
struct ObjectPoolRef;

using ObjectPoolRemapping = std::pair<size_t, size_t>; // old index -> new index

/*
 * Sparse object store with O(1) insert and erase; slots are tracked by a
 * bool marker array and recycled via a free-index stack.
 *
 * Example:
 *   ObjectPool<MyObj> pool;
 *   auto ref = pool.emplace(args...);
 *   ref->doSomething();
 *   pool.erase(ref.index());
 */
template <typename T>
struct ObjectPool
{
  using element_t = T;
  using storage_t = std::vector<element_t>;
  using marker_t = std::vector<bool>;
  using index_pool_t = std::vector<size_t>;

  ObjectPool() = default;
  ObjectPool(size_t reserveSize);
  ~ObjectPool() = default;

  T &operator[](size_t i) const; // raw access
  ObjectPoolRef<T> at(size_t i) const; // safe access

  size_t size() const;
  size_t capacity() const;
  bool empty() const;

  bool slot_empty(size_t i) const;
  float density() const;
  bool isDense() const;

  ObjectPoolRef<T> insert(T &&v);
  template <typename... Args>
  ObjectPoolRef<T> emplace(Args &&...args);
  bool erase(size_t i);

  void clear();
  void reserve(size_t size);

  std::vector<ObjectPoolRemapping> defragment();

  template <typename U>
  void sync_slots(const ObjectPool<U> &o);

 private:
  template <typename U>
  friend struct ObjectPool;

  mutable storage_t m_values;
  marker_t m_slots;
  index_pool_t m_freeIndices;
};

/*
 * Nullable reference into an ObjectPool that validates slot occupancy on
 * every dereference; acts as a safe handle to a pool-managed object.
 *
 * Example:
 *   ObjectPoolRef<MyObj> ref = pool.at(idx);
 *   if (ref) ref->update();
 *   size_t i = ref.index();
 */
template <typename T>
struct ObjectPoolRef
{
  ObjectPoolRef() = default;
  ObjectPoolRef(const ObjectPool<T> &iv, size_t idx);

  size_t index() const;

  const ObjectPool<T> &storage() const;

  T value_or(T alt) const;

  T *data();
  const T *data() const;

  T &operator*();
  const T &operator*() const;
  T *operator->();
  const T *operator->() const;

  bool valid() const;
  operator bool() const;

  ObjectPoolRef(const ObjectPoolRef &) = default;
  ObjectPoolRef &operator=(const ObjectPoolRef &) = default;
  ObjectPoolRef(ObjectPoolRef &&) = default;
  ObjectPoolRef &operator=(ObjectPoolRef &&) = default;

 private:
  template <typename U>
  friend bool operator==(const ObjectPoolRef<U> &a, const ObjectPoolRef<U> &b);

  size_t m_idx{INVALID_INDEX};
  const ObjectPool<T> *m_iv{nullptr};
};

template <typename T>
bool operator==(const ObjectPoolRef<T> &a, const ObjectPoolRef<T> &b);
template <typename T>
bool operator!=(const ObjectPoolRef<T> &a, const ObjectPoolRef<T> &b);

template <typename T, typename FCN_T>
inline void foreach_item_ref(ObjectPool<T> &iv, FCN_T &&fcn)
{
  for (size_t i = 0; i < iv.capacity(); i++)
    fcn(iv.at(i));
}

template <typename T, typename FCN_T>
inline void foreach_item(ObjectPool<T> &iv, FCN_T &&fcn)
{
  for (size_t i = 0; i < iv.capacity(); i++)
    fcn(iv.slot_empty(i) ? nullptr : &iv[i]);
}

template <typename T, typename FCN_T>
inline void foreach_item_const(const ObjectPool<T> &iv, FCN_T &&fcn)
{
  for (size_t i = 0; i < iv.capacity(); i++)
    fcn(iv.slot_empty(i) ? nullptr : &iv[i]);
}

template <typename T, typename FCN_T>
inline ObjectPoolRef<T> find_item_if(const ObjectPool<T> &iv, FCN_T &&fcn)
{
  for (size_t i = 0; i < iv.capacity(); i++) {
    if (fcn(iv.slot_empty(i) ? nullptr : &iv[i]))
      return iv.at(i);
  }
  return {};
}

// Inlined definitions ////////////////////////////////////////////////////////

// ObjectPool //

template <typename T>
inline ObjectPool<T>::ObjectPool(size_t reserveSize)
{
  reserve(reserveSize);
}

template <typename T>
inline T &ObjectPool<T>::operator[](size_t i) const
{
  return m_values[i];
}

template <typename T>
inline ObjectPoolRef<T> ObjectPool<T>::at(size_t i) const
{
  return i >= capacity() || slot_empty(i) ? ObjectPoolRef<T>{}
                                          : ObjectPoolRef<T>(*this, i);
}

template <typename T>
inline size_t ObjectPool<T>::size() const
{
  return m_values.size() - m_freeIndices.size();
}

template <typename T>
inline size_t ObjectPool<T>::capacity() const
{
  return m_values.size();
}

template <typename T>
inline bool ObjectPool<T>::empty() const
{
  return size() == 0;
}

template <typename T>
inline bool ObjectPool<T>::slot_empty(size_t i) const
{
  return !m_slots[i];
}

template <typename T>
inline float ObjectPool<T>::density() const
{
  return capacity() == 0 ? 1.f : 1.f - float(m_freeIndices.size()) / capacity();
}

template <typename T>
inline bool ObjectPool<T>::isDense() const
{
  return density() == 1.f;
}

template <typename T>
inline ObjectPoolRef<T> ObjectPool<T>::insert(T &&v)
{
  if (m_freeIndices.empty()) {
    m_values.emplace_back(std::move(v));
    m_slots.push_back(true);
    return at(m_values.size() - 1);
  } else {
    size_t i = m_freeIndices.back();
    m_freeIndices.pop_back();
    m_values[i] = std::move(v);
    m_slots[i] = true;
    return at(i);
  }
}

template <typename T>
template <typename... Args>
ObjectPoolRef<T> ObjectPool<T>::emplace(Args &&...args)
{
  return insert(std::move(T(std::forward<Args>(args)...)));
}

template <typename T>
inline bool ObjectPool<T>::erase(size_t i)
{
  if (slot_empty(i))
    return false;

  m_values[i] = {};
  m_slots[i] = false;
  m_freeIndices.push_back(i);

  return true;
}

template <typename T>
inline void ObjectPool<T>::clear()
{
  m_values.clear();
  m_slots.clear();
  m_freeIndices = {};
}

template <typename T>
inline void ObjectPool<T>::reserve(size_t size)
{
  m_values.reserve(size);
  m_slots.reserve(size);
}

template <typename T>
inline std::vector<ObjectPoolRemapping> ObjectPool<T>::defragment()
{
  if (isDense())
    return {};

  std::vector<ObjectPoolRemapping> retval;
  retval.reserve(m_freeIndices.size());

  const size_t finalSize = size();
  if (finalSize == 0) {
    clear();
    return {};
  }

  auto findLastOccupiedIdx = [&](size_t current) -> size_t {
    if (!slot_empty(current))
      return current;
    while (current > 0 && slot_empty(current - 1))
      current--;
    return current > 0 ? current - 1 : INVALID_INDEX;
  };

  std::sort(m_freeIndices.begin(), m_freeIndices.end(), std::greater<size_t>());

  size_t lastOccupied = capacity() - 1;
  while (!m_freeIndices.empty()) {
    size_t freeIdx = m_freeIndices.back();
    lastOccupied = findLastOccupiedIdx(lastOccupied);
    if (lastOccupied == INVALID_INDEX)
      break;
    m_values[freeIdx] = std::move(m_values[lastOccupied]);
    m_slots[lastOccupied] = false;
    retval.push_back({lastOccupied, freeIdx});
    m_freeIndices.pop_back();
  }

  m_freeIndices = {};
  m_values.resize(finalSize);
  m_slots.resize(finalSize);
  std::fill(m_slots.begin(), m_slots.end(), true);

  return retval;
}

template <typename T>
template <typename U>
inline void ObjectPool<T>::sync_slots(const ObjectPool<U> &o)
{
  m_slots = o.m_slots;
  m_freeIndices = o.m_freeIndices;
}

// ObjectPoolRef //

template <typename T>
inline ObjectPoolRef<T>::ObjectPoolRef(const ObjectPool<T> &iv, size_t idx)
    : m_iv(&iv), m_idx(idx)
{}

template <typename T>
inline size_t ObjectPoolRef<T>::index() const
{
  return m_idx;
}

template <typename T>
inline const ObjectPool<T> &ObjectPoolRef<T>::storage() const
{
  return *m_iv;
}

template <typename T>
inline T ObjectPoolRef<T>::value_or(T alt) const
{
  return valid() ? *data() : alt;
}

template <typename T>
inline T *ObjectPoolRef<T>::data()
{
  return valid() ? &storage()[index()] : nullptr;
}

template <typename T>
inline const T *ObjectPoolRef<T>::data() const
{
  return valid() ? &storage()[index()] : nullptr;
}

template <typename T>
inline T &ObjectPoolRef<T>::operator*()
{
  return *data();
}

template <typename T>
inline const T &ObjectPoolRef<T>::operator*() const
{
  return *data();
}

template <typename T>
inline T *ObjectPoolRef<T>::operator->()
{
  return data();
}

template <typename T>
inline const T *ObjectPoolRef<T>::operator->() const
{
  return data();
}

template <typename T>
inline bool ObjectPoolRef<T>::valid() const
{
  return m_idx != INVALID_INDEX && m_iv;
}

template <typename T>
inline ObjectPoolRef<T>::operator bool() const
{
  return valid();
}

template <typename T>
inline bool operator==(const ObjectPoolRef<T> &a, const ObjectPoolRef<T> &b)
{
  return a.m_iv == b.m_iv && a.m_idx == b.m_idx;
}

template <typename T>
inline bool operator!=(const ObjectPoolRef<T> &a, const ObjectPoolRef<T> &b)
{
  return !(a == b);
}

} // namespace tsd::core

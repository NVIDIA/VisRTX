// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tsd::core {

template <typename KEY, typename VALUE>
struct FlatMap
{
  using item_t = std::pair<KEY, VALUE>;
  using storage_t = std::vector<item_t>;
  using iterator_t = decltype(std::declval<storage_t>().begin());
  using citerator_t = decltype(std::declval<storage_t>().cbegin());

  FlatMap() = default;
  ~FlatMap() = default;

  FlatMap(const FlatMap &) = default;
  FlatMap &operator=(const FlatMap &) = default;

  FlatMap(FlatMap &&) = default;
  FlatMap &operator=(FlatMap &&) = default;

  // Key-based lookups //

  VALUE *at(const KEY &key);
  const VALUE *at(const KEY &key) const;

  VALUE &operator[](const KEY &key);
  const VALUE &operator[](const KEY &key) const;

  void set(const KEY &key, const VALUE &v);

  // Index-based lookups //

  item_t &at_index(size_t index);
  const item_t &at_index(size_t index) const;

  // Property queries //

  size_t size() const;
  bool empty() const;

  bool contains(const KEY &key) const;

  std::optional<size_t> indexOfKey(const KEY &key) const;
  std::optional<size_t> indexOfFirstValue(const VALUE &value) const;
  std::optional<KEY> keyOfFirstValue(const VALUE &value) const;

  // Storage mutation //

  void erase(const KEY &key);
  void erase(size_t index);

  void clear();
  void reserve(size_t size);
  void shrink(size_t newSize);

  // Iterators //

  iterator_t begin();
  citerator_t begin() const;
  citerator_t cbegin() const;

  iterator_t end();
  citerator_t end() const;
  citerator_t cend() const;

 private:
  // Helpers //

  iterator_t lookup(const KEY &key);
  citerator_t lookup(const KEY &key) const;

  // Data //

  storage_t values;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename KEY, typename VALUE>
inline VALUE *FlatMap<KEY, VALUE>::at(const KEY &key)
{
  auto itr = lookup(key);
  return itr == values.end() ? nullptr : &itr->second;
}

template <typename KEY, typename VALUE>
inline const VALUE *FlatMap<KEY, VALUE>::at(const KEY &key) const
{
  auto itr = lookup(key);
  return itr == values.end() ? nullptr : &itr->second;
}

template <typename KEY, typename VALUE>
inline VALUE &FlatMap<KEY, VALUE>::operator[](const KEY &key)
{
  auto itr = lookup(key);
  if (itr == values.end()) {
    values.push_back(std::make_pair(key, VALUE()));
    return values.back().second;
  } else {
    return itr->second;
  }
}

template <typename KEY, typename VALUE>
inline const VALUE &FlatMap<KEY, VALUE>::operator[](const KEY &key) const
{
  auto itr = lookup(key);
  if (itr == values.end()) {
    values.push_back(std::make_pair(key, VALUE()));
    return values.back().second;
  } else {
    return itr->second;
  }
}

template <typename KEY, typename VALUE>
inline void FlatMap<KEY, VALUE>::set(const KEY &key, const VALUE &value)
{
  if (auto *v = this->at(key); v != nullptr)
    *v = value;
  else
    values.push_back(std::make_pair(key, value));
}

template <typename KEY, typename VALUE>
inline typename FlatMap<KEY, VALUE>::item_t &FlatMap<KEY, VALUE>::at_index(
    size_t index)
{
  return values.at(index);
}

template <typename KEY, typename VALUE>
inline const typename FlatMap<KEY, VALUE>::item_t &
FlatMap<KEY, VALUE>::at_index(size_t index) const
{
  return values.at(index);
}

template <typename KEY, typename VALUE>
inline size_t FlatMap<KEY, VALUE>::size() const
{
  return values.size();
}

template <typename KEY, typename VALUE>
inline bool FlatMap<KEY, VALUE>::empty() const
{
  return values.empty();
}

template <typename KEY, typename VALUE>
inline bool FlatMap<KEY, VALUE>::contains(const KEY &key) const
{
  return lookup(key) != values.cend();
}

template <typename KEY, typename VALUE>
inline std::optional<size_t> FlatMap<KEY, VALUE>::indexOfKey(
    const KEY &key) const
{
  std::optional<size_t> idx;
  for (size_t i = 0; !idx && i < values.size(); i++) {
    if (values[i].first == key)
      idx = i;
  }
  return idx;
}

template <typename KEY, typename VALUE>
inline std::optional<size_t> FlatMap<KEY, VALUE>::indexOfFirstValue(
    const VALUE &value) const
{
  std::optional<size_t> idx;
  for (size_t i = 0; !idx && i < values.size(); i++) {
    if (values[i].second == value)
      idx = i;
  }
  return idx;
}

template <typename KEY, typename VALUE>
inline std::optional<KEY> FlatMap<KEY, VALUE>::keyOfFirstValue(
    const VALUE &value) const
{
  std::optional<KEY> retval;
  if (auto idx = indexOfFirstValue(value); idx)
    retval = at_index(*idx).first;
  return retval;
}

template <typename KEY, typename VALUE>
inline void FlatMap<KEY, VALUE>::erase(const KEY &key)
{
  auto itr = std::stable_partition(values.begin(),
      values.end(),
      [&](const item_t &i) { return i.first != key; });

  values.resize(std::distance(values.begin(), itr));
}

template <typename KEY, typename VALUE>
inline void FlatMap<KEY, VALUE>::erase(size_t index)
{
  if (index >= values.size())
    return;
  values.erase(values.begin() + index);
}

template <typename KEY, typename VALUE>
inline void FlatMap<KEY, VALUE>::clear()
{
  values.clear();
}

template <typename KEY, typename VALUE>
inline void FlatMap<KEY, VALUE>::reserve(size_t size)
{
  return values.reserve(size);
}

template <typename KEY, typename VALUE>
inline void FlatMap<KEY, VALUE>::shrink(size_t newSize)
{
  if (newSize >= values.size())
    return;
  values.resize(newSize);
}

// Iterators //

template <typename KEY, typename VALUE>
inline typename FlatMap<KEY, VALUE>::iterator_t FlatMap<KEY, VALUE>::begin()
{
  return values.begin();
}

template <typename KEY, typename VALUE>
inline typename FlatMap<KEY, VALUE>::citerator_t FlatMap<KEY, VALUE>::begin()
    const
{
  return cbegin();
}

template <typename KEY, typename VALUE>
inline typename FlatMap<KEY, VALUE>::citerator_t FlatMap<KEY, VALUE>::cbegin()
    const
{
  return values.cbegin();
}

template <typename KEY, typename VALUE>
inline typename FlatMap<KEY, VALUE>::iterator_t FlatMap<KEY, VALUE>::end()
{
  return values.end();
}

template <typename KEY, typename VALUE>
inline typename FlatMap<KEY, VALUE>::citerator_t FlatMap<KEY, VALUE>::end()
    const
{
  return cend();
}

template <typename KEY, typename VALUE>
inline typename FlatMap<KEY, VALUE>::citerator_t FlatMap<KEY, VALUE>::cend()
    const
{
  return values.cend();
}

// Helper functions //

template <typename KEY, typename VALUE>
inline typename FlatMap<KEY, VALUE>::iterator_t FlatMap<KEY, VALUE>::lookup(
    const KEY &key)
{
  return std::find_if(values.begin(), values.end(), [&](item_t &item) {
    return item.first == key;
  });
}

template <typename KEY, typename VALUE>
inline typename FlatMap<KEY, VALUE>::citerator_t FlatMap<KEY, VALUE>::lookup(
    const KEY &key) const
{
  return std::find_if(values.cbegin(), values.cend(), [&](const item_t &item) {
    return item.first == key;
  });
}

} // namespace tsd::core

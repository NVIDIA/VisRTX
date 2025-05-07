// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/containers/Forest.hpp"
#include "tsd/core/Any.hpp"
// std
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace tsd::serialization {

using Any = utility::Any;
struct DataTree;

struct DataNode
{
  using Ptr = std::unique_ptr<DataNode>;

  DataNode() = default;
  ~DataNode();
  DataNode(const DataNode &) = default;
  DataNode(DataNode &&) = default;
  DataNode &operator=(const DataNode &) = default;
  DataNode &operator=(DataNode &&v) = default;

  const std::string &name() const;

  // Setting values //

  template <typename T>
  DataNode &operator=(const T &v);
  template <typename T>
  DataNode &operator=(T &&v);

  template <typename T>
  void setValue(const T &v);

  template <typename T>
  void setValueAsArray(const std::vector<T> &v);
  template <typename T>
  void setValueAsArray(const T *v, size_t numElements);
  void setValueAsArray(anari::DataType type, const void *v, size_t numElements);

  void clearValue();

  // Getting values //

  template <typename T>
  void valueAsArray(T **ptr, size_t *size) const;

  template <typename T>
  T valueAs() const;
  const Any &value() const;

  bool holdsArray() const;
  anari::DataType arrayType() const;
  bool empty() const;

  // Access children //

  size_t numChildren() const;

  DataNode *child(const std::string &childName);
  const DataNode *child(const std::string &childName) const;
  DataNode *child(size_t childIdx);
  const DataNode *child(size_t childIdx) const;
  DataNode &operator[](const std::string &childName);

  DataNode &append(const std::string &newChildName = "");
  void remove(const std::string &name);
  void remove(DataNode &childNode);

#ifndef TSD_DATA_TREE_TEST_MODE // allow access to self() in unit tests only (!)
 private:
#endif
  utility::ForestNodeRef<DataNode::Ptr> self() const;

 private:
  friend struct DataTree;

  DataNode(const std::string &name); // only Data[Node|Tree] can construct nodes

  // Data members //

  struct NodeData
  {
    utility::ForestNodeRef<DataNode::Ptr> self;
    std::string name;
    Any value;
    std::vector<uint8_t> arrayBytes;
    anari::DataType arrayType{ANARI_UNKNOWN};
  } m_data;
};

struct DataTree
{
  DataTree();
  ~DataTree();

  // Root node access //

  DataNode &root();

  // File I/O //

  // TODO

  // Not movable or copyable //

  DataTree(const DataTree &) = delete;
  DataTree &operator=(const DataTree &) = delete;
  DataTree(DataTree &&) = delete;
  DataTree &operator=(DataTree &&) = delete;

 private:
  utility::Forest<std::unique_ptr<DataNode>> m_tree;
};

// Inlined definitions ////////////////////////////////////////////////////////

namespace detail {

template <typename T>
inline constexpr void validateStorageType()
{
  constexpr anari::DataType type = anari::ANARITypeFor<T>::value;
  static_assert(!anari::isObject(type), "cannot store objects in DataNode");
  static_assert(
      type != ANARI_UNKNOWN, "unknown ANARI data type set on a DataNode");
}

inline constexpr bool validateStorageType(anari::DataType type)
{
  return !anari::isObject(type) && type != ANARI_UNKNOWN;
}

} // namespace detail

// DataNode //

inline DataNode::~DataNode() = default;

inline const std::string &DataNode::name() const
{
  return m_data.name;
}

template <typename T>
inline DataNode &DataNode::operator=(const T &v)
{
  setValue(v);
  return *this;
}

template <typename T>
inline DataNode &DataNode::operator=(T &&v)
{
  setValue(v);
  return *this;
}

template <typename T>
inline void DataNode::setValue(const T &v)
{
  detail::validateStorageType<T>();
  clearValue();
  m_data.value = v;
}

template <typename T>
inline void DataNode::setValueAsArray(const std::vector<T> &v)
{
  setValueAsArray(v.data(), v.size());
}

template <typename T>
inline void DataNode::setValueAsArray(const T *v, size_t numElements)
{
  setValueAsArray(anari::ANARITypeFor<T>::value, v, numElements);
}

inline void DataNode::setValueAsArray(
    anari::DataType type, const void *v, size_t numElements)
{
  detail::validateStorageType(type);
  clearValue();
  m_data.arrayType = type;
  m_data.arrayBytes.resize(numElements * anari::sizeOf(type));
  std::memcpy(m_data.arrayBytes.data(), v, m_data.arrayBytes.size());
}

inline void DataNode::clearValue()
{
  m_data.value.reset();
  m_data.arrayBytes.clear();
  m_data.arrayType = ANARI_UNKNOWN;
}

template <typename T>
inline void DataNode::valueAsArray(T **ptr, size_t *size) const
{
  detail::validateStorageType<T>();
  if (!holdsArray() || m_data.arrayType != anari::ANARITypeFor<T>::value) {
    *ptr = nullptr;
    *size = 0;
  } else {
    *ptr = (T *)m_data.arrayBytes.data();
    *size = m_data.arrayBytes.size() / sizeof(T);
  }
}

template <typename T>
inline T DataNode::valueAs() const
{
  return value().getAs<T>();
}

inline const Any &DataNode::value() const
{
  return m_data.value;
}

inline bool DataNode::holdsArray() const
{
  return arrayType() != ANARI_UNKNOWN;
}

inline anari::DataType DataNode::arrayType() const
{
  return m_data.arrayType;
}

inline bool DataNode::empty() const
{
  return !m_data.value && !holdsArray();
}

inline size_t DataNode::numChildren() const
{
  size_t num = 0;
  utility::foreach_child(self(), [&](auto &n) { num++; });
  return num;
}

inline DataNode *DataNode::child(const std::string &childName)
{
  auto n = utility::find_first_child(
      self(), [&](DataNode::Ptr &cn) { return cn->name() == childName; });
  return n ? (**n).get() : nullptr;
}

inline const DataNode *DataNode::child(const std::string &childName) const
{
  auto n = utility::find_first_child(
      self(), [&](DataNode::Ptr &cn) { return cn->name() == childName; });
  return n ? (**n).get() : nullptr;
}

inline DataNode &DataNode::operator[](const std::string &childName)
{
  auto *dn = child(childName);
  return dn ? *dn : append(childName);
}

inline DataNode *DataNode::child(size_t childIdx)
{
  size_t i = 0;
  auto n = utility::find_first_child(
      self(), [&](DataNode::Ptr &cn) { return i++ == childIdx; });
  return n ? (**n).get() : nullptr;
}

inline const DataNode *DataNode::child(size_t childIdx) const
{
  size_t i = 0;
  auto n = utility::find_first_child(
      self(), [&](DataNode::Ptr &cn) { return i++ == childIdx; });
  return n ? (**n).get() : nullptr;
}

inline DataNode &DataNode::append(const std::string &newChildName)
{
  if (newChildName.empty()) {
    std::string name = '<' + std::to_string(numChildren() + 1) + '>';
    auto ref = self()->insert_last_child(DataNode::Ptr{new DataNode(name)});
    ref->value()->m_data.self = ref;
    return ***ref;
  } else {
    if (auto *c = child(newChildName); c != nullptr)
      return *c;
    else {
      auto ref =
          self()->insert_last_child(DataNode::Ptr{new DataNode(newChildName)});
      ref->value()->m_data.self = ref;
      return ***ref;
    }
  }
}

inline void DataNode::remove(const std::string &name)
{
  if (auto *c = child(name); c != nullptr)
    self()->container()->erase(c->self());
}

inline void DataNode::remove(DataNode &childNode)
{
  self()->container()->erase(childNode.self());
}

inline utility::ForestNodeRef<DataNode::Ptr> DataNode::self() const
{
  return m_data.self;
}

inline DataNode::DataNode(const std::string &name)
{
  m_data.name = name;
}

// DataTree //

inline DataTree::DataTree() : m_tree(DataNode::Ptr{new DataNode("root")})
{
  root().m_data.self = m_tree.root();
}

inline DataTree::~DataTree() = default;

inline DataNode &DataTree::root()
{
  return ***m_tree.root();
}

} // namespace tsd::serialization

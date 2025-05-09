// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/containers/Forest.hpp"
#include "tsd/core/Any.hpp"
// std
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace tsd::serialization {

using Any = utility::Any;
struct DataTree;

struct DataNode
{
  using Ptr = std::unique_ptr<DataNode>;
  using Ref = utility::ForestNodeRef<DataNode::Ptr>;

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
  void setValue(anari::DataType type, const void *v);

  template <typename T>
  void setValueAsArray(const std::vector<T> &v);
  template <typename T>
  void setValueAsArray(const T *v, size_t numElements);
  void setValueAsArray(anari::DataType type, const void *v, size_t numElements);

  void *setValueAsArray(anari::DataType type, size_t numElements);

  void clearValue();

  // Getting values //

  template <typename T>
  void getValueAsArray(const T **ptr, size_t *size) const;
  template <typename T>
  void getValueAsArray(T **ptr, size_t *size);
  void getValueAsArray(
      anari::DataType *type, const void **ptr, size_t *size) const;
  void getValueAsArray(anari::DataType *type, void **ptr, size_t *size);

  template <typename T>
  T getValueAs() const;
  const Any &getValue() const;

  bool holdsArray() const;
  anari::DataType arrayType() const;
  bool empty() const;
  bool isLeaf() const;

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
  Ref self() const;

 private:
  friend struct DataTree;

  DataNode(const std::string &name); // only Data[Node|Tree] can construct nodes

  // Data members //

  struct NodeData
  {
    Ref self;
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

  void save(const char *filename);
  void load(const char *filename);

  // Not movable or copyable //

  DataTree(const DataTree &) = delete;
  DataTree &operator=(const DataTree &) = delete;
  DataTree(DataTree &&) = delete;
  DataTree &operator=(DataTree &&) = delete;

 private:
  void writeDataNode(
      std::FILE *fp, const DataNode &node, const std::string &path) const;

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
  self()->erase_subtree();
}

inline void DataNode::setValue(anari::DataType type, const void *v)
{
  detail::validateStorageType(type);
  clearValue();
  m_data.value = Any(type, v);
  self()->erase_subtree();
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
  self()->erase_subtree();
}

inline void *DataNode::setValueAsArray(anari::DataType type, size_t numElements)
{
  detail::validateStorageType(type);
  clearValue();
  m_data.arrayType = type;
  m_data.arrayBytes.resize(numElements * anari::sizeOf(type));
  self()->erase_subtree();
  return m_data.arrayBytes.data();
}

inline void DataNode::clearValue()
{
  m_data.value.reset();
  m_data.arrayBytes.clear();
  m_data.arrayType = ANARI_UNKNOWN;
}

template <typename T>
inline void DataNode::getValueAsArray(T **ptr, size_t *size)
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
inline void DataNode::getValueAsArray(const T **ptr, size_t *size) const
{
  detail::validateStorageType<T>();
  if (!holdsArray() || m_data.arrayType != anari::ANARITypeFor<T>::value) {
    *ptr = nullptr;
    *size = 0;
  } else {
    *ptr = (const T *)m_data.arrayBytes.data();
    *size = m_data.arrayBytes.size() / sizeof(T);
  }
}

inline void DataNode::getValueAsArray(
    anari::DataType *type, void **ptr, size_t *size)
{
  if (!holdsArray()) {
    *type = ANARI_UNKNOWN;
    *ptr = nullptr;
    *size = 0;
  } else {
    *type = m_data.arrayType;
    *ptr = m_data.arrayBytes.data();
    *size = m_data.arrayBytes.size() / anari::sizeOf(m_data.arrayType);
  }
}

inline void DataNode::getValueAsArray(
    anari::DataType *type, const void **ptr, size_t *size) const
{
  if (!holdsArray()) {
    *type = ANARI_UNKNOWN;
    *ptr = nullptr;
    *size = 0;
  } else {
    *type = m_data.arrayType;
    *ptr = m_data.arrayBytes.data();
    *size = m_data.arrayBytes.size() / anari::sizeOf(m_data.arrayType);
  }
}

template <typename T>
inline T DataNode::getValueAs() const
{
  return getValue().getAs<T>();
}

template <>
inline std::string DataNode::getValueAs() const
{
  return getValue().getString();
}

inline const Any &DataNode::getValue() const
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

inline bool DataNode::isLeaf() const
{
  return self()->isLeaf();
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
  clearValue();

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

inline DataNode::Ref DataNode::self() const
{
  return m_data.self;
}

inline DataNode::DataNode(const std::string &name)
{
  m_data.name = name;
}

// DataTree //

inline DataTree::DataTree() : m_tree(DataNode::Ptr{new DataNode("<root>")})
{
  root().m_data.self = m_tree.root();
}

inline DataTree::~DataTree() = default;

inline DataNode &DataTree::root()
{
  return ***m_tree.root();
}

inline void DataTree::save(const char *filename)
{
  std::FILE *fp = std::fopen(filename, "wb");
  if (!fp)
    return;

  // Count + write number of leaf nodes //

  size_t numLeafNodes = 0;
  m_tree.traverse(m_tree.root(), [&](auto &nodeRef, int level) {
    if (level == 0)
      return true;
    else if (auto &node = **nodeRef; nodeRef.isLeaf())
      numLeafNodes++;
    return true;
  });

  std::fwrite(&numLeafNodes, sizeof(size_t), 1, fp);

  // Travese tree and write nodes //

  std::string path;
  path.reserve(256);
  m_tree.traverse(
      m_tree.root(),
      [&](auto &nodeRef, int level) {
        if (level == 0)
          return true;

        if (auto &node = **nodeRef; nodeRef.isLeaf()) {
          writeDataNode(fp, node, path);
          numLeafNodes++;
        } else {
          const auto &name = node.name();
          if (!path.empty())
            path.push_back('\0');
          std::copy(name.begin(), name.end(), std::back_inserter(path));
        }

        return true;
      },
      [&](auto &nodeRef, int level) {
        if (level == 0)
          return;
        else if (level == 1) {
          path.clear();
          return;
        }

        auto &node = **nodeRef;
        auto &parent = ***node.self()->parent();
        if (!nodeRef.isLeaf())
          path.resize(path.size() - (parent.name().size() + 2));
      });

  std::fclose(fp);
}

inline void DataTree::load(const char *filename)
{
  auto splitNullSeparatedStrings =
      [](const char *buffer, size_t bufferSize) -> std::vector<std::string> {
    std::vector<std::string> result;
    for (size_t start = 0; start < bufferSize;) {
      size_t end = start;
      while (end < bufferSize && buffer[end] != '\0')
        ++end;
      if (end > start)
        result.emplace_back(buffer + start, end - start);
      start = end + 1;
    }

    return result;
  };

  /////////////////////////////////////////////////////////////////////////////

  auto *fp = std::fopen(filename, "rb");
  if (!fp)
    return;

  m_tree.root()->erase_subtree();
  auto &rootNode = root();

  size_t numLeafNodes = 0;
  auto r = std::fread(&numLeafNodes, sizeof(size_t), 1, fp);

  for (size_t i = 0; i < numLeafNodes; i++) {
    size_t size = 0;

    // name
    r = std::fread(&size, sizeof(size_t), 1, fp);
    std::string name(size, '\0');
    r = std::fread(name.data(), sizeof(char), size, fp);

    // path
    r = std::fread(&size, sizeof(size_t), 1, fp);
    std::string fullPath(size, '\0');
    r = std::fread(fullPath.data(), sizeof(char), size, fp);

    // Create node //

    auto path =
        splitNullSeparatedStrings(fullPath.c_str(), fullPath.size() + 1);

    DataNode *parentPtr = &rootNode;
    for (auto &loc : path) {
      if (!loc.empty())
        parentPtr = &parentPtr->append(loc);
    }

    auto &node = parentPtr->append(name);

    // Read node value //

    // isArray
    uint8_t isArray = 0;
    r = std::fread(&isArray, sizeof(uint8_t), 1, fp);

    // type
    anari::DataType type = ANARI_UNKNOWN;
    r = std::fread(&type, sizeof(anari::DataType), 1, fp);

    if (isArray) {
      // array size + data
      r = std::fread(&size, sizeof(size_t), 1, fp);
      void *dataPtr = node.setValueAsArray(type, size);
      r = std::fread(dataPtr, anari::sizeOf(type), size, fp);
    } else {
      // value data
      if (type == ANARI_STRING) {
        r = std::fread(&size, sizeof(size_t), 1, fp);
        std::string str(size, '\0');
        r = std::fread(str.data(), sizeof(char), size, fp);
        node = str.c_str();
      } else {
        constexpr int MAX_SIZE = 16 * sizeof(float);
        if (anari::sizeOf(type) <= MAX_SIZE) {
          uint8_t data[MAX_SIZE];
          r = std::fread(data, anari::sizeOf(type), 1, fp);
          node.setValue(type, (void *)data);
        } else {
          printf("ERROR: type %s is too large to read when parsing DataTree\n",
              anari::toString(type));
        }
      }
    }
  }

  std::fclose(fp);
}

inline void DataTree::writeDataNode(
    std::FILE *fp, const DataNode &node, const std::string &path) const
{
  // name
  size_t size = node.name().size();
  std::fwrite(&size, sizeof(size_t), 1, fp);
  std::fwrite(node.name().c_str(), sizeof(char), size, fp);
  // path
  size = path.size();
  std::fwrite(&size, sizeof(size_t), 1, fp);
  std::fwrite(path.c_str(), sizeof(char), size, fp);
  // isArray
  const uint8_t isArray = node.holdsArray();
  std::fwrite(&isArray, sizeof(uint8_t), 1, fp);
  if (isArray) {
    // array info + data
    anari::DataType type = ANARI_UNKNOWN;
    const void *data = nullptr;
    size = 0;
    node.getValueAsArray(&type, &data, &size);
    std::fwrite(&type, sizeof(anari::DataType), 1, fp);
    std::fwrite(&size, sizeof(size_t), 1, fp);
    std::fwrite(data, sizeof(uint8_t), size * anari::sizeOf(type), fp);
  } else {
    // value info + data
    auto &v = node.getValue();
    auto type = v.type();
    std::fwrite(&type, sizeof(anari::DataType), 1, fp);
    if (type == ANARI_STRING) {
      const char *data = v.getCStr();
      size = std::strlen(data);
      std::fwrite(&size, sizeof(size_t), 1, fp);
      std::fwrite(data, sizeof(char), size, fp);
    } else {
      std::fwrite(v.data(), anari::sizeOf(type), 1, fp);
    }
  }
}

} // namespace tsd::serialization

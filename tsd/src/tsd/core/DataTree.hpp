// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Any.hpp"
#include "tsd/core/Forest.hpp"
// std
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace tsd::core {

struct DataTree;

struct DataNode
{
  using Ptr = std::unique_ptr<DataNode>;
  using Ref = ForestNodeRef<DataNode::Ptr>;

  DataNode() = default;
  ~DataNode();
  DataNode(const DataNode &) = default;
  DataNode(DataNode &&) = default;
  DataNode &operator=(const DataNode &) = default;
  DataNode &operator=(DataNode &&v) = default;

  const std::string &name() const;

  void reset(); // clear value and remove children

  // Setting values //

  template <typename T>
  DataNode &operator=(const T &v);

  void setValue(const Any &v);
  template <typename T>
  void setValue(const T &v);
  void setValue(anari::DataType type, const void *v);

  template <typename T>
  void setValueAsArray(const std::vector<T> &v);
  template <typename T>
  void setValueAsArray(const T *v, size_t numElements);
  void setValueAsArray(anari::DataType type, const void *v, size_t numElements);
  void *setValueAsArray(anari::DataType type, size_t numElements);

  void setValueAsExternalArray(
      anari::DataType type, const void *v, size_t numElements);

  void setValueObject(anari::DataType type, size_t idx);

  void clearValue(); // only clear value if present

  // Getting values //

  template <typename T>
  T getValueAs() const;
  template <typename T>
  T getValueOr(const T &alt) const;
  const Any &getValue() const;

  // NOTE: If getting ANARI_STRING, pass ptr to std::string
  bool getValue(anari::DataType type, void *ptr) const;

  template <typename T>
  void getValueAsArray(const T **ptr, size_t *size) const;
  template <typename T>
  void getValueAsArray(T **ptr, size_t *size);
  void getValueAsArray(
      anari::DataType *type, const void **ptr, size_t *size) const;
  void getValueAsArray(anari::DataType *type, void **ptr, size_t *size);

  void getValueAsObjectIdx(anari::DataType *type, size_t *idx) const;

  bool holdsObjectIdx() const;
  bool holdsArray() const;
  bool holdsExternalArray() const; // cannot get mutable pointer to data
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

  // Algorithms //

  void traverse(std::function<bool(DataNode &n, int level)> &&fcn);
  void forall_children(std::function<void(DataNode &)> &&fcn);
  void foreach_child(std::function<void(DataNode &)> &&fcn);

#ifndef TSD_DATA_TREE_TEST_MODE // allow access to self() in unit tests only (!)
 private:
#endif
  Ref self() const;
  Ref parent() const;

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
    const void *externalArray{nullptr};
    size_t externalArraySize{0};
  } m_data;
};

using DataTreeVisitorEntryFunction =
    std::function<bool(DataNode &n, int level)>;
using DataTreeVisitorExitFunction = std::function<void(DataNode &n, int level)>;

struct DataTree
{
  DataTree();
  ~DataTree();

  // Root node access //

  DataNode &root();
  const DataNode &root() const;

  // Traverse nodes //

  void traverse(DataTreeVisitorEntryFunction &&onNodeEntry,
      DataTreeVisitorExitFunction &&onNodeExit = {});
  void traverse(DataNode::Ref start,
      DataTreeVisitorEntryFunction &&onNodeEntry,
      DataTreeVisitorExitFunction &&onNodeExit = {});

  // File I/O //

  void save(const char *filename);
  void load(const char *filename);

  // Visual inspection //

  void print();

  // Not movable or copyable //

  DataTree(const DataTree &) = delete;
  DataTree &operator=(const DataTree &) = delete;
  DataTree(DataTree &&) = delete;
  DataTree &operator=(DataTree &&) = delete;

 private:
  void writeDataNode(
      std::FILE *fp, const DataNode &node, const std::string &path) const;
  std::string printablePath(const std::string &path) const;

  Forest<std::unique_ptr<DataNode>> m_tree;
};

// Inlined definitions ////////////////////////////////////////////////////////

// DataNode //

inline DataNode::~DataNode() = default;

inline const std::string &DataNode::name() const
{
  return m_data.name;
}

inline void DataNode::reset()
{
  clearValue();
  self()->erase_subtree();
}

template <typename T>
inline DataNode &DataNode::operator=(const T &v)
{
  setValue(Any(v));
  return *this;
}

template <>
inline DataNode &DataNode::operator=(const std::string &v)
{
  setValue(Any(v.c_str()));
  return *this;
}

inline void DataNode::setValue(const Any &v)
{
  reset();
  m_data.value = v;
}

template <typename T>
inline void DataNode::setValue(const T &v)
{
  setValue(Any(v));
}

inline void DataNode::setValue(anari::DataType type, const void *v)
{
  setValue(Any(type, v));
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
  auto *ptr = setValueAsArray(type, numElements);
  std::memcpy(ptr, v, m_data.arrayBytes.size());
}

inline void *DataNode::setValueAsArray(anari::DataType type, size_t numElements)
{
  reset();
  m_data.arrayType = type;
  m_data.arrayBytes.resize(numElements * anari::sizeOf(type));
  return m_data.arrayBytes.data();
}

inline void DataNode::setValueAsExternalArray(
    anari::DataType type, const void *v, size_t numElements)
{
  reset();
  m_data.arrayType = type;
  m_data.externalArray = v;
  m_data.externalArraySize = numElements * anari::sizeOf(type);
}

inline void DataNode::setValueObject(anari::DataType type, size_t idx)
{
  setValue(Any(type, idx));
}

inline void DataNode::clearValue()
{
  m_data.value.reset();
  m_data.arrayBytes.clear();
  m_data.arrayType = ANARI_UNKNOWN;
  m_data.externalArray = nullptr;
  m_data.externalArraySize = 0;
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

template <typename T>
inline T DataNode::getValueOr(const T &alt) const
{
  return getValue().is<T>() ? getValueAs<T>() : alt;
}

template <>
inline std::string DataNode::getValueOr(const std::string &alt) const
{
  return getValue().is(ANARI_STRING) ? getValueAs<std::string>() : alt;
}

inline const Any &DataNode::getValue() const
{
  return m_data.value;
}

inline bool DataNode::getValue(anari::DataType type, void *ptr) const
{
  const bool invalidQueryType = type == ANARI_STRING_LIST
      || type == ANARI_DATA_TYPE_LIST || anari::isObject(type);

  if (invalidQueryType || !m_data.value.is(type))
    return false;
  else if (m_data.value.is(ANARI_STRING)) {
    *(std::string *)ptr = getValue().getCStr();
    return true;
  } else {
    std::memcpy(ptr, m_data.value.data(), anari::sizeOf(type));
    return true;
  }
}

template <typename T>
inline void DataNode::getValueAsArray(T **ptr, size_t *size)
{
  static_assert(!anari::isObject(anari::ANARITypeFor<T>::value),
      "getValueAsArray<T> does not work for ANARI object types");

  *ptr = nullptr;
  *size = 0;

  const bool compatible = holdsArray()
      && m_data.arrayType == anari::ANARITypeFor<T>::value
      && !m_data.arrayBytes.empty();

  if (compatible) {
    *ptr = (T *)m_data.arrayBytes.data();
    *size = m_data.arrayBytes.size() / anari::sizeOf(m_data.arrayType);
  }
}

template <typename T>
inline void DataNode::getValueAsArray(const T **ptr, size_t *size) const
{
  static_assert(!anari::isObject(anari::ANARITypeFor<T>::value),
      "getValueAsArray<T> does not work for ANARI object types");
  if (!holdsArray() || m_data.arrayType != anari::ANARITypeFor<T>::value) {
    *ptr = nullptr;
    *size = 0;
  } else {
    if (!m_data.arrayBytes.empty()) {
      *ptr = (const T *)m_data.arrayBytes.data();
      *size = m_data.arrayBytes.size() / anari::sizeOf(m_data.arrayType);
    } else {
      *ptr = (const T *)m_data.externalArray;
      *size = m_data.externalArraySize / anari::sizeOf(m_data.arrayType);
    }
  }
}

inline void DataNode::getValueAsArray(
    anari::DataType *type, void **ptr, size_t *size)
{
  *type = ANARI_UNKNOWN;
  *ptr = nullptr;
  *size = 0;

  if (holdsArray() && !m_data.arrayBytes.empty()) {
    *type = m_data.arrayType;
    *ptr = m_data.arrayBytes.data();
    *size = m_data.arrayBytes.size() / anari::sizeOf(m_data.arrayType);
  }
}

inline void DataNode::getValueAsArray(
    anari::DataType *type, const void **ptr, size_t *size) const
{
  *type = ANARI_UNKNOWN;
  *ptr = nullptr;
  *size = 0;

  if (holdsArray()) {
    *type = m_data.arrayType;
    if (!m_data.arrayBytes.empty()) {
      *ptr = m_data.arrayBytes.data();
      *size = m_data.arrayBytes.size() / anari::sizeOf(m_data.arrayType);
    } else {
      *ptr = m_data.externalArray;
      *size = m_data.externalArraySize / anari::sizeOf(m_data.arrayType);
    }
  }
}

inline void DataNode::getValueAsObjectIdx(
    anari::DataType *type, size_t *idx) const
{
  if (!holdsObjectIdx()) {
    *type = ANARI_UNKNOWN;
    *idx = INVALID_INDEX;
  } else {
    *type = m_data.value.type();
    *idx = m_data.value.getAsObjectIndex();
  }
}

inline bool DataNode::holdsObjectIdx() const
{
  return m_data.value.holdsObject();
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
  ::tsd::core::foreach_child(self(), [&](auto &n) { num++; });
  return num;
}

inline DataNode *DataNode::child(const std::string &childName)
{
  auto n = find_first_child(
      self(), [&](DataNode::Ptr &cn) { return cn->name() == childName; });
  return n ? (**n).get() : nullptr;
}

inline const DataNode *DataNode::child(const std::string &childName) const
{
  auto n = find_first_child(
      self(), [&](DataNode::Ptr &cn) { return cn->name() == childName; });
  return n ? (**n).get() : nullptr;
}

inline DataNode &DataNode::operator[](const std::string &childName)
{
  auto *n = child(childName);
  return n ? *n : append(childName);
}

inline DataNode *DataNode::child(size_t childIdx)
{
  size_t i = 0;
  auto n = find_first_child(
      self(), [&](DataNode::Ptr &cn) { return i++ == childIdx; });
  return n ? (**n).get() : nullptr;
}

inline const DataNode *DataNode::child(size_t childIdx) const
{
  size_t i = 0;
  auto n = find_first_child(
      self(), [&](DataNode::Ptr &cn) { return i++ == childIdx; });
  return n ? (**n).get() : nullptr;
}

inline DataNode &DataNode::append(const std::string &newChildName)
{
  clearValue();

  std::string name = newChildName;
  if (name.empty()) {
#if 1
    static int counter = 0;
    name = '<' + std::to_string(counter++) + '>';
#else // for some reason, this breaks in TSD context export...
    name = '<' + std::to_string(numChildren()) + '>';
#endif
  }

  if (auto *c = child(name); c != nullptr)
    return *c;
  else {
    auto ref = self()->insert_last_child(DataNode::Ptr{new DataNode(name)});
    ref->value()->m_data.self = ref;
    return ***ref;
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

inline void DataNode::traverse(
    std::function<bool(DataNode &n, int level)> &&fcn)
{
  self()->container()->traverse(
      self(), [&](auto &ref, int level) { return fcn(**ref, level); });
}

inline void DataNode::forall_children(std::function<void(DataNode &)> &&fcn)
{
  ::tsd::core::forall_children(self(), [&](auto &ref) { fcn(*ref); });
}

inline void DataNode::foreach_child(std::function<void(DataNode &)> &&fcn)
{
  ::tsd::core::foreach_child(self(), [&](auto &ref) { fcn(*ref); });
}

inline DataNode::Ref DataNode::self() const
{
  return m_data.self;
}

inline DataNode::Ref DataNode::parent() const
{
  return m_data.self->parent();
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

inline const DataNode &DataTree::root() const
{
  return ***m_tree.root();
}

inline void DataTree::traverse(DataTreeVisitorEntryFunction &&onNodeEntry,
    DataTreeVisitorExitFunction &&onNodeExit)
{
  traverse(root().self(), std::move(onNodeEntry), std::move(onNodeExit));
}

inline void DataTree::traverse(DataNode::Ref start,
    DataTreeVisitorEntryFunction &&onNodeEntry,
    DataTreeVisitorExitFunction &&onNodeExit)
{
  // clang-format off
  m_tree.traverse(
      start,
      [&](auto &n, int l) { return onNodeEntry(**n, l); },
      [&](auto &n, int l) { if (onNodeExit) onNodeExit(**n, l); });
  // clang-format on
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
          std::copy(name.begin(), name.end(), std::back_inserter(path));
          path.push_back('\0');
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

        if (!nodeRef.isLeaf())
          path.resize(path.size() - ((*nodeRef)->name().size() + 1));
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

    // isArray
    uint8_t isArray = 0;
    r = std::fread(&isArray, sizeof(uint8_t), 1, fp);

    // type
    anari::DataType type = ANARI_UNKNOWN;
    r = std::fread(&type, sizeof(anari::DataType), 1, fp);

    // Create node //

    auto path = splitNullSeparatedStrings(fullPath.c_str(), fullPath.size());

    DataNode *parentPtr = &rootNode;
    for (auto &loc : path) {
      if (!loc.empty())
        parentPtr = &parentPtr->append(loc);
    }

    auto &node = parentPtr->append(name);

    // Read node value //

    if (isArray) {
      // array size + data
      r = std::fread(&size, sizeof(size_t), 1, fp);
      void *dataPtr = node.setValueAsArray(type, size);
      r = std::fread(dataPtr, anari::sizeOf(type), size, fp);
    } else {
      if (anari::isObject(type)) {
        size_t idx = INVALID_INDEX;
        r = std::fread(&idx, sizeof(size_t), 1, fp);
        node.setValueObject(type, idx);
      } else if (type == ANARI_STRING) {
        r = std::fread(&size, sizeof(size_t), 1, fp);
        std::string str(size, '\0');
        r = std::fread(str.data(), sizeof(char), size, fp);
        node = str.c_str();
      } else if (type != ANARI_UNKNOWN) {
        constexpr int MAX_SIZE = 16 * sizeof(float);
        if (anari::sizeOf(type) <= MAX_SIZE) {
          uint8_t data[MAX_SIZE];
          r = std::fread(data, anari::sizeOf(type), 1, fp);
          node.setValue(type, (void *)data);
        } else {
          printf("ERROR: type %s is too large to read when parsing DataTree\n",
              anari::toString(type));
          fflush(stdout);
          std::fclose(fp);
          abort();
        }
      }
    }
  }

  std::fclose(fp);
}

inline void DataTree::print()
{
  traverse([](tsd::core::DataNode &node, int level) {
    if (level == 0)
      return true;

    for (int i = 1; i < level; i++)
      printf("    ");

    if (!node.isLeaf())
      printf("%s:\n", node.name().c_str());
    else {
      printf("%s: ", node.name().c_str());

      if (node.holdsObjectIdx()) {
        anari::DataType type = ANARI_UNKNOWN;
        size_t index = 0;
        node.getValueAsObjectIdx(&type, &index);
        printf("%s @%zu", anari::toString(type), index);
      } else if (node.holdsArray()) {
        anari::DataType type = ANARI_UNKNOWN;
        const void *data = nullptr;
        size_t size = 0;
        node.getValueAsArray(&type, &data, &size);
        printf("%s[%zu]", anari::toString(type), size);
      } else {
        auto &value = node.getValue();
        printf("%s", anari::toString(value.type()));
        if (value.is(ANARI_STRING))
          printf(" | \"%s\"", value.getCStr());
        else if (value.is<bool>())
          printf(" | %s", value.get<bool>() ? "true" : "false");
        else if (value.is<int>())
          printf(" | %d", value.get<int>());
        else if (value.is<uint32_t>())
          printf(" | %d", value.get<uint32_t>());
        else if (value.is<float>())
          printf(" | %f", value.get<float>());
        else if (value.is<double>())
          printf(" | %f", value.get<double>());
      }

      printf("\n");
    }

    return true;
  });

  printf("\n");
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
    if (anari::isObject(type)) {
      size_t idx = v.getAsObjectIndex();
      std::fwrite(&idx, sizeof(size_t), 1, fp);
    } else if (type == ANARI_STRING) {
      const char *data = v.getCStr();
      size = data ? std::strlen(data) : 0;
      std::fwrite(&size, sizeof(size_t), 1, fp);
      std::fwrite(data, sizeof(char), size, fp);
    } else if (type != ANARI_UNKNOWN) {
      std::fwrite(v.data(), anari::sizeOf(type), 1, fp);
    }
  }
}

inline std::string DataTree::printablePath(const std::string &path) const
{
  std::string printable = path;
  std::replace(printable.begin(), printable.end(), '\0', '/');
  return printable;
}

} // namespace tsd::core

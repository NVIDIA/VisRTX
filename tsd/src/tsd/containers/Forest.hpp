// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "IndexedVector.hpp"
// std
#include <functional>

namespace tsd::utility {

template <typename T>
struct Forest;

template <typename T>
struct ForestNode
{
  using Ref = IndexedVectorRef<ForestNode<T>>;

  ForestNode() = default;
  ForestNode(T initialValue);

  T &value();
  const T &value() const;

  T &operator*();
  const T &operator*() const;
  T *operator->();
  const T *operator->() const;
  operator bool() const;

  bool isRoot() const;
  bool isLeaf() const;

  Ref next() const;
  Ref sibling() const;
  Ref prev() const;

  size_t index() const;

 private:
  friend struct Forest<T>;
  template <typename U>
  friend bool operator==(const ForestNode<U> &a, const ForestNode<U> &b);

  T m_value{};
  Ref m_prev;
  Ref m_children_begin;
  Ref m_children_end;
  Ref m_next;
  Ref m_self;
};

template <typename T>
bool operator==(const ForestNode<T> &a, const ForestNode<T> &b);
template <typename T>
bool operator!=(const ForestNode<T> &a, const ForestNode<T> &b);

// clang-format off
template <typename T>
struct ForestVisitor
{
  virtual ~ForestVisitor() = default;
  virtual bool preChildren(ForestNode<T> &n, int level) { return true; }
  virtual void postChildren(ForestNode<T> &n, int level) {}
};
// clang-format on

template <typename T>
using ForestVisitorEntryFunction =
    std::function<bool(ForestNode<T> &n, int level)>;
template <typename T>
using ForestVisitorExitFunction =
    std::function<void(ForestNode<T> &n, int level)>;

///////////////////////////////////////////////////////////////////////////////
// Forest<> -- a tree-based hierarchy free of cycles
//
//     Data structure based off of the stlab::forest<> described here:
//        https://stlab.cc/2020/12/01/forest-introduction.html
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct Forest
{
  using NodeRef = typename ForestNode<T>::Ref;

  Forest(T &&initialValue);
  ~Forest() = default;

  // ForestNode access //

  size_t size() const;
  bool empty() const;

  NodeRef at(size_t i) const;
  T &operator[](size_t i);
  const T &operator[](size_t i) const;

  NodeRef root() const;

  // Mutation //

  NodeRef insert_first_child(NodeRef n, T &&v);
  NodeRef insert_last_child(NodeRef n, T &&v);

  void erase(NodeRef n);
  void erase_subtree(NodeRef n);

  // Traversal //

  void traverse(NodeRef start, ForestVisitor<T> &visitor);
  void traverse(NodeRef start, ForestVisitorEntryFunction<T> &&f);
  void traverse(NodeRef start,
      ForestVisitorEntryFunction<T> &&onNodeEntry,
      ForestVisitorExitFunction<T> &&onNodeExit);

 private:
  void traverse_impl(NodeRef n, ForestVisitor<T> &visitor, int level);
  NodeRef make_ForestNode(T &&v);

  IndexedVector<ForestNode<T>> m_nodes;
  NodeRef m_root;
};

// Inlined definitions ////////////////////////////////////////////////////////

// ForestNode<> //

template <typename T>
inline ForestNode<T>::ForestNode(T v) : m_value(v)
{}

template <typename T>
inline T &ForestNode<T>::value()
{
  return m_value;
}

template <typename T>
inline const T &ForestNode<T>::value() const
{
  return m_value;
}

template <typename T>
inline T &ForestNode<T>::operator*()
{
  return value();
}

template <typename T>
inline const T &ForestNode<T>::operator*() const
{
  return value();
}

template <typename T>
inline T *ForestNode<T>::operator->()
{
  return &m_value;
}

template <typename T>
inline const T *ForestNode<T>::operator->() const
{
  return &m_value;
}

template <typename T>
inline ForestNode<T>::operator bool() const
{
  return m_self;
}

template <typename T>
inline bool ForestNode<T>::isRoot() const
{
  return *this && !m_prev && !m_next;
}

template <typename T>
inline bool ForestNode<T>::isLeaf() const
{
  return *this && m_children_begin == m_self && m_children_end == m_self;
}

template <typename T>
inline typename ForestNode<T>::Ref ForestNode<T>::next() const
{
  return m_children_begin != m_self ? m_children_begin : m_next;
}

template <typename T>
inline typename ForestNode<T>::Ref ForestNode<T>::sibling() const
{
  return m_next;
}

template <typename T>
inline typename ForestNode<T>::Ref ForestNode<T>::prev() const
{
  return m_prev;
}

template <typename T>
inline size_t ForestNode<T>::index() const
{
  return m_self.index();
}

template <typename T>
inline bool operator==(const ForestNode<T> &a, const ForestNode<T> &b)
{
  return a.m_self == b.m_self;
}

template <typename T>
inline bool operator!=(const ForestNode<T> &a, const ForestNode<T> &b)
{
  return !(a == b);
}

// Forest<> //

template <typename T>
inline Forest<T>::Forest(T &&v)
{
  m_root = make_ForestNode(std::forward<T>(v));
}

template <typename T>
inline size_t Forest<T>::size() const
{
  return m_nodes.size();
}

template <typename T>
inline bool Forest<T>::empty() const
{
  return size() == 1;
}

template <typename T>
inline typename Forest<T>::NodeRef Forest<T>::at(size_t i) const
{
  return m_nodes.at(i);
}

template <typename T>
inline T &Forest<T>::operator[](size_t i)
{
  return *m_nodes[i];
}

template <typename T>
inline const T &Forest<T>::operator[](size_t i) const
{
  return *m_nodes[i];
}

template <typename T>
inline typename Forest<T>::NodeRef Forest<T>::root() const
{
  return m_root;
}

template <typename T>
inline typename Forest<T>::NodeRef Forest<T>::insert_first_child(
    Forest<T>::NodeRef n, T &&v)
{
  auto newForestNode = make_ForestNode(std::forward<T>(v));
  newForestNode->m_prev = n->m_self;
  newForestNode->m_next = n->m_children_begin;
  if (n->isLeaf())
    n->m_children_end = newForestNode;
  else
    n->m_children_begin->m_prev = newForestNode;
  n->m_children_begin = newForestNode;
  return newForestNode;
}

template <typename T>
inline typename Forest<T>::NodeRef Forest<T>::insert_last_child(
    Forest<T>::NodeRef n, T &&v)
{
  auto newForestNode = make_ForestNode(std::forward<T>(v));
  newForestNode->m_prev = n->m_children_end;
  newForestNode->m_next = n->m_self;
  if (n->isLeaf())
    n->m_children_begin = newForestNode;
  else
    n->m_children_end->m_next = newForestNode;
  n->m_children_end = newForestNode;
  return newForestNode;
}

template <typename T>
inline void Forest<T>::erase(Forest<T>::NodeRef n)
{
  erase_subtree(n);

  if (n->isRoot())
    return;

  auto self = n->m_self;
  auto prev = n->prev();
  auto next = n->next();

  const bool prevIsParent = self == prev->m_children_begin;
  if (prevIsParent)
    prev->m_children_begin = next;
  else
    prev->m_next = next;

  const bool nextIsParent = self == next->m_children_end;
  if (nextIsParent)
    next->m_children_end = prev;
  else
    next->m_prev = prev;

  m_nodes.erase(n.index());
}

template <typename T>
inline void Forest<T>::erase_subtree(Forest<T>::NodeRef n)
{
  if (n->isLeaf())
    return;

  struct SubtreeIndices : public ForestVisitor<T>
  {
    bool preChildren(ForestNode<T> &node, int level) override
    {
      if (level != 0)
        indices.push_back(node.m_self.index());
      return true;
    }

    std::vector<size_t> indices;
  };

  SubtreeIndices visitor;
  traverse(n, visitor);

  for (auto &i : visitor.indices)
    m_nodes.erase(i);

  n->m_children_begin = n;
  n->m_children_end = n;
}

template <typename T>
inline void Forest<T>::traverse(
    Forest<T>::NodeRef start, ForestVisitor<T> &visitor)
{
  traverse_impl(start, visitor, 0);
}

template <typename T>
inline void Forest<T>::traverse(
    Forest<T>::NodeRef start, ForestVisitorEntryFunction<T> &&f)
{
  struct FcnVisitor : public ForestVisitor<T>
  {
    FcnVisitor(ForestVisitorEntryFunction<T> &f) : fcn(f) {}
    bool preChildren(tsd::utility::ForestNode<T> &node, int level) override
    {
      return fcn(node, level);
    }
    ForestVisitorEntryFunction<T> &fcn;
  };

  FcnVisitor visitor(f);
  traverse_impl(start, visitor, 0);
}

template <typename T>
inline void Forest<T>::traverse(Forest<T>::NodeRef start,
    ForestVisitorEntryFunction<T> &&onNodeEntry,
    ForestVisitorExitFunction<T> &&onNodeExit)
{
  struct FcnVisitor : public ForestVisitor<T>
  {
    FcnVisitor(
        ForestVisitorEntryFunction<T> &f1, ForestVisitorExitFunction<T> &f2)
        : onEntry(f1), onExit(f2)
    {}
    bool preChildren(tsd::utility::ForestNode<T> &node, int level) override
    {
      return onEntry(node, level);
    }
    void postChildren(tsd::utility::ForestNode<T> &node, int level) override
    {
      onExit(node, level);
    }
    ForestVisitorEntryFunction<T> &onEntry;
    ForestVisitorExitFunction<T> &onExit;
  };

  FcnVisitor visitor(onNodeEntry, onNodeExit);
  traverse_impl(start, visitor, 0);
}

template <typename T>
inline void Forest<T>::traverse_impl(
    NodeRef n, ForestVisitor<T> &visitor, int level)
{
  const bool traverseChildren = visitor.preChildren(*n, level);
  if (traverseChildren && !n->isLeaf()) {
    for (auto s = n->next(); s != n; s = s->sibling())
      traverse_impl(s, visitor, level + 1);
  }
  visitor.postChildren(*n, level);
}

template <typename T>
inline typename Forest<T>::NodeRef Forest<T>::make_ForestNode(T &&v)
{
  auto n = m_nodes.insert(ForestNode<T>(v));
  n->m_self = n;
  n->m_children_end = n;
  n->m_children_begin = n;
  return n;
}

} // namespace tsd::utility
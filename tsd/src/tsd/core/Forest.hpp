// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ObjectPool.hpp"
// std
#include <functional>
#include <utility>

namespace tsd::core {

template <typename T>
struct Forest;

template <typename T>
struct ForestNode
{
  using Ref = ObjectPoolRef<ForestNode<T>>;

  ForestNode() = default;
  ForestNode(T initialValue, Forest<T> *f);

  T &value();
  const T &value() const;

  T &operator*();
  const T &operator*() const;
  T *operator->();
  const T *operator->() const;
  operator bool() const;

  bool isRoot() const;
  bool isLeaf() const;

  Forest<T> *container() const;

  Ref next() const;
  Ref parent() const;
  Ref sibling() const;
  Ref prev() const;

  size_t index() const;

  Ref insert_first_child(T &&v);
  Ref insert_last_child(T &&v);

  void erase_subtree();
  void erase_self(); // remove this node from the forest

 private:
  Ref self() const;
  Forest<T> *forest() const;

  friend struct Forest<T>;
  template <typename U>
  friend bool operator==(const ForestNode<U> &a, const ForestNode<U> &b);

  T m_value{};
  Ref m_prev;
  Ref m_children_begin;
  Ref m_children_end;
  Ref m_next;
  Ref m_parent;
  Ref m_self;
  Forest<T> *m_forest{nullptr};
};

template <typename T>
using ForestNodeRef = ObjectPoolRef<ForestNode<T>>;

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
  using Ref = ObjectPoolRef<Forest<T>>;
  using Node = ForestNode<T>;
  using NodeRef = typename ForestNode<T>::Ref;
  using Visitor = ForestVisitor<T>;

  Forest(T &&initialRootValue);
  Forest() = delete;
  ~Forest() = default;

  Forest(const Forest &) = delete;
  Forest &operator=(const Forest &) = delete;
  Forest(Forest &&) = delete;
  Forest &operator=(Forest &&) = delete;

  void reserve(size_t size);

  // ForestNode access //

  size_t size() const;
  size_t capacity() const; // to get highest possible node index
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
  NodeRef copy_subtree(NodeRef source, NodeRef newParent);
  void move_subtree(NodeRef source, NodeRef newParent);

  void clear();

  // Queries //

  bool isAncestorOf(NodeRef potentialAncestor, NodeRef node) const;

  // Traversal //

  void traverse(NodeRef start, ForestVisitor<T> &visitor);
  void traverse(NodeRef start, ForestVisitorEntryFunction<T> &&f);
  void traverse(NodeRef start,
      ForestVisitorEntryFunction<T> &&onNodeEntry,
      ForestVisitorExitFunction<T> &&onNodeExit);

 private:
  void traverse_impl(NodeRef n, ForestVisitor<T> &visitor, int level);
  NodeRef make_ForestNode(T &&v);

  ObjectPool<ForestNode<T>> m_nodes;
  NodeRef m_root;
};

// Algorithms /////////////////////////////////////////////////////////////////

template <typename T, typename FCN>
void foreach_child(ForestNodeRef<T> node, FCN &&fcn);

template <typename T, typename FCN>
void forall_children(ForestNodeRef<T> node, FCN &&fcn);

template <typename T, typename PREDICATE>
ForestNodeRef<T> find_first_child(ForestNodeRef<T> node, PREDICATE &&pred);

///////////////////////////////////////////////////////////////////////////////
// Inlined definitions ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// ForestNode<> //

template <typename T>
inline ForestNode<T>::ForestNode(T v, Forest<T> *f)
    : m_value(std::move(v)), m_forest(f)
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
  return *this && !m_parent;
}

template <typename T>
inline bool ForestNode<T>::isLeaf() const
{
  return *this && m_children_begin == self() && m_children_end == self();
}

template <typename T>
Forest<T> *ForestNode<T>::container() const
{
  return m_forest;
}

template <typename T>
inline typename ForestNode<T>::Ref ForestNode<T>::next() const
{
  return m_children_begin != self() ? m_children_begin : m_next;
}

template <typename T>
inline typename ForestNode<T>::Ref ForestNode<T>::parent() const
{
  return m_parent;
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
  return self().index();
}

template <typename T>
inline typename ForestNode<T>::Ref ForestNode<T>::insert_first_child(T &&v)
{
  if (auto *f = forest(); f != nullptr)
    return f->insert_first_child(self(), std::forward<T>(v));
  else
    return {};
}

template <typename T>
inline typename ForestNode<T>::Ref ForestNode<T>::insert_last_child(T &&v)
{
  if (auto *f = forest(); f != nullptr)
    return f->insert_last_child(self(), std::forward<T>(v));
  else
    return {};
}

template <typename T>
inline void ForestNode<T>::erase_subtree()
{
  if (auto *f = forest(); f != nullptr)
    f->erase_subtree(self());
}

template <typename T>
inline void ForestNode<T>::erase_self()
{
  if (auto *f = forest(); f != nullptr && self())
    f->erase(self());
}

template <typename T>
inline typename ForestNode<T>::Ref ForestNode<T>::self() const
{
  return m_self;
}

template <typename T>
inline Forest<T> *ForestNode<T>::forest() const
{
  return m_forest;
}

template <typename T>
inline bool operator==(const ForestNode<T> &a, const ForestNode<T> &b)
{
  return a.self() == b.self();
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
inline void Forest<T>::reserve(size_t size)
{
  m_nodes.reserve(size);
}

template <typename T>
inline size_t Forest<T>::size() const
{
  return m_nodes.size();
}

template <typename T>
inline size_t Forest<T>::capacity() const
{
  return m_nodes.capacity();
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
  newForestNode->m_parent = n;
  newForestNode->m_prev = n->self();
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
  newForestNode->m_parent = n;
  newForestNode->m_prev = n->m_children_end;
  newForestNode->m_next = n->self();
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

  auto self = n->self();
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
        indices.push_back(node.self().index());
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
inline typename Forest<T>::NodeRef Forest<T>::copy_subtree(
    Forest<T>::NodeRef source, Forest<T>::NodeRef newParent)
{
  if (!source || !newParent)
    return NodeRef();

  if (source->isRoot())
    return NodeRef();

  if (source == newParent) {
    newParent = newParent->parent();
  }

  auto newNode = insert_last_child(newParent, T(source->value()));

  if (!source->isLeaf()) {
    auto child = source->next();
    while (child && child != source) {
      copy_subtree(child, newNode);
      child = child->sibling();
    }
  }

  return newNode;
}

template <typename T>
inline bool Forest<T>::isAncestorOf(
    Forest<T>::NodeRef potentialAncestor, Forest<T>::NodeRef node) const
{
  if (!potentialAncestor || !node)
    return false;

  auto current = node;
  while (current.valid()) {
    if (current == potentialAncestor)
      return true;
    current = current->parent();
  }

  return false;
}

template <typename T>
inline void Forest<T>::move_subtree(
    Forest<T>::NodeRef source, Forest<T>::NodeRef newParent)
{
  if (!source || !newParent || source == newParent)
    return;

  if (source->isRoot())
    return;

  if (isAncestorOf(source, newParent))
    return;

  // If already a child of newParent, no-op
  if (source->parent() == newParent)
    return;

  // Detach source from its current parent
  // Use m_next (sibling) directly, not next() which may return first child
  auto self = source->self();
  auto prev = source->prev();
  auto nextSibling = source->sibling();

  const bool prevIsParent = self == prev->m_children_begin;
  if (prevIsParent)
    prev->m_children_begin = nextSibling;
  else
    prev->m_next = nextSibling;

  const bool nextIsParent = self == nextSibling->m_children_end;
  if (nextIsParent)
    nextSibling->m_children_end = prev;
  else
    nextSibling->m_prev = prev;

  // Attach source to newParent as last child
  source->m_parent = newParent;
  source->m_prev = newParent->m_children_end;
  source->m_next = newParent->self();
  if (newParent->isLeaf())
    newParent->m_children_begin = source;
  else
    newParent->m_children_end->m_next = source;
  newParent->m_children_end = source;
}

template <typename T>
inline void Forest<T>::clear()
{
  erase_subtree(m_root);
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
    bool preChildren(tsd::core::ForestNode<T> &node, int level) override
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
    bool preChildren(tsd::core::ForestNode<T> &node, int level) override
    {
      return onEntry(node, level);
    }
    void postChildren(tsd::core::ForestNode<T> &node, int level) override
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
    for (auto s = n->next(); s && s != n; s = s->sibling())
      traverse_impl(s, visitor, level + 1);
  }
  visitor.postChildren(*n, level);
}

template <typename T>
inline typename Forest<T>::NodeRef Forest<T>::make_ForestNode(T &&v)
{
  auto n = m_nodes.emplace(std::forward<T>(v), this);
  n->m_self = n;
  n->m_children_end = n;
  n->m_children_begin = n;
  return n;
}

// Algorithms //

template <typename T, typename FCN>
inline void foreach_child(ForestNodeRef<T> node, FCN &&fcn)
{
  if (auto *forest = node->container(); forest != nullptr) {
    forest->traverse(node, [&](auto &v, int level) {
      if (level != 0)
        fcn(*v);
      return level == 0;
    });
  }
}

template <typename T, typename FCN>
inline void forall_children(ForestNodeRef<T> node, FCN &&fcn)
{
  if (auto *forest = node->container(); forest != nullptr) {
    forest->traverse(node, [&](auto &v, int level) {
      if (level != 0)
        fcn(*v);
      return true;
    });
  }
}

template <typename T, typename PREDICATE>
inline ForestNodeRef<T> find_first_child(ForestNodeRef<T> n, PREDICATE &&p)
{
  for (auto s = n->next(); s && s != n; s = s->sibling()) {
    if (p(**s))
      return s;
  }
  return {};
}

} // namespace tsd::core
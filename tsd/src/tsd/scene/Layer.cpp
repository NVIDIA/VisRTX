// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/scene/Layer.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/scene/Scene.hpp"

namespace tsd::scene {

Layer::Layer(Scene *scene, std::string name)
    : m_scene(scene), m_name(std::move(name))
{}

Layer::~Layer() = default;

Scene *Layer::scene() const
{
  return m_scene;
}

const std::string &Layer::name() const
{
  return m_name;
}

std::string &Layer::editableName()
{
  return m_name;
}

size_t Layer::size() const
{
  return m_tree.size();
}

size_t Layer::capacity() const
{
  return m_tree.capacity();
}

bool Layer::empty() const
{
  return m_tree.empty();
}

LayerNodeRef Layer::root() const
{
  return m_tree.root();
}

LayerNodeRef Layer::at(size_t i) const
{
  return m_tree.at(i);
}

void Layer::erase(LayerNodeRef obj)
{
  m_tree.erase(obj);
}

void Layer::clear()
{
  m_tree.clear();
}

bool Layer::isAncestorOf(
    LayerNodeRef potentialAncestor, LayerNodeRef node) const
{
  return m_tree.isAncestorOf(potentialAncestor, node);
}

void Layer::traverse(LayerNodeRef start, Visitor &visitor)
{
  m_tree.traverse(start, visitor);
}

void Layer::traverse(LayerNodeRef start, VisitorEntryFunction &&f)
{
  m_tree.traverse(start, std::move(f));
}

void Layer::traverse(LayerNodeRef start,
    VisitorEntryFunction &&onNodeEntry,
    VisitorExitFunction &&onNodeExit)
{
  m_tree.traverse(start, std::move(onNodeEntry), std::move(onNodeExit));
}

void Layer::traverse_const(LayerNodeRef start, Visitor &visitor) const
{
  m_tree.traverse_const(start, visitor);
}

void Layer::traverse_const(
    LayerNodeRef start, ConstVisitorEntryFunction &&f) const
{
  m_tree.traverse_const(start, std::move(f));
}

void Layer::traverse_const(LayerNodeRef start,
    ConstVisitorEntryFunction &&onNodeEntry,
    ConstVisitorExitFunction &&onNodeExit) const
{
  m_tree.traverse_const(start, std::move(onNodeEntry), std::move(onNodeExit));
}

} // namespace tsd::scene

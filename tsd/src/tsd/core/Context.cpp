// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Context.hpp"
// std
#include <sstream>

namespace tsd {

std::string objectDBInfo(const ObjectDatabase &db)
{
  std::stringstream ss;
  ss << "OBJECT REGISTRY:\n";
  ss << "      arrays: " << db.array.size() << '\n';
  ss << "  geometries: " << db.geometry.size() << '\n';
  ss << "   materials: " << db.material.size() << '\n';
  ss << "    samplers: " << db.sampler.size() << '\n';
  ss << "     volumes: " << db.volume.size() << '\n';
  ss << "      fields: " << db.field.size() << '\n';
  ss << "      lights: " << db.light.size();
  return ss.str();
}

// Context definitions ////////////////////////////////////////////////////////

Context::Context()
{
  auto defaultMat = createObject<Material>(tokens::material::matte);
  defaultMat->setName("default_material");
}

MaterialRef Context::defaultMaterial() const
{
  return getObject<Material>(0);
}

ArrayRef Context::createArray(
    anari::DataType type, size_t items0, size_t items1, size_t items2)
{
  return createArrayImpl(type, items0, items1, items2, Array::MemoryKind::HOST);
}

ArrayRef Context::createArrayCUDA(
    anari::DataType type, size_t items0, size_t items1, size_t items2)
{
  return createArrayImpl(type, items0, items1, items2, Array::MemoryKind::CUDA);
}

SurfaceRef Context::createSurface(
    const char *name, GeometryRef g, MaterialRef m)
{
  auto surface = createObject<Surface>();
  surface->setGeometry(g);
  surface->setMaterial(m ? m : defaultMaterial());
  surface->setName(name);
  return surface;
}

Object *Context::getObject(const utility::Any &a) const
{
  return getObject(a.type(), a.getAsObjectIndex());
}

Object *Context::getObject(ANARIDataType type, size_t i) const
{
  Object *obj = nullptr;

  switch (type) {
  case ANARI_SURFACE:
    obj = m_db.surface.at(i).data();
    break;
  case ANARI_GEOMETRY:
    obj = m_db.geometry.at(i).data();
    break;
  case ANARI_MATERIAL:
    obj = m_db.material.at(i).data();
    break;
  case ANARI_SAMPLER:
    obj = m_db.sampler.at(i).data();
    break;
  case ANARI_VOLUME:
    obj = m_db.volume.at(i).data();
    break;
  case ANARI_SPATIAL_FIELD:
    obj = m_db.field.at(i).data();
    break;
  case ANARI_LIGHT:
    obj = m_db.light.at(i).data();
    break;
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D:
    obj = m_db.array.at(i).data();
    break;
  default:
    break; // no-op
  }

  return obj;
}

size_t Context::numberOfObjects(anari::DataType type) const
{
  size_t numObjects = 0;

  switch (type) {
  case ANARI_SURFACE:
    numObjects = m_db.surface.capacity();
    break;
  case ANARI_GEOMETRY:
    numObjects = m_db.geometry.capacity();
    break;
  case ANARI_MATERIAL:
    numObjects = m_db.material.capacity();
    break;
  case ANARI_SAMPLER:
    numObjects = m_db.sampler.capacity();
    break;
  case ANARI_VOLUME:
    numObjects = m_db.volume.capacity();
    break;
  case ANARI_SPATIAL_FIELD:
    numObjects = m_db.field.capacity();
    break;
  case ANARI_LIGHT:
    numObjects = m_db.light.capacity();
    break;
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D:
    numObjects = m_db.array.capacity();
    break;
  default:
    break; // no-op
  }

  return numObjects;
}

void Context::removeObject(const utility::Any &o)
{
  if (auto *optr = getObject(o.type(), o.getAsObjectIndex()); optr)
    removeObject(*optr);
}

void Context::removeObject(const Object &o)
{
  if (m_updateDelegate)
    m_updateDelegate->signalObjectRemoved(&o);

  const auto type = o.type();
  const auto index = o.index();

  switch (type) {
  case ANARI_SURFACE:
    m_db.surface.erase(index);
    break;
  case ANARI_GEOMETRY:
    m_db.geometry.erase(index);
    break;
  case ANARI_MATERIAL:
    m_db.material.erase(index);
    break;
  case ANARI_SAMPLER:
    m_db.sampler.erase(index);
    break;
  case ANARI_VOLUME:
    m_db.volume.erase(index);
    break;
  case ANARI_SPATIAL_FIELD:
    m_db.field.erase(index);
    break;
  case ANARI_LIGHT:
    m_db.light.erase(index);
    break;
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D:
    m_db.array.erase(index);
    break;
  default:
    break; // no-op
  }
}

void Context::removeAllObjects()
{
  if (m_updateDelegate)
    m_updateDelegate->signalRemoveAllObjects();

  tree.erase_subtree(tree.root());

  m_db.array.clear();
  m_db.surface.clear();
  m_db.geometry.clear();
  m_db.material.clear();
  m_db.sampler.clear();
  m_db.volume.clear();
  m_db.field.clear();
  m_db.light.clear();
}

BaseUpdateDelegate *Context::updateDelegate() const
{
  return m_updateDelegate;
}

void Context::setUpdateDelegate(BaseUpdateDelegate *ud)
{
  m_updateDelegate = ud;

  auto setDelegateOnObjects = [&](auto &array) {
    foreach_item_const(array, [&](auto *o) {
      if (o)
        o->setUpdateDelegate(ud);
    });
  };

  setDelegateOnObjects(m_db.array);
  setDelegateOnObjects(m_db.light);
  setDelegateOnObjects(m_db.surface);
  setDelegateOnObjects(m_db.geometry);
  setDelegateOnObjects(m_db.material);
  setDelegateOnObjects(m_db.sampler);
  setDelegateOnObjects(m_db.volume);
  setDelegateOnObjects(m_db.field);
}

const ObjectDatabase &Context::objectDB() const
{
  return m_db;
}

InstanceNode::Ref Context::insertChildNode(
    InstanceNode::Ref parent, const char *name)
{
  auto inst = tree.insert_last_child(parent, tsd::utility::Any{});
  (*inst)->name = name;
  return inst;
}

InstanceNode::Ref Context::insertChildTransformNode(
    InstanceNode::Ref parent, mat4 xfm, const char *name)
{
  auto inst = tree.insert_last_child(parent, tsd::utility::Any{xfm});
  (*inst)->name = name;
  signalInstanceTreeChange();
  return inst;
}

void Context::removeInstancedObject(InstanceNode::Ref obj)
{
  if (obj->isRoot())
    return;

  std::vector<InstanceNode::Ref> objects;
  tree.traverse(obj, [&](auto &node, int level) {
    if (node.isLeaf())
      objects.push_back(tree.at(node.index()));
    return true;
  });

  for (auto &o : objects)
    removeObject(o->value().value);

  tree.erase(obj);

  signalInstanceTreeChange();
}

void Context::signalInstanceTreeChange()
{
  if (m_updateDelegate)
    m_updateDelegate->signalInstanceStructureChanged();
}

ArrayRef Context::createArrayImpl(anari::DataType type,
    size_t items0,
    size_t items1,
    size_t items2,
    Array::MemoryKind kind)
{
  ArrayRef retval;

  if (items2 != 0)
    retval = m_db.array.emplace(type, items0, items1, items2, kind);
  else if (items1 != 0)
    retval = m_db.array.emplace(type, items0, items1, kind);
  else
    retval = m_db.array.emplace(type, items0, kind);

  retval->m_context = this;
  retval->m_index = retval.index();

  if (m_updateDelegate) {
    retval->setUpdateDelegate(m_updateDelegate);
    m_updateDelegate->signalObjectAdded(retval.data());
  }

  return retval;
}

} // namespace tsd

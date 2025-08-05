// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/Context.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <sstream>

namespace tsd::core {

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
  addLayer("default");
  createObject<Material>(tokens::material::matte)->setName("default_material");
}

MaterialRef Context::defaultMaterial() const
{
  return getObject<Material>(0);
}

Layer *Context::defaultLayer() const
{
  return layer(0);
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

Object *Context::getObject(const Any &a) const
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

void Context::removeObject(const Any &o)
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

  removeAllSecondaryLayers();
  defaultLayer()->root()->erase_subtree();

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

const LayerMap &Context::layers() const
{
  return m_layers;
}

Layer *Context::layer(size_t i) const
{
  return m_layers.at_index(i).second.get();
}

Layer *Context::addLayer(Token name)
{
  auto &l = m_layers[name];
  if (!l)
    l.reset(new Layer({tsd::math::mat4(tsd::math::identity), "root"}));
  if (m_updateDelegate)
    m_updateDelegate->signalLayerAdded(l.get());
  return l.get();
}

void Context::removeLayer(Token name)
{
  if (!m_layers.contains(name))
    return;
  if (m_updateDelegate)
    m_updateDelegate->signalLayerRemoved(m_layers[name].get());
  m_layers.erase(name);
}

void Context::removeLayer(const Layer *layer)
{
  for (size_t i = 0; i < m_layers.size(); i++) {
    if (m_layers.at_index(i).second.get() == layer) {
      if (m_updateDelegate)
        m_updateDelegate->signalLayerRemoved(m_layers.at_index(i).second.get());
      m_layers.erase(i);
      return;
    }
  }
}

LayerNodeRef Context::insertChildNode(LayerNodeRef parent, const char *name)
{
  auto *layer = parent->container();
  auto inst = layer->insert_last_child(parent, tsd::core::Any{});
  (*inst)->name = name;
  return inst;
}

LayerNodeRef Context::insertChildTransformNode(
    LayerNodeRef parent, mat4 xfm, const char *name)
{
  auto *layer = parent->container();
  auto inst = layer->insert_last_child(parent, tsd::core::Any{xfm});
  (*inst)->name = name;
  signalLayerChange(parent->container());
  return inst;
}

LayerNodeRef Context::insertChildObjectNode(
    LayerNodeRef parent, anari::DataType type, size_t idx, const char *name)
{
  auto inst = parent->insert_last_child(tsd::core::Any{type, idx});
  (*inst)->name = name;
  signalLayerChange(parent->container());
  return inst;
}

void Context::removeInstancedObject(
    LayerNodeRef obj, bool deleteReferencedObjects)
{
  if (obj->isRoot())
    return;

  auto *layer = obj->container();

  if (deleteReferencedObjects) {
    std::vector<LayerNodeRef> objects;

    layer->traverse(obj, [&](auto &node, int level) {
      if (node.isLeaf())
        objects.push_back(layer->at(node.index()));
      return true;
    });

    for (auto &o : objects)
      removeObject(o->value().value);
  }

  layer->erase(obj);
  signalLayerChange(layer);
}

void Context::signalLayerChange(const Layer *l)
{
  if (m_updateDelegate)
    m_updateDelegate->signalLayerUpdated(l);
}

void Context::defragmentObjectStorage()
{
  FlatMap<anari::DataType, bool> defragmentations;

  // Defragment object storage and stash whether something happened //

  bool defrag = false;

  defrag |= defragmentations[ANARI_ARRAY] = m_db.array.defragment();
  defrag |= defragmentations[ANARI_SURFACE] = m_db.surface.defragment();
  defrag |= defragmentations[ANARI_GEOMETRY] = m_db.geometry.defragment();
  defrag |= defragmentations[ANARI_MATERIAL] = m_db.material.defragment();
  defrag |= defragmentations[ANARI_SAMPLER] = m_db.sampler.defragment();
  defrag |= defragmentations[ANARI_VOLUME] = m_db.volume.defragment();
  defrag |= defragmentations[ANARI_SPATIAL_FIELD] = m_db.field.defragment();
  defrag |= defragmentations[ANARI_LIGHT] = m_db.light.defragment();

  if (!defrag) {
    tsd::core::logStatus("No defragmentation needed");
    return;
  } else {
    tsd::core::logStatus("Defragmenting context arrays:");
    for (const auto &pair : defragmentations) {
      if (pair.second)
        tsd::core::logStatus("    --> %s", anari::toString(pair.first));
    }
  }

  // Function to find the object holding an index and returning the new index //

  auto getUpdatedIndex = [&](anari::DataType objType, size_t idx) -> size_t {
    auto findIdx = [](const auto &a, size_t i) {
      auto ref = find_item_if(a, [&](auto *o) { return o->index() == i; });
      return ref ? ref.index() : INVALID_INDEX;
    };

    size_t newObjIndex = INVALID_INDEX;
    switch (objType) {
    case ANARI_SURFACE:
      return findIdx(m_db.surface, idx);
    case ANARI_GEOMETRY:
      return findIdx(m_db.geometry, idx);
    case ANARI_MATERIAL:
      return findIdx(m_db.material, idx);
    case ANARI_SAMPLER:
      return findIdx(m_db.sampler, idx);
    case ANARI_VOLUME:
      return findIdx(m_db.volume, idx);
    case ANARI_SPATIAL_FIELD:
      return findIdx(m_db.field, idx);
    case ANARI_LIGHT:
      return findIdx(m_db.light, idx);
    case ANARI_ARRAY:
    case ANARI_ARRAY1D:
    case ANARI_ARRAY2D:
    case ANARI_ARRAY3D:
      return findIdx(m_db.array, idx);
    default:
      break; // no-op
    }

    return INVALID_INDEX;
  };

  // Function to update indices to objects in layer nodes //

  std::vector<LayerNode *> toErase;
  auto updateLayerObjReferenceIndices = [&](Layer &layer) {
    layer.traverse(layer.root(), [&](LayerNode &node, int /*level*/) {
      if (!node->isObject())
        return true;
      auto objType = node->value.type();
      if (!defragmentations[objType])
        return true;

      size_t newIdx = getUpdatedIndex(objType, node->value.getAsObjectIndex());
      if (newIdx != INVALID_INDEX)
        node->value = Any(objType, newIdx);
      else
        toErase.push_back(&node);

      return true;
    });
  };

  // Invoke above function on all layers//

  for (auto itr = m_layers.begin(); itr != m_layers.end(); itr++)
    updateLayerObjReferenceIndices(*itr->second);

  for (auto *ln : toErase)
    ln->erase_self();
  toErase.clear();

  // Function to update indices to objects on object parameters //

  auto updateParameterReferences = [&](auto &array) {
    foreach_item(array, [&](Object *o) {
      if (!o)
        return;
      for (size_t i = 0; i < o->numParameters(); i++) {
        auto &p = o->parameterAt(i);
        const auto &v = p.value();
        if (!v.holdsObject())
          continue;
        auto objType = v.type();
        if (!defragmentations[objType])
          continue;

        auto newIdx = getUpdatedIndex(objType, v.getAsObjectIndex());
        p.setValue(newIdx != INVALID_INDEX ? Any(objType, newIdx) : Any());
      }
    });
  };

  // Invoke above function on all object arrays //

  updateParameterReferences(m_db.array);
  updateParameterReferences(m_db.surface);
  updateParameterReferences(m_db.geometry);
  updateParameterReferences(m_db.material);
  updateParameterReferences(m_db.sampler);
  updateParameterReferences(m_db.volume);
  updateParameterReferences(m_db.field);
  updateParameterReferences(m_db.light);

  // Function to update all self-held index values to the new actual index //

  auto updateObjectHeldIndex = [&](auto &array) {
    foreach_item_ref(array, [&](auto ref) {
      if (!ref)
        return;
      ref->m_index = ref.index();
    });
  };

  // Invoke above function on all object arrays //

  updateObjectHeldIndex(m_db.array);
  updateObjectHeldIndex(m_db.surface);
  updateObjectHeldIndex(m_db.geometry);
  updateObjectHeldIndex(m_db.material);
  updateObjectHeldIndex(m_db.sampler);
  updateObjectHeldIndex(m_db.volume);
  updateObjectHeldIndex(m_db.field);
  updateObjectHeldIndex(m_db.light);

  // Signal updates to any delegates //
  if (m_updateDelegate)
    m_updateDelegate->signalInvalidateCachedObjects();
}

void Context::removeUnusedObjects()
{
  tsd::core::logStatus("Removing unused context objects");

  FlatMap<anari::DataType, std::vector<int>> usages;

  usages[ANARI_ARRAY].resize(m_db.array.capacity(), 0);
  usages[ANARI_SURFACE].resize(m_db.surface.capacity(), 0);
  usages[ANARI_GEOMETRY].resize(m_db.geometry.capacity(), 0);
  usages[ANARI_MATERIAL].resize(m_db.material.capacity(), 0);
  usages[ANARI_SAMPLER].resize(m_db.sampler.capacity(), 0);
  usages[ANARI_VOLUME].resize(m_db.volume.capacity(), 0);
  usages[ANARI_SPATIAL_FIELD].resize(m_db.field.capacity(), 0);
  usages[ANARI_LIGHT].resize(m_db.light.capacity(), 0);

  // Always keep around the default material //

  if (!usages[ANARI_MATERIAL].empty())
    usages[ANARI_MATERIAL][0] = 1;

  // Function to count object references in layers //

  auto countLayerObjReferenceIndices = [&](Layer &layer) {
    layer.traverse(layer.root(), [&](LayerNode &node, int /*level*/) {
      if (node->isObject()) {
        auto objType = node->value.type();
        if (anari::isArray(objType))
          objType = ANARI_ARRAY;
        auto idx = node->value.getAsObjectIndex();
        if (idx != INVALID_INDEX)
          usages[objType][idx]++;
      }

      return true;
    });
  };

  // Function to count object references in object parameters //

  auto countParameterReferences = [&](auto &array) {
    foreach_item(array, [&](Object *o) {
      if (!o)
        return;
      for (size_t i = 0; i < o->numParameters(); i++) {
        auto &p = o->parameterAt(i);
        const auto &v = p.value();
        if (!v.holdsObject())
          continue;
        auto objType = v.type();
        if (anari::isArray(objType))
          objType = ANARI_ARRAY;
        auto idx = v.getAsObjectIndex();
        if (idx != INVALID_INDEX)
          usages[objType][idx]++;
      }
    });
  };

  // Function to remove unused objects from object arrays //

  auto removeUnused = [&](auto &array) {
    foreach_item_ref(array, [&](auto ref) {
      if (!ref)
        return;
      auto objType = ref->type();
      if (anari::isArray(objType))
        objType = ANARI_ARRAY;
      if (usages[objType][ref.index()] <= 0) {
        // Decrement reference counts for this object's object parameters
        for (size_t i = 0; i < ref->numParameters(); i++) {
          auto &p = ref->parameterAt(i);
          const auto &v = p.value();
          if (!v.holdsObject())
            continue;
          auto objType = v.type();
          if (anari::isArray(objType))
            objType = ANARI_ARRAY;
          auto idx = v.getAsObjectIndex();
          if (idx != INVALID_INDEX)
            usages[objType][idx]--;
        }

        removeObject(*ref);
      }
    });
  };

  // Invoke above functions on all object arrays, top-down //

  for (auto itr = m_layers.begin(); itr != m_layers.end(); itr++)
    countLayerObjReferenceIndices(*itr->second);

  countParameterReferences(m_db.surface);
  countParameterReferences(m_db.volume);
  countParameterReferences(m_db.light);
  countParameterReferences(m_db.geometry);
  countParameterReferences(m_db.material);
  countParameterReferences(m_db.field);
  countParameterReferences(m_db.sampler);
  countParameterReferences(m_db.array);

  removeUnused(m_db.surface);
  removeUnused(m_db.volume);
  removeUnused(m_db.light);
  removeUnused(m_db.geometry);
  removeUnused(m_db.material);
  removeUnused(m_db.field);
  removeUnused(m_db.sampler);
  removeUnused(m_db.array);
}

void Context::removeAllSecondaryLayers()
{
  for (auto itr = m_layers.begin() + 1; itr != m_layers.end(); itr++) {
    if (m_updateDelegate)
      m_updateDelegate->signalLayerRemoved(itr->second.get());
  }

  m_layers.shrink(1);
}

ArrayRef Context::createArrayImpl(anari::DataType type,
    size_t items0,
    size_t items1,
    size_t items2,
    Array::MemoryKind kind)
{
  if (items0 + items1 + items2 == 0) {
    tsd::core::logWarning("Not creating an array with zero elements");
    return {};
  }

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

} // namespace tsd::core

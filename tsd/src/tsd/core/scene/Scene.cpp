// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/Scene.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/ObjectUsePtr.hpp"
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
  ss << "      lights: " << db.light.size() << '\n';
  ss << "     cameras: " << db.camera.size();
  return ss.str();
}

// Scene definitions //////////////////////////////////////////////////////////

Scene::Scene()
{
  createObject<Material>(tokens::material::matte)->setName("default_material");
}

Scene::~Scene()
{
  m_updateDelegate = nullptr;
  m_layers.clear();
  m_animations.objects.clear();

  auto reportObjectUsages = [&](auto &array) {
    foreach_item_const(array, [&](auto *o) {
      if (!o || o->totalUseCount() == 0)
        return;

      if (o->type() == ANARI_MATERIAL && o->index() == 0)
        return;

      logWarning(
          "Scene::~Scene(): object of type %s, index [%zu], and name '%s' has"
          " non-zero use count of %zu at scene destruction",
          anari::toString(o->type()),
          o->index(),
          o->name().c_str(),
          o->totalUseCount());
    });
    array.clear();
  };

  reportObjectUsages(m_db.light);
  reportObjectUsages(m_db.surface);
  reportObjectUsages(m_db.volume);
  reportObjectUsages(m_db.geometry);
  reportObjectUsages(m_db.material);
  reportObjectUsages(m_db.sampler);
  reportObjectUsages(m_db.field);
  reportObjectUsages(m_db.array);
  reportObjectUsages(m_db.camera);
}

MaterialRef Scene::defaultMaterial() const
{
  return getObject<Material>(0);
}

Layer *Scene::defaultLayer()
{
  if (m_layers.empty())
    addLayer("default");
  return layer(0);
}

int Scene::mpiRank() const
{
  return m_mpi.rank;
}

int Scene::mpiNumRanks() const
{
  return m_mpi.numRanks;
}

void Scene::setMpiRankInfo(int rank, int numRanks)
{
  if (rank < 0 || numRanks <= 0 || rank >= numRanks) {
    logWarning(
        "[Scene::setMpiRankInfo()] invalid MPI rank (%d) or number of "
        "ranks (%d); ignoring",
        rank,
        numRanks);
    return;
  } else if (m_mpi.numRanks > 1) {
    logError("[Scene::setMpiRankInfo()] MPI rank info already set; ignoring");
    return;
  }

  m_mpi.rank = rank;
  m_mpi.numRanks = numRanks;
}

ArrayRef Scene::createArray(
    anari::DataType type, size_t items0, size_t items1, size_t items2)
{
  return createArrayImpl(type, items0, items1, items2, Array::MemoryKind::HOST);
}

ArrayRef Scene::createArrayCUDA(
    anari::DataType type, size_t items0, size_t items1, size_t items2)
{
  return createArrayImpl(type, items0, items1, items2, Array::MemoryKind::CUDA);
}

SurfaceRef Scene::createSurface(const char *name, GeometryRef g, MaterialRef m)
{
  auto surface = createObject<Surface>();
  surface->setGeometry(g);
  surface->setMaterial(m ? m : defaultMaterial());
  surface->setName(name);
  return surface;
}

Object *Scene::getObject(const Any &a) const
{
  return getObject(a.type(), a.getAsObjectIndex());
}

Object *Scene::getObject(ANARIDataType type, size_t i) const
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
  case ANARI_CAMERA:
    obj = m_db.camera.at(i).data();
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

size_t Scene::numberOfObjects(anari::DataType type) const
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
  case ANARI_CAMERA:
    numObjects = m_db.camera.capacity();
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

void Scene::removeObject(const Any &o)
{
  if (auto *optr = getObject(o.type(), o.getAsObjectIndex()); optr)
    removeObject(optr);
}

void Scene::removeObject(const Object *_o)
{
  if (!_o)
    return;

  auto &o = *_o;

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
  case ANARI_CAMERA:
    m_db.camera.erase(index);
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

void Scene::removeAllObjects()
{
  if (m_updateDelegate)
    m_updateDelegate->signalRemoveAllObjects();

  removeAllLayers();

  m_db.array.clear();
  m_db.surface.clear();
  m_db.geometry.clear();
  m_db.material.clear();
  m_db.sampler.clear();
  m_db.volume.clear();
  m_db.field.clear();
  m_db.light.clear();
  m_db.camera.clear();
}

BaseUpdateDelegate *Scene::updateDelegate() const
{
  return m_updateDelegate;
}

void Scene::setUpdateDelegate(BaseUpdateDelegate *ud)
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
  setDelegateOnObjects(m_db.camera);
}

const ObjectDatabase &Scene::objectDB() const
{
  return m_db;
}

const LayerMap &Scene::layers() const
{
  return m_layers;
}

size_t Scene::numberOfLayers() const
{
  return m_layers.size();
}

Layer *Scene::layer(Token name) const
{
  auto *ls = m_layers.at(name);
  return ls ? ls->ptr.get() : nullptr;
}

Layer *Scene::layer(size_t i) const
{
  return m_layers.at_index(i).second.ptr.get();
}

Layer *Scene::addLayer(Token name)
{
  auto &ls = m_layers[name];
  if (!ls.ptr) {
    ls.ptr.reset(new Layer({tsd::math::mat4(tsd::math::identity), "root"}));
    if (m_updateDelegate)
      m_updateDelegate->signalLayerAdded(ls.ptr.get());
    m_numActiveLayers++;
  }
  return ls.ptr.get();
}

bool Scene::layerIsActive(Token name) const
{
  auto *ls = m_layers.at(name);
  return ls ? ls->active : false;
}

void Scene::setLayerActive(Token name, bool active)
{
  if (auto *ls = m_layers.at(name); ls && ls->active != active) {
    m_numActiveLayers += active ? 1 : -1;
    ls->active = active;
    signalActiveLayersChanged();
  }
}

void Scene::setOnlyLayerActive(Token name)
{
  if (auto *ls = m_layers.at(name); ls) {
    for (auto &ls : m_layers)
      ls.second.active = false;

    m_numActiveLayers = 1;
    ls->active = true;
    signalActiveLayersChanged();
  }
}

void Scene::setAllLayersActive()
{
  for (auto &ls : m_layers)
    ls.second.active = true;
  m_numActiveLayers = m_layers.size();
  signalActiveLayersChanged();
}

std::vector<const Layer *> Scene::getActiveLayers() const
{
  std::vector<const Layer *> activeLayers;
  if (numberOfActiveLayers() != numberOfLayers()) {
    activeLayers.reserve(m_layers.size());
    for (const auto &ls : m_layers) {
      if (ls.second.active)
        activeLayers.push_back(ls.second.ptr.get());
    }
  }
  return activeLayers;
}

size_t Scene::numberOfActiveLayers() const
{
  return m_numActiveLayers;
}

void Scene::removeLayer(Token name)
{
  if (!m_layers.contains(name))
    return;
  if (m_updateDelegate)
    m_updateDelegate->signalLayerRemoved(m_layers[name].ptr.get());
  if (m_layers[name].active)
    m_numActiveLayers--;
  m_layers.erase(name);
}

void Scene::removeLayer(const Layer *layer)
{
  for (size_t i = 0; i < m_layers.size(); i++) {
    if (m_layers.at_index(i).second.ptr.get() == layer) {
      if (m_updateDelegate) {
        m_updateDelegate->signalLayerRemoved(
            m_layers.at_index(i).second.ptr.get());
      }
      m_layers.erase(i);
      return;
    }
  }
}

LayerNodeRef Scene::insertChildNode(LayerNodeRef parent, const char *name)
{
  auto *layer = parent->container();
  auto inst = layer->insert_last_child(parent, {});
  (*inst)->name() = name;
  return inst;
}

LayerNodeRef Scene::insertChildTransformNode(
    LayerNodeRef parent, mat4 xfm, const char *name)
{
  auto *layer = parent->container();
  auto inst = layer->insert_last_child(parent, xfm);
  (*inst)->name() = name;
  signalLayerChange(parent->container());
  return inst;
}

LayerNodeRef Scene::insertChildObjectNode(
    LayerNodeRef parent, anari::DataType type, size_t idx, const char *name)
{
  auto inst = parent->insert_last_child({type, idx, this});
  (*inst)->name() = name;
  signalLayerChange(parent->container());
  return inst;
}

void Scene::removeInstancedObject(
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
      removeObject(o->value().getObject());
  }

  layer->erase(obj);
  signalLayerChange(layer);
}

void Scene::signalLayerChange(const Layer *l)
{
  if (m_updateDelegate)
    m_updateDelegate->signalLayerUpdated(l);
}

void Scene::signalActiveLayersChanged()
{
  if (m_updateDelegate)
    m_updateDelegate->signalActiveLayersChanged();
}

void Scene::signalObjectParameterUseCountZero(const Object *obj)
{
  if (m_updateDelegate)
    m_updateDelegate->signalObjectParameterUseCountZero(obj);
}

void Scene::signalObjectLayerUseCountZero(const Object *obj)
{
  if (m_updateDelegate)
    m_updateDelegate->signalObjectLayerUseCountZero(obj);
}

Animation *Scene::addAnimation(const char *name)
{
  auto anim = std::unique_ptr<Animation>(new Animation(this, name));
  auto *retval = anim.get();
  m_animations.objects.push_back(std::move(anim));
  return retval;
}

size_t Scene::numberOfAnimations() const
{
  return m_animations.objects.size();
}

Animation *Scene::animation(size_t i) const
{
  if (i < m_animations.objects.size())
    return m_animations.objects[i].get();
  return nullptr;
}

void Scene::removeAnimation(Animation *a)
{
  auto itr = std::find_if(m_animations.objects.begin(),
      m_animations.objects.end(),
      [&](auto &anim) { return anim.get() == a; });
  if (itr != m_animations.objects.end())
    m_animations.objects.erase(itr);
}

void Scene::removeAllAnimations()
{
  m_animations.objects.clear();
}

void Scene::setAnimationTime(float time)
{
  m_animations.time = time;
  for (auto &a : m_animations.objects)
    a->update(time);
  
  // Signal delegates that animation time changed
  if (m_updateDelegate)
    m_updateDelegate->signalAnimationTimeChanged(time);
}

float Scene::getAnimationTime() const
{
  return m_animations.time;
}

void Scene::setAnimationIncrement(float increment)
{
  m_animations.incrementSize = increment;
  if (increment > 0.5f) {
    logWarning(
        "[scene] setting animation increment > 0.5 will cause odd"
        " animation behavior.");
  }
}

float Scene::getAnimationIncrement() const
{
  return m_animations.incrementSize;
}

void Scene::incrementAnimationTime()
{
  auto newTime = m_animations.time + m_animations.incrementSize;
  if (newTime > 1.f)
    newTime = 0.f;
  setAnimationTime(newTime);
}

void Scene::removeUnusedObjects()
{
  tsd::core::logStatus("Removing unused context objects");

  // Always keep around the default material //
  ObjectUsePtr<Material> defaultMat = getObject<Material>(0).data();

  auto removeUnused = [&](auto &array) {
    foreach_item_ref(array, [&](auto ref) {
      if (!ref)
        return;
      if (auto *obj = ref.data(); obj && obj->totalUseCount() == 0)
        removeObject(obj);
    });
  };

  removeUnused(m_db.surface);
  removeUnused(m_db.volume);
  removeUnused(m_db.light);
  removeUnused(m_db.geometry);
  removeUnused(m_db.material);
  removeUnused(m_db.field);
  removeUnused(m_db.sampler);
  removeUnused(m_db.array);
}

void Scene::defragmentObjectStorage()
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
  defrag |= defragmentations[ANARI_CAMERA] = m_db.camera.defragment();

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
    case ANARI_CAMERA:
      return findIdx(m_db.camera, idx);
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
      auto objType = anari::isArray(node->type()) ? ANARI_ARRAY : node->type();
      if (!defragmentations[objType])
        return true;

      size_t newIdx = getUpdatedIndex(objType, node->getObjectIndex());
      if (newIdx != INVALID_INDEX)
        node->setAsObject(node->type(), newIdx, this);
      else
        toErase.push_back(&node);

      return true;
    });
  };

  // Invoke above function on all layers//

  for (auto itr = m_layers.begin(); itr != m_layers.end(); itr++)
    updateLayerObjReferenceIndices(*itr->second.ptr);

  for (auto *ln : toErase)
    ln->erase_self();
  if (!toErase.empty()) {
    tsd::core::logStatus(
        "    Removed %zu layer nodes referencing deleted objects",
        toErase.size());
  }
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
        auto objType = anari::isArray(v.type()) ? ANARI_ARRAY : v.type();
        if (!defragmentations[objType])
          continue;

        auto newIdx = getUpdatedIndex(objType, v.getAsObjectIndex());
        Any newValue = newIdx != INVALID_INDEX ? Any(v.type(), newIdx) : Any();
        p.m_value = newValue; // we don't want refcount changes, essentially
                              // this is move semantics
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
  updateParameterReferences(m_db.camera);

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
  updateObjectHeldIndex(m_db.camera);

  // Signal updates to any delegates //
  if (m_updateDelegate)
    m_updateDelegate->signalInvalidateCachedObjects();
}

void Scene::cleanupScene()
{
  removeUnusedObjects();
  defragmentObjectStorage();
}

void Scene::removeAllLayers()
{
  for (auto itr = m_layers.begin(); itr != m_layers.end(); itr++) {
    if (m_updateDelegate)
      m_updateDelegate->signalLayerRemoved(itr->second.ptr.get());
  }

  m_layers.clear();
}

ArrayRef Scene::createArrayImpl(anari::DataType type,
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

  retval->m_scene = this;
  retval->m_index = retval.index();

  if (m_updateDelegate) {
    retval->setUpdateDelegate(m_updateDelegate);
    m_updateDelegate->signalObjectAdded(retval.data());
  }

  return retval;
}

} // namespace tsd::core

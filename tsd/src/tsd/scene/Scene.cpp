// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/scene/Scene.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/scene/ObjectUsePtr.hpp"
// std
#include <algorithm>
#include <sstream>
#if defined(__cpp_rtti) || defined(__GXX_RTTI) || defined(_CPPRTTI)
#include <typeinfo>
#endif

namespace tsd::scene {

namespace {

const char *delegateTypeName(const BaseUpdateDelegate &delegate)
{
#if defined(__cpp_rtti) || defined(__GXX_RTTI) || defined(_CPPRTTI)
  return typeid(delegate).name();
#else
  return "<rtti unavailable>";
#endif
}

void logRemainingDelegates(const MultiUpdateDelegate &delegate)
{
  if (delegate.size() == 0)
    return;

  logError("Scene::~Scene(): %zu update delegate(s) still registered",
      delegate.size());

  for (size_t i = 0; i < delegate.size(); i++) {
    auto *child = delegate.get(i);
    logError("  delegate[%zu]: type=%s ptr=%p",
        i,
        child ? delegateTypeName(*child) : "<null>",
        (const void *)child);
  }
}

} // namespace

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
  ss << "     cameras: " << db.camera.size() << '\n';
  ss << "   renderers: " << db.renderer.size();
  return ss.str();
}

// Scene definitions //////////////////////////////////////////////////////////

Scene::Scene()
{
  defaultMaterial();
  defaultCamera();

  // Layer-node object reference patching
  addDefragCallback([this](const IndexRemapper &remap) {
    for (auto itr = m_layers.begin(); itr != m_layers.end(); itr++) {
      auto &layer = *itr->second.ptr;
      layer.traverse(layer.root(), [&](LayerNode &node, int) {
        if (!node->isObject())
          return true;
        size_t newIdx = remap(node->type(), node->getObjectIndex());
        node->setValueRaw({node->type(), newIdx}, true);
        return true;
      });
    }
  });

  // Object parameter reference patching
  addDefragCallback([this](const IndexRemapper &remap) {
    auto updateParams = [&](auto &array) {
      foreach_item(array, [&](Object *o) {
        if (!o)
          return;
        for (size_t i = 0; i < o->numParameters(); i++) {
          auto &p = o->parameterAt(i);
          const auto &v = p.value();
          if (!v.holdsObject())
            continue;
          size_t newIdx = remap(v.type(), v.getAsObjectIndex());
          p.m_value = Any(v.type(), newIdx);
        }
      });
    };

    updateParams(m_db.array);
    updateParams(m_db.surface);
    updateParams(m_db.geometry);
    updateParams(m_db.material);
    updateParams(m_db.sampler);
    updateParams(m_db.volume);
    updateParams(m_db.field);
    updateParams(m_db.light);
    updateParams(m_db.camera);
    updateParams(m_db.renderer);
  });
}

Scene::~Scene()
{
  logRemainingDelegates(m_updateDelegate);
  m_updateDelegate.clear();

  m_defaultObjects.material.reset();
  m_defaultObjects.camera.reset();
  m_layers.clear();

  auto reportObjectUsages = [&](auto &array) {
    foreach_item_const(array, [&](auto *o) {
      if (!o || o->totalUseCount() == 0)
        return;

      logWarning(
          "Scene::~Scene(): object of type %s, index [%zu], and name '%s' has"
          " non-zero use count of (%zu) --> "
          " [app(%zu) | param(%zu) | layer(%zu) | anim(%zu) | internal(%zu)]"
          " at scene destruction",
          anari::toString(o->type()),
          o->index(),
          o->name().c_str(),
          o->totalUseCount(),
          o->useCount(Object::UseKind::APP),
          o->useCount(Object::UseKind::PARAMETER),
          o->useCount(Object::UseKind::LAYER),
          o->useCount(Object::UseKind::ANIM),
          o->useCount(Object::UseKind::INTERNAL));
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
  reportObjectUsages(m_db.renderer);
}

MaterialRef Scene::defaultMaterial()
{
  if (!m_defaultObjects.material) {
    if (numberOfObjects(ANARI_MATERIAL) == 0) {
      m_defaultObjects.material =
          createObject<Material>(tokens::material::matte);
      m_defaultObjects.material->setName("default");
    } else
      m_defaultObjects.material = getObject<Material>(0);
  }
  return m_defaultObjects.material.ref();
}

CameraRef Scene::defaultCamera()
{
  if (!m_defaultObjects.camera) {
    if (numberOfObjects(ANARI_CAMERA) == 0) {
      m_defaultObjects.camera =
          createObject<Camera>(tokens::camera::perspective);
      m_defaultObjects.camera->setName("default");
    } else
      m_defaultObjects.camera = getObject<Camera>(0);
  }
  return m_defaultObjects.camera.ref();
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

Object *Scene::createObject(anari::DataType type, Token subtype)
{
  Object *obj = nullptr;

  switch (type) {
  case ANARI_SURFACE:
    obj = createObjectImpl(m_db.surface).data();
    break;
  case ANARI_GEOMETRY:
    obj = createObjectImpl(m_db.geometry, subtype).data();
    break;
  case ANARI_MATERIAL:
    obj = createObjectImpl(m_db.material, subtype).data();
    break;
  case ANARI_SAMPLER:
    obj = createObjectImpl(m_db.sampler, subtype).data();
    break;
  case ANARI_VOLUME:
    obj = createObjectImpl(m_db.volume, subtype).data();
    break;
  case ANARI_SPATIAL_FIELD:
    obj = createObjectImpl(m_db.field, subtype).data();
    break;
  case ANARI_LIGHT:
    obj = createObjectImpl(m_db.light, subtype).data();
    break;
  case ANARI_CAMERA:
    obj = createObjectImpl(m_db.camera, subtype).data();
    break;
  default:
    logError("[Scene::createObject(type, subtype)] unsupported object type %s",
        anari::toString(type));
    break;
  }

  return obj;
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

ArrayRef Scene::createArrayProxy(
    anari::DataType type, size_t items0, size_t items1, size_t items2)
{
  return createArrayImpl(
      type, items0, items1, items2, Array::MemoryKind::PROXY);
}

SurfaceRef Scene::createSurface(const char *name, GeometryRef g, MaterialRef m)
{
  auto surface = createObjectImpl(m_db.surface);
  if (g)
    surface->setGeometry(g);
  surface->setMaterial(m ? m : defaultMaterial());
  surface->setName(name);
  return surface;
}

Object *Scene::getObject(const Any &a) const
{
  return getObject(a.type(), a.getAsObjectIndex());
}

Object *Scene::getObject(anari::DataType type, size_t i) const
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
  case ANARI_RENDERER:
    obj = m_db.renderer.at(i).data();
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
    numObjects = m_db.surface.size();
    break;
  case ANARI_GEOMETRY:
    numObjects = m_db.geometry.size();
    break;
  case ANARI_MATERIAL:
    numObjects = m_db.material.size();
    break;
  case ANARI_SAMPLER:
    numObjects = m_db.sampler.size();
    break;
  case ANARI_VOLUME:
    numObjects = m_db.volume.size();
    break;
  case ANARI_SPATIAL_FIELD:
    numObjects = m_db.field.size();
    break;
  case ANARI_LIGHT:
    numObjects = m_db.light.size();
    break;
  case ANARI_CAMERA:
    numObjects = m_db.camera.size();
    break;
  case ANARI_RENDERER:
    numObjects = m_db.renderer.size();
    break;
  case ANARI_ARRAY:
  case ANARI_ARRAY1D:
  case ANARI_ARRAY2D:
  case ANARI_ARRAY3D:
    numObjects = m_db.array.size();
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

  m_updateDelegate.signalObjectRemoved(&o);

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
  case ANARI_RENDERER:
    m_db.renderer.erase(index);
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
  m_updateDelegate.signalRemoveAllObjects();

  m_defaultObjects.material.reset();
  m_defaultObjects.camera.reset();

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
  m_db.renderer.clear();
}

RendererAppRef Scene::createRenderer(Token device, Token subtype)
{
  return createObjectImpl(m_db.renderer, device, subtype);
}

std::vector<RendererAppRef> Scene::createStandardRenderers(
    Token deviceName, anari::Device d)
{
  if (!d)
    return {};

  auto subtypes = tsd::scene::getANARIObjectSubtypes(d, ANARI_RENDERER);
  std::vector<RendererAppRef> retval;
  retval.reserve(subtypes.size());

  for (auto &subtype : subtypes) {
    auto r = createObjectImpl(m_db.renderer, deviceName, subtype);
    tsd::scene::parseANARIObjectInfo(*r, d, ANARI_RENDERER, subtype.c_str());
    retval.push_back(r);
  }

  return retval;
}

std::vector<RendererAppRef> Scene::renderersOfDevice(Token deviceName) const
{
  std::vector<RendererAppRef> renderers;
  renderers.reserve(5);
  foreach_item_const(m_db.renderer, [&](auto *r) {
    if (r && r->rendererDeviceName() == deviceName)
      renderers.push_back(getObject<Renderer>(r->index()));
  });
  return renderers;
}

void Scene::removeRenderersForDevice(Token deviceName)
{
  auto renderers = renderersOfDevice(deviceName);
  for (auto &r : renderers)
    removeObject(r.get());
}

MultiUpdateDelegate &Scene::updateDelegate()
{
  return m_updateDelegate;
}

const MultiUpdateDelegate &Scene::updateDelegate() const
{
  return m_updateDelegate;
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
    ls.ptr.reset(new Layer(this, name.str()));
    m_updateDelegate.signalLayerAdded(ls.ptr.get());
    m_numActiveLayers++;
  }
  return ls.ptr.get();
}

Token Scene::getLayerName(const Layer *layer) const
{
  for (size_t i = 0; i < m_layers.size(); i++) {
    if (m_layers.at_index(i).second.ptr.get() == layer)
      return m_layers.at_index(i).first;
  }
  return {};
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
  activeLayers.reserve(m_layers.size());
  for (const auto &ls : m_layers) {
    if (ls.second.active)
      activeLayers.push_back(ls.second.ptr.get());
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
  m_updateDelegate.signalLayerRemoved(m_layers[name].ptr.get());
  if (m_layers[name].active)
    m_numActiveLayers--;
  m_layers.erase(name);
}

void Scene::removeLayer(const Layer *layer)
{
  for (size_t i = 0; i < m_layers.size(); i++) {
    if (m_layers.at_index(i).second.ptr.get() == layer) {
      m_updateDelegate.signalLayerRemoved(m_layers.at_index(i).second.ptr.get());
      m_layers.erase(i);
      return;
    }
  }
}

void Scene::removeAllLayers()
{
  for (auto itr = m_layers.begin(); itr != m_layers.end(); itr++) {
    m_updateDelegate.signalLayerRemoved(itr->second.ptr.get());
  }

  m_layers.clear();
  m_numActiveLayers = 0;
}

LayerNodeRef Scene::insertChildNode(LayerNodeRef parent, const char *name)
{
  return parent->insert_last_child({(*parent)->layer(), name});
}

LayerNodeRef Scene::insertChildTransformNode(
    LayerNodeRef parent, mat4 xfm, const char *name)
{
  auto *layer = (*parent)->layer();
  auto inst = parent->insert_last_child({layer, xfm, name});
  signalLayerStructureChanged(layer);
  return inst;
}

LayerNodeRef Scene::insertChildTransformArrayNode(
    LayerNodeRef parent, Array *a, const char *name)
{
  auto *layer = (*parent)->layer();
  auto inst = parent->insert_last_child({layer, a, name});
  signalLayerStructureChanged(layer);
  return inst;
}

LayerNodeRef Scene::insertChildObjectNode(
    LayerNodeRef parent, anari::DataType type, size_t idx, const char *name)
{
  auto *layer = (*parent)->layer();
  auto *obj = getObject(type, idx);
  if (!obj) {
    logError(
        "[Scene::insertChildObjectNode()] failed to find object of type %s and"
        " index [%zu]",
        anari::toString(type),
        idx);
    return {};
  }
  auto inst = parent->insert_last_child({layer, obj, name});
  signalLayerStructureChanged(layer);
  return inst;
}

void Scene::removeNode(LayerNodeRef obj, bool deleteReferencedObjects)
{
  if (obj->isRoot())
    return;

  auto *layer = (*obj)->layer();

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
  signalLayerStructureChanged(layer);
}

void Scene::beginLayerEditBatch()
{
  m_inLayerBatch = true;
}

void Scene::endLayerEditBatch()
{
  m_inLayerBatch = false;

  auto &bl = m_batchedLayerUpdates;
  if (bl.empty())
    return;

  // Remove duplicates
  std::sort(bl.begin(), bl.end());
  bl.erase(std::unique(bl.begin(), bl.end()), bl.end());

  for (auto *l : bl)
    signalLayerStructureChanged(l);

  bl.clear();
}

void Scene::signalLayerStructureChanged(const Layer *l)
{
  if (m_inLayerBatch)
    m_batchedLayerUpdates.push_back(l);
  else
    m_updateDelegate.signalLayerStructureUpdated(l);
}

void Scene::signalLayerTransformChanged(const Layer *l)
{
  m_updateDelegate.signalLayerTransformUpdated(l);
}

void Scene::signalActiveLayersChanged()
{
  m_updateDelegate.signalActiveLayersChanged();
}

void Scene::signalObjectParameterUseCountZero(const Object *obj)
{
  m_updateDelegate.signalObjectParameterUseCountZero(obj);
}

void Scene::signalObjectLayerUseCountZero(const Object *obj)
{
  m_updateDelegate.signalObjectLayerUseCountZero(obj);
}

void Scene::removeUnusedObjects(bool includeRenderersAndCameras)
{
  tsd::core::logDebug("Removing unused context objects");

  // Always keep around the default material + default camera //

  ObjectUsePtr<Material> defaultMat = defaultMaterial();
  ObjectUsePtr<Camera> defaultCam = defaultCamera();

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

  if (includeRenderersAndCameras) {
    removeUnused(m_db.renderer);
    removeUnused(m_db.camera);
  }
}

size_t Scene::addDefragCallback(DefragCallback cb)
{
  size_t token = m_nextDefragToken++;
  m_defragCallbacks.push_back({token, std::move(cb)});
  return token;
}

void Scene::removeDefragCallback(size_t token)
{
  auto it = std::remove_if(m_defragCallbacks.begin(),
      m_defragCallbacks.end(),
      [token](const auto &e) { return e.token == token; });
  m_defragCallbacks.erase(it, m_defragCallbacks.end());
}

void Scene::defragmentObjectStorage()
{
  FlatMap<anari::DataType, std::vector<core::ObjectPoolRemapping>> defrags;

  // Defragment object storage and stash whether something happened //

  bool defrag = false;

  defrag |= !(defrags[ANARI_ARRAY] = m_db.array.defragment()).empty();
  defrag |= !(defrags[ANARI_SURFACE] = m_db.surface.defragment()).empty();
  defrag |= !(defrags[ANARI_GEOMETRY] = m_db.geometry.defragment()).empty();
  defrag |= !(defrags[ANARI_MATERIAL] = m_db.material.defragment()).empty();
  defrag |= !(defrags[ANARI_SAMPLER] = m_db.sampler.defragment()).empty();
  defrag |= !(defrags[ANARI_VOLUME] = m_db.volume.defragment()).empty();
  defrag |= !(defrags[ANARI_SPATIAL_FIELD] = m_db.field.defragment()).empty();
  defrag |= !(defrags[ANARI_LIGHT] = m_db.light.defragment()).empty();
  defrag |= !(defrags[ANARI_CAMERA] = m_db.camera.defragment()).empty();
  defrag |= !(defrags[ANARI_RENDERER] = m_db.renderer.defragment()).empty();

  if (!defrag) {
    tsd::core::logStatus("No defragmentation needed");
    return;
  } else {
    tsd::core::logStatus("Defragmenting context arrays:");
    for (const auto &pair : defrags) {
      if (!pair.second.empty())
        tsd::core::logStatus("    --> %s", anari::toString(pair.first));
    }
  }

  // Build index remapper //

  IndexRemapper getUpdatedIndex = [&](anari::DataType objType,
                                      size_t idx) -> size_t {
    if (anari::isArray(objType))
      objType = ANARI_ARRAY;
    auto &remaps = defrags[objType];
    auto it = std::find_if(
        remaps.begin(), remaps.end(), [&](auto &r) { return r.first == idx; });
    if (it != remaps.end())
      return it->second;
    else
      return idx; // index was not remapped, so return original
  };

  // Invoke all registered defrag callbacks //

  for (auto &entry : m_defragCallbacks)
    entry.callback(getUpdatedIndex);

  // Update all self-held index values to the new actual index //

  auto updateObjectHeldIndex = [&](auto &array) {
    foreach_item_ref(array, [&](auto ref) {
      if (!ref)
        return;
      ref->m_index = ref.index();
    });
  };

  updateObjectHeldIndex(m_db.array);
  updateObjectHeldIndex(m_db.surface);
  updateObjectHeldIndex(m_db.geometry);
  updateObjectHeldIndex(m_db.material);
  updateObjectHeldIndex(m_db.sampler);
  updateObjectHeldIndex(m_db.volume);
  updateObjectHeldIndex(m_db.field);
  updateObjectHeldIndex(m_db.light);
  updateObjectHeldIndex(m_db.camera);
  updateObjectHeldIndex(m_db.renderer);

  // Signal updates to any delegates //
  m_updateDelegate.signalInvalidateCachedObjects();
}

void Scene::cleanupScene()
{
  removeUnusedObjects();
  defragmentObjectStorage();
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

  retval->setUpdateDelegate(&m_updateDelegate);
  m_updateDelegate.signalObjectAdded(retval.data());

  return retval;
}

} // namespace tsd::scene

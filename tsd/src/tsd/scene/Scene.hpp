// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/scene/DefragCallback.hpp"
#include "tsd/scene/Layer.hpp"
#include "tsd/scene/objects/Array.hpp"
#include "tsd/scene/objects/Camera.hpp"
#include "tsd/scene/objects/Geometry.hpp"
#include "tsd/scene/objects/Light.hpp"
#include "tsd/scene/objects/Material.hpp"
#include "tsd/scene/objects/Renderer.hpp"
#include "tsd/scene/objects/Sampler.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scene/objects/Surface.hpp"
#include "tsd/scene/objects/Volume.hpp"
// std
#include <memory>
#include <type_traits>
#include <utility>

namespace tsd::scene {
struct Scene;
} // namespace tsd::scene

namespace tsd::animation {
struct AnimationManager;
} // namespace tsd::animation

namespace tsd::io {
// clang-format off
void save_Scene(scene::Scene &, core::DataNode &, bool, animation::AnimationManager *);
void load_Scene(scene::Scene &, core::DataNode &, animation::AnimationManager *);
// clang-format on
} // namespace tsd::io

namespace tsd::scene {

struct BaseUpdateDelegate;

/*
 * Flat collection of ObjectPool instances, one per ANARI object type; serves
 * as the canonical storage for all scene objects indexed by their pool slot.
 *
 * Example:
 *   ObjectDatabase &db = scene.objectDB();
 *   auto ref = db.geometry.at(idx);
 */
struct ObjectDatabase
{
  ObjectPool<Array> array;
  ObjectPool<Surface> surface;
  ObjectPool<Geometry> geometry;
  ObjectPool<Material> material;
  ObjectPool<Sampler> sampler;
  ObjectPool<Volume> volume;
  ObjectPool<SpatialField> field;
  ObjectPool<Light> light;
  ObjectPool<Camera> camera;
  ObjectPool<Renderer> renderer;

  ObjectDatabase() = default;
  TSD_NOT_MOVEABLE(ObjectDatabase)
  TSD_NOT_COPYABLE(ObjectDatabase)
};

std::string objectDBInfo(const ObjectDatabase &db);

///////////////////////////////////////////////////////////////////////////////
// Main TSD Scene /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// clang-format off
using LayerPtr = std::shared_ptr<Layer>;
struct LayerState { LayerPtr ptr; bool active{true}; };
using LayerMap = FlatMap<Token, LayerState>;
// clang-format on

/*
 * Central scene container that owns all ANARI-typed objects, a named layer
 * hierarchy, animations, and an optional update delegate; the main entry point
 * for creating, retrieving, and removing scene content.
 *
 * Example:
 *   Scene scene;
 *   auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);
 *   auto surf = scene.createSurface("s", geom, mat);
 *   scene.insertChildObjectNode(scene.defaultLayer()->root(), surf);
 */
struct Scene
{
  Scene();
  ~Scene();

  TSD_NOT_MOVEABLE(Scene)
  TSD_NOT_COPYABLE(Scene)

  MaterialRef defaultMaterial();
  CameraRef defaultCamera();
  Layer *defaultLayer();

  int mpiRank() const;
  int mpiNumRanks() const;
  void setMpiRankInfo(int rank, int numRanks);

  /////////////////////////////
  // Flat object collections //
  /////////////////////////////

  // Generic objets //

  template <typename T>
  ObjectPoolRef<T> createObject(Token subtype);
  Object *createObject(anari::DataType type, Token subtype);
  ArrayRef createArray(anari::DataType type,
      size_t items0,
      size_t items1 = 0,
      size_t items2 = 0);
  ArrayRef createArrayCUDA(anari::DataType type,
      size_t items0,
      size_t items1 = 0,
      size_t items2 = 0);
  ArrayRef createArrayProxy(anari::DataType type,
      size_t items0,
      size_t items1 = 0,
      size_t items2 = 0);
  SurfaceRef createSurface(
      const char *name = "", GeometryRef g = {}, MaterialRef m = {});

  template <typename T>
  ObjectPoolRef<T> getObject(size_t i) const;
  Object *getObject(const Any &a) const;
  Object *getObject(anari::DataType type, size_t i) const;
  size_t numberOfObjects(anari::DataType type) const;

  void removeObject(const Object *o);
  void removeObject(const Any &o);
  void removeAllObjects();

  // Renderers (specially handled per-device) //

  RendererAppRef createRenderer(
      Token deviceName, Token subtype = tokens::defaultToken);
  std::vector<RendererAppRef> createStandardRenderers(
      Token deviceName, anari::Device device);
  std::vector<RendererAppRef> renderersOfDevice(Token deviceName) const;
  void removeRenderersForDevice(Token deviceName);

  BaseUpdateDelegate *updateDelegate() const;
  void setUpdateDelegate(BaseUpdateDelegate *ud);

  const ObjectDatabase &objectDB() const;

  ///////////////////////////////////////////////////////
  // Instanced objects (surfaces, volumes, and lights) //
  ///////////////////////////////////////////////////////

  // Layers //

  const LayerMap &layers() const;
  size_t numberOfLayers() const;
  Layer *layer(Token name) const;
  Layer *layer(size_t i) const;

  Layer *addLayer(Token name);

  Token getLayerName(const Layer *layer) const;

  bool layerIsActive(Token name) const;
  void setLayerActive(const Layer *layer, bool active);
  void setLayerActive(Token name, bool active);
  void setOnlyLayerActive(Token name);
  void setAllLayersActive();
  size_t numberOfActiveLayers() const;
  std::vector<const Layer *> getActiveLayers() const;

  void removeLayer(Token name);
  void removeLayer(const Layer *layer);
  void removeAllLayers();

  // Insert nodes //

  LayerNodeRef insertChildNode(LayerNodeRef parent, const char *name = "");
  LayerNodeRef insertChildTransformNode(LayerNodeRef parent,
      mat4 xfm = math::IDENTITY_MAT4,
      const char *name = "");
  LayerNodeRef insertChildTransformArrayNode(
      LayerNodeRef parent, Array *a, const char *name = "");
  template <typename T>
  LayerNodeRef insertChildObjectNode(
      LayerNodeRef parent, ObjectPoolRef<T> obj, const char *name = "");
  LayerNodeRef insertChildObjectNode(LayerNodeRef parent,
      anari::DataType type,
      size_t idx,
      const char *name = "");

  // NOTE: convenience to create an object _and_ insert it into the tree
  template <typename T>
  using AddedObject = std::pair<LayerNodeRef, ObjectPoolRef<T>>;
  template <typename T>
  AddedObject<T> insertNewChildObjectNode(
      LayerNodeRef parent, Token subtype, const char *name = "");

  // Remove nodes //

  void removeNode(LayerNodeRef obj, bool deleteReferencedObjects = false);

  // Indicate changes occurred //

  void beginLayerEditBatch(); // structural layer changes are batched
  void endLayerEditBatch(); // stop batching + flush all layer update signals

  void signalLayerStructureChanged(const Layer *l);
  void signalLayerTransformChanged(const Layer *l);
  void signalActiveLayersChanged();
  void signalObjectParameterUseCountZero(const Object *obj);
  void signalObjectLayerUseCountZero(const Object *obj);

  ////////////////////////
  // Cleanup operations //
  ////////////////////////

  void removeUnusedObjects(bool includeRenderersAndCameras = false);
  void defragmentObjectStorage();
  void cleanupScene(); // remove unused + defragment

  // Defragmentation callbacks //

  size_t addDefragCallback(DefragCallback cb);
  void removeDefragCallback(size_t token);

 private:
  friend void ::tsd::io::save_Scene(
      Scene &, core::DataNode &, bool, tsd::animation::AnimationManager *);
  friend void ::tsd::io::load_Scene(
      Scene &, core::DataNode &, tsd::animation::AnimationManager *);

  template <typename OBJ_T, typename... Args>
  ObjectPoolRef<OBJ_T> createObjectImpl(ObjectPool<OBJ_T> &iv, Args &&...args);

  ArrayRef createArrayImpl(anari::DataType type,
      size_t items0,
      size_t items1,
      size_t items2,
      Array::MemoryKind kind);

  ObjectDatabase m_db;

  struct DefaultObjects
  {
    ObjectUsePtr<Material, Object::UseKind::INTERNAL> material;
    ObjectUsePtr<Camera, Object::UseKind::INTERNAL> camera;
  } m_defaultObjects;

  BaseUpdateDelegate *m_updateDelegate{nullptr};

  LayerMap m_layers;
  size_t m_numActiveLayers{0};
  bool m_inLayerBatch{false};
  std::vector<const Layer *> m_batchedLayerUpdates;

  struct DefragCallbackEntry
  {
    size_t token;
    DefragCallback callback;
  };
  std::vector<DefragCallbackEntry> m_defragCallbacks;
  size_t m_nextDefragToken{0};

  struct MpiData
  {
    int rank{0};
    int numRanks{1};
  } m_mpi;
};

///////////////////////////////////////////////////////////////////////////////
// Inlined definitions ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Scene //

template <typename T>
inline ObjectPoolRef<T> Scene::createObject(Token subtype)
{
  static_assert(std::is_base_of<Object, T>::value,
      "Scene::createObject<> can only create tsd::Object subclasses");
  static_assert(!std::is_same<T, Array>::value,
      "Use Scene::createArray() to create tsd::Array objects");
  static_assert(!std::is_same<T, Renderer>::value,
      "Use Scene::createRenderer() to create tsd::Renderer objects");
  return {};
}

template <>
inline GeometryRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.geometry, subtype);
}

template <>
inline MaterialRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.material, subtype);
}

template <>
inline SamplerRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.sampler, subtype);
}

template <>
inline VolumeRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.volume, subtype);
}

template <>
inline SpatialFieldRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.field, subtype);
}

template <>
inline LightRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.light, subtype);
}

template <>
inline CameraRef Scene::createObject(Token subtype)
{
  return createObjectImpl(m_db.camera, subtype);
}

template <typename T>
inline ObjectPoolRef<T> Scene::getObject(size_t i) const
{
  static_assert(std::is_base_of<Object, T>::value,
      "Scene::getObject<> can only get tsd::Object subclasses");
  return {};
}

template <>
inline SurfaceRef Scene::getObject(size_t i) const
{
  return m_db.surface.at(i);
}

template <>
inline ArrayRef Scene::getObject(size_t i) const
{
  return m_db.array.at(i);
}

template <>
inline GeometryRef Scene::getObject(size_t i) const
{
  return m_db.geometry.at(i);
}

template <>
inline MaterialRef Scene::getObject(size_t i) const
{
  return m_db.material.at(i);
}

template <>
inline SamplerRef Scene::getObject(size_t i) const
{
  return m_db.sampler.at(i);
}

template <>
inline VolumeRef Scene::getObject(size_t i) const
{
  return m_db.volume.at(i);
}

template <>
inline SpatialFieldRef Scene::getObject(size_t i) const
{
  return m_db.field.at(i);
}

template <>
inline LightRef Scene::getObject(size_t i) const
{
  return m_db.light.at(i);
}

template <>
inline CameraRef Scene::getObject(size_t i) const
{
  return m_db.camera.at(i);
}

template <>
inline RendererRef Scene::getObject(size_t i) const
{
  return m_db.renderer.at(i);
}

template <typename OBJ_T, typename... Args>
inline ObjectPoolRef<OBJ_T> Scene::createObjectImpl(
    ObjectPool<OBJ_T> &iv, Args &&...args)
{
  auto retval = iv.emplace(std::forward<Args>(args)...);
  retval->m_scene = this;
  retval->m_index = retval.index();
  if (m_updateDelegate) {
    retval->setUpdateDelegate(m_updateDelegate);
    m_updateDelegate->signalObjectAdded(retval.data());
  }
  return retval;
}

template <typename T>
inline LayerNodeRef Scene::insertChildObjectNode(
    LayerNodeRef parent, ObjectPoolRef<T> obj, const char *name)
{
  return insertChildObjectNode(parent, obj->type(), obj->index(), name);
}

template <typename T>
inline Scene::AddedObject<T> Scene::insertNewChildObjectNode(
    LayerNodeRef parent, Token subtype, const char *name)
{
  auto obj = createObject<T>(subtype);
  auto inst = insertChildObjectNode(parent, obj, name);
  return std::make_pair(inst, obj);
}

// Object definitions /////////////////////////////////////////////////////////

template <typename T>
inline T *Object::parameterValueAsObject(Token name) const
{
  static_assert(isObject<T>(),
      "Object::parameterValueAsObject() can only retrieve object values");

  auto *p = parameter(name);
  auto *s = scene();
  if (!p || !s || !p->value().holdsObject())
    return nullptr;
  return (T *)s->getObject(p->value());
}

} // namespace tsd::scene

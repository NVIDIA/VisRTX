// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Animation.hpp"
#include "tsd/core/scene/Layer.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include "tsd/core/scene/objects/Camera.hpp"
#include "tsd/core/scene/objects/Geometry.hpp"
#include "tsd/core/scene/objects/Light.hpp"
#include "tsd/core/scene/objects/Material.hpp"
#include "tsd/core/scene/objects/Sampler.hpp"
#include "tsd/core/scene/objects/SpatialField.hpp"
#include "tsd/core/scene/objects/Surface.hpp"
#include "tsd/core/scene/objects/Volume.hpp"
// std
#include <memory>
#include <type_traits>
#include <utility>

namespace tsd::core {
struct Scene;
} // namespace tsd::core

namespace tsd::io {
void save_Scene(core::Scene &, core::DataNode &, bool);
void load_Scene(core::Scene &, core::DataNode &);
} // namespace tsd::io

namespace tsd::core {

struct BaseUpdateDelegate;

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

  // Not copyable or moveable //
  ObjectDatabase() = default;
  ObjectDatabase(const ObjectDatabase &) = delete;
  ObjectDatabase(ObjectDatabase &&) = delete;
  ObjectDatabase &operator=(const ObjectDatabase &) = delete;
  ObjectDatabase &operator=(ObjectDatabase &&) = delete;
  //////////////////////////////
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

struct Scene
{
  Scene();
  ~Scene();

  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;
  Scene(Scene &&) = delete;
  Scene &operator=(Scene &&) = delete;

  MaterialRef defaultMaterial() const;
  Layer *defaultLayer();

  int mpiRank() const;
  int mpiNumRanks() const;
  void setMpiRankInfo(int rank, int numRanks);

  /////////////////////////////
  // Flat object collections //
  /////////////////////////////

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
      mat4 xfm = mat4(tsd::math::identity),
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

  void removeNode(
      LayerNodeRef obj, bool deleteReferencedObjects = false);

  // Indicate changes occurred //

  void signalLayerChange(const Layer *l);
  void signalActiveLayersChanged();
  void signalObjectParameterUseCountZero(const Object *obj);
  void signalObjectLayerUseCountZero(const Object *obj);

  ////////////////
  // Animations //
  ////////////////

  Animation *addAnimation(const char *name = "");
  size_t numberOfAnimations() const;
  Animation *animation(size_t i) const;
  void removeAnimation(Animation *a);
  void removeAllAnimations();

  void setAnimationTime(float time /* 0.f - 1.f */);
  float getAnimationTime() const;

  void setAnimationIncrement(float increment);
  float getAnimationIncrement() const;
  void incrementAnimationTime();

  ////////////////////////
  // Cleanup operations //
  ////////////////////////

  void removeUnusedObjects();
  void defragmentObjectStorage();
  void cleanupScene(); // remove unused + defragment

 private:
  friend void ::tsd::io::save_Scene(Scene &, core::DataNode &, bool);
  friend void ::tsd::io::load_Scene(Scene &, core::DataNode &);

  template <typename OBJ_T, typename... Args>
  ObjectPoolRef<OBJ_T> createObjectImpl(ObjectPool<OBJ_T> &iv, Args &&...args);

  ArrayRef createArrayImpl(anari::DataType type,
      size_t items0,
      size_t items1,
      size_t items2,
      Array::MemoryKind kind);

  ObjectDatabase m_db;
  BaseUpdateDelegate *m_updateDelegate{nullptr};
  LayerMap m_layers;
  size_t m_numActiveLayers{0};
  struct AnimationData
  {
    float incrementSize{0.01f};
    float time{0.f};
    std::vector<std::unique_ptr<Animation>> objects;
  } m_animations;
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

} // namespace tsd::core

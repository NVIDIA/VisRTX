// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/containers/Forest.hpp"
#include "tsd/containers/IndexedVector.hpp"
#include "tsd/objects/Array.hpp"
#include "tsd/objects/Geometry.hpp"
#include "tsd/objects/Light.hpp"
#include "tsd/objects/Material.hpp"
#include "tsd/objects/Sampler.hpp"
#include "tsd/objects/SpatialField.hpp"
#include "tsd/objects/Surface.hpp"
#include "tsd/objects/Volume.hpp"
// std
#include <type_traits>
#include <utility>

namespace tsd {

struct BaseUpdateDelegate;
struct Context;

struct InstanceTreeData
{
  InstanceTreeData() = default;
  template <typename T>
  InstanceTreeData(T v, const char *n = "");
  InstanceTreeData(utility::Any v, const char *n);

  bool hasDefault() const;
  bool isObject() const;
  bool isTransform() const;
  bool isEmpty() const;

  Object *getObject(Context *ctx) const;
  math::mat4 getTransform() const;

  // Data //

  utility::Any value;
  utility::Any defaultValue;
  std::string name;
  bool enabled{true};
  FlatMap<std::string, utility::Any> customParameters;
  FlatMap<std::string, utility::Any> valueCache;
};
using InstanceTree = utility::Forest<InstanceTreeData>;
using InstanceNode = utility::ForestNode<InstanceTreeData>;
using InstanceVisitor = utility::ForestVisitor<InstanceTreeData>;

struct ObjectDatabase
{
  IndexedVector<Array> array;
  IndexedVector<Surface> surface;
  IndexedVector<Geometry> geometry;
  IndexedVector<Material> material;
  IndexedVector<Sampler> sampler;
  IndexedVector<Volume> volume;
  IndexedVector<SpatialField> field;
  IndexedVector<Light> light;

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
// Main TSD Context ///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct Context
{
  Context();

  MaterialRef defaultMaterial() const;

  /////////////////////////////
  // Flat object collections //
  /////////////////////////////

  template <typename T>
  IndexedVectorRef<T> createObject();
  template <typename T>
  IndexedVectorRef<T> createObject(Token subtype);
  ArrayRef createArray(anari::DataType type,
      size_t items0,
      size_t items1 = 0,
      size_t items2 = 0);
  ArrayRef createArrayCUDA(anari::DataType type,
      size_t items0,
      size_t items1 = 0,
      size_t items2 = 0);
  SurfaceRef createSurface(const char *name, GeometryRef g, MaterialRef m = {});

  template <typename T>
  IndexedVectorRef<T> getObject(size_t i) const;
  Object *getObject(const utility::Any &a) const;
  Object *getObject(anari::DataType type, size_t i) const;
  size_t numberOfObjects(anari::DataType type) const;

  void removeObject(const Object &o);
  void removeObject(const utility::Any &o);
  void removeAllObjects();

  BaseUpdateDelegate *updateDelegate() const;
  void setUpdateDelegate(BaseUpdateDelegate *ud);

  const ObjectDatabase &objectDB() const;

  ///////////////////////////////////////////////////////
  // Instanced objects (surfaces, volumes, and lights) //
  ///////////////////////////////////////////////////////

  // Insert nodes //

  InstanceNode::Ref insertChildNode(
      InstanceNode::Ref parent, const char *name = "");
  InstanceNode::Ref insertChildTransformNode(InstanceNode::Ref parent,
      mat4 xfm = tsd::mat4(tsd::math::identity),
      const char *name = "");
  template <typename T>
  InstanceNode::Ref insertChildObjectNode(
      InstanceNode::Ref parent, IndexedVectorRef<T> obj, const char *name = "");

  // NOTE: convenience to create an object _and_ insert it into the tree
  template <typename T>
  using AddedObject = std::pair<InstanceNode::Ref, IndexedVectorRef<T>>;
  template <typename T>
  AddedObject<T> insertNewChildObjectNode(
      InstanceNode::Ref parent, Token subtype, const char *name = "");

  // Remove nodes //

  void removeInstancedObject(InstanceNode::Ref obj);

  // Indicate changes occurred //

  void signalInstanceTreeChange();

  InstanceTree tree{
      {tsd::mat4(tsd::math::identity), "root"}}; // root must be a matrix

 private:
  friend void save_Context(Context &ctx, const char *filename);
  friend void import_Context(Context &ctx, const char *filename);

  template <typename OBJ_T>
  IndexedVectorRef<OBJ_T> createObjectImpl(
      IndexedVector<OBJ_T> &iv, Token subtype);
  template <typename OBJ_T>
  IndexedVectorRef<OBJ_T> createObjectImpl(IndexedVector<OBJ_T> &iv);

  ArrayRef createArrayImpl(anari::DataType type,
      size_t items0,
      size_t items1,
      size_t items2,
      Array::MemoryKind kind);

  ObjectDatabase m_db;
  BaseUpdateDelegate *m_updateDelegate{nullptr};
};

///////////////////////////////////////////////////////////////////////////////
// Inlined definitions ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// InstanceTreeData //

template <typename T>
inline InstanceTreeData::InstanceTreeData(T v, const char *n)
    : InstanceTreeData(utility::Any(v), n)
{}

inline InstanceTreeData::InstanceTreeData(utility::Any v, const char *n)
    : value(v), name(n)
{
  if (!isObject())
    defaultValue = v;
}

inline bool InstanceTreeData::hasDefault() const
{
  return defaultValue;
}

inline bool InstanceTreeData::isObject() const
{
  return anari::isObject(value.type());
}

inline bool InstanceTreeData::isTransform() const
{
  return value.type() == ANARI_FLOAT32_MAT4;
}

inline bool InstanceTreeData::isEmpty() const
{
  return value;
}

inline Object *InstanceTreeData::getObject(Context *ctx) const
{
  return ctx->getObject(value);
}

inline math::mat4 InstanceTreeData::getTransform() const
{
  return isTransform() ? value.getAs<math::mat4>() : math::IDENTITY_MAT4;
}

// Context //

template <typename T>
inline IndexedVectorRef<T> Context::createObject()
{
  static_assert(std::is_base_of<Object, T>::value,
      "Context::createObject<> can only create tsd::Object subclasses");
  static_assert(!std::is_same<T, Array>::value,
      "Use Context::createArray() to create tsd::Array objects");
  return {};
}

template <>
inline SurfaceRef Context::createObject()
{
  return createObjectImpl(m_db.surface);
}

template <typename T>
inline IndexedVectorRef<T> Context::createObject(Token subtype)
{
  static_assert(std::is_base_of<Object, T>::value,
      "Context::createObject<> can only create tsd::Object subclasses");
  static_assert(!std::is_same<T, Array>::value,
      "Use Context::createArray() to create tsd::Array objects");
  return {};
}

template <>
inline GeometryRef Context::createObject(Token subtype)
{
  return createObjectImpl(m_db.geometry, subtype);
}

template <>
inline MaterialRef Context::createObject(Token subtype)
{
  return createObjectImpl(m_db.material, subtype);
}

template <>
inline SamplerRef Context::createObject(Token subtype)
{
  return createObjectImpl(m_db.sampler, subtype);
}

template <>
inline VolumeRef Context::createObject(Token subtype)
{
  return createObjectImpl(m_db.volume, subtype);
}

template <>
inline SpatialFieldRef Context::createObject(Token subtype)
{
  return createObjectImpl(m_db.field, subtype);
}

template <>
inline LightRef Context::createObject(Token subtype)
{
  return createObjectImpl(m_db.light, subtype);
}

template <typename T>
inline IndexedVectorRef<T> Context::getObject(size_t i) const
{
  static_assert(std::is_base_of<Object, T>::value,
      "Context::getObject<> can only get tsd::Object subclasses");
  return {};
}

template <>
inline ArrayRef Context::getObject(size_t i) const
{
  return m_db.array.at(i);
}

template <>
inline GeometryRef Context::getObject(size_t i) const
{
  return m_db.geometry.at(i);
}

template <>
inline MaterialRef Context::getObject(size_t i) const
{
  return m_db.material.at(i);
}

template <>
inline SamplerRef Context::getObject(size_t i) const
{
  return m_db.sampler.at(i);
}

template <>
inline VolumeRef Context::getObject(size_t i) const
{
  return m_db.volume.at(i);
}

template <>
inline SpatialFieldRef Context::getObject(size_t i) const
{
  return m_db.field.at(i);
}

template <>
inline LightRef Context::getObject(size_t i) const
{
  return m_db.light.at(i);
}

template <typename OBJ_T>
inline IndexedVectorRef<OBJ_T> Context::createObjectImpl(
    IndexedVector<OBJ_T> &iv, Token subtype)
{
  auto retval = iv.emplace(subtype);
  retval->m_context = this;
  retval->m_index = retval.index();
  if (m_updateDelegate) {
    retval->setUpdateDelegate(m_updateDelegate);
    m_updateDelegate->signalObjectAdded(retval.data());
  }
  return retval;
}

template <typename OBJ_T>
inline IndexedVectorRef<OBJ_T> Context::createObjectImpl(
    IndexedVector<OBJ_T> &iv)
{
  auto retval = iv.emplace();
  retval->m_context = this;
  retval->m_index = retval.index();
  if (m_updateDelegate) {
    retval->setUpdateDelegate(m_updateDelegate);
    m_updateDelegate->signalObjectAdded(retval.data());
  }
  return retval;
}

template <typename T>
inline InstanceNode::Ref Context::insertChildObjectNode(
    InstanceNode::Ref parent, IndexedVectorRef<T> obj, const char *name)
{
  auto inst = tree.insert_last_child(
      parent, tsd::utility::Any{obj->type(), obj->index()});
  (*inst)->name = name;
  signalInstanceTreeChange();
  return inst;
}

template <typename T>
inline Context::AddedObject<T> Context::insertNewChildObjectNode(
    InstanceNode::Ref parent, Token subtype, const char *name)
{
  auto obj = createObject<T>(subtype);
  auto inst = tree.insert_last_child(
      parent, tsd::utility::Any{obj->type(), obj->index()});
  (*inst)->name = name;
  signalInstanceTreeChange();
  return std::make_pair(inst, obj);
}

// Object definitions /////////////////////////////////////////////////////////

template <typename T>
inline T *Object::parameterValueAsObject(Token name)
{
  static_assert(isObject<T>(),
      "Object::parameterValueAsObject() can only retrieve object values");

  auto *p = parameter(name);
  auto *ctx = context();
  if (!p || !ctx || !p->value().holdsObject())
    return nullptr;
  return (T *)ctx->getObject(p->value());
}

} // namespace tsd

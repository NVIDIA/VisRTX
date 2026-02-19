// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/DataTree.hpp"
#include "tsd/core/FlatMap.hpp"
#include "tsd/core/ObjectPool.hpp"
#include "tsd/core/Parameter.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/core/scene/UpdateDelegate.hpp"
// std
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>

namespace tsd::core {
struct Object;
} // namespace tsd::core

namespace tsd::io {
void nodeToNewObject(core::DataNode &node, core::Object &obj);
} // namespace tsd::io

namespace tsd::core {

struct Scene;
struct AnariObjectCache;

// Token declarations /////////////////////////////////////////////////////////

namespace tokens {

extern Token none;
extern Token unknown;
extern Token defaultToken;

} // namespace tokens

// Helper macros //////////////////////////////////////////////////////////////

#define DECLARE_OBJECT_DEFAULT_LIFETIME(TYPE_NAME)                             \
  TYPE_NAME(const TYPE_NAME &) = delete;                                       \
  TYPE_NAME &operator=(const TYPE_NAME &) = delete;                            \
  TYPE_NAME(TYPE_NAME &&) = default;                                           \
  TYPE_NAME &operator=(TYPE_NAME &&) = default;

// Type declarations //////////////////////////////////////////////////////////

struct Object : public ParameterObserver
{
  using ParameterMap = FlatMap<Token, Parameter>;
  // clang-format off
  enum class UseKind { APP, PARAMETER, LAYER };
  // clang-format on

  Object(anari::DataType type = ANARI_UNKNOWN, Token subtype = tokens::none);
  virtual ~Object();

  // Movable, not copyable
  Object(const Object &) = delete;
  Object &operator=(const Object &) = delete;
  Object(Object &&);
  Object &operator=(Object &&);

  virtual anari::DataType type() const;
  Token subtype() const;
  size_t index() const;
  Scene *scene() const;
  Token rendererDeviceName() const; // only populated by Renderer

  //// Use count tracking (Scene garbage collection) ////

  size_t totalUseCount() const;
  size_t useCount(UseKind kind) const;
  void incUseCount(UseKind kind = UseKind::APP);
  void decUseCount(UseKind kind = UseKind::APP);

  //// Metadata ////

  const std::string &name() const;
  std::string &editableName();
  void setName(const char *n);
  void setName(const std::string &n);

  Any getMetadataValue(const std::string &name) const;
  void getMetadataArray(const std::string &name,
      anari::DataType *type,
      const void **ptr,
      size_t *size) const;

  void setMetadataValue(const std::string &name, Any v);
  void setMetadataArray(const std::string &name,
      anari::DataType type,
      const void *v,
      size_t numElements);
  void removeMetadata(const std::string &name);

  size_t numMetadata() const;
  const char *getMetadataName(size_t i) const;

  //// Parameters ////

  // Token-based access
  Parameter &addParameter(Token name);
  template <typename T>
  Parameter *setParameter(Token name, T value);
  Parameter *setParameter(Token name, ANARIDataType type, const void *v);
  Parameter *setParameterObject(Token name, const Object &obj);

  const Parameter *parameter(Token name) const;
  Parameter *parameter(Token name);
  template <typename T>
  std::optional<T> parameterValueAs(Token name);
  template <typename T>
  const std::optional<T> parameterValueAs(Token name) const;
  template <typename T = Object>
  T *parameterValueAsObject(Token name) const;

  void removeParameter(Token name);
  void removeAllParameters();

  // Index-based access
  size_t numParameters() const;
  const Parameter &parameterAt(size_t i) const;
  Parameter &parameterAt(size_t i);
  const char *parameterNameAt(size_t i) const;

  //// ANARI Objects /////

  virtual anari::Object makeANARIObject(anari::Device d) const;

  void updateANARIParameter(anari::Device d,
      anari::Object o,
      const Parameter &p,
      const char *n,
      AnariObjectCache *cache = nullptr) const;
  void updateAllANARIParameters(anari::Device d,
      anari::Object o,
      AnariObjectCache *cache = nullptr) const;

  //// Updates ////

  void setUpdateDelegate(BaseUpdateDelegate *ud);

 protected:
  virtual void parameterChanged(const Parameter *p, const Any &oldVal) override;
  virtual void removeParameter(const Parameter *p) override;
  BaseUpdateDelegate *updateDelegate() const;

  Token m_rendererDeviceName{}; // only used by Renderer

 private:
  friend struct Scene;
  friend void io::nodeToNewObject(core::DataNode &node, Object &obj);

  void incObjectUseCountParameter(const Parameter *p);
  void decObjectUseCountParameter(const Parameter *p);

  void initMetadata() const;

  Scene *m_scene{nullptr};
  ParameterMap m_parameters;
  anari::DataType m_type{ANARI_UNKNOWN};
  Token m_subtype;
  std::string m_name;
  std::string m_description;
  size_t m_index{0};
  BaseUpdateDelegate *m_updateDelegate{nullptr};
  mutable std::unique_ptr<core::DataTree> m_metadata;
  struct UseCounts
  {
    size_t app{0};
    size_t parameter{0};
    size_t layer{0};
  } m_useCounts;
};

void print(const Object &obj, std::ostream &out = std::cout);

// Type trait-like helper functions //

template <typename T>
constexpr bool isObject()
{
  return std::is_same<Object, T>::value || std::is_base_of<Object, T>::value;
}

// ANARI object info parsing //////////////////////////////////////////////////

std::vector<std::string> getANARIObjectSubtypes(
    anari::Device d, anari::DataType type);

void parseANARIObjectInfo(
    Object &o, anari::Device d, ANARIDataType objectType, const char *subtype);

Object parseANARIObjectInfo(
    anari::Device d, ANARIDataType type, const char *subtype);

// Inlined definitions ////////////////////////////////////////////////////////

// Object //

template <typename T>
inline Parameter *Object::setParameter(Token name, T value)
{
  auto *p = m_parameters.at(name);
  if (p)
    p->setValue(value);
  else
    p = &(addParameter(name).setValue(value));
  return p;
}

template <typename T>
inline std::optional<T> Object::parameterValueAs(Token name)
{
  static_assert(!isObject<T>(),
      "Object::parameterValueAs() does not work on parameters holding objects");

  auto *p = parameter(name);
  if (!p || !p->value().is<T>())
    return {};
  return p->value().get<T>();
}

template <typename T>
inline const std::optional<T> Object::parameterValueAs(Token name) const
{
  static_assert(!isObject<T>(),
      "Object::parameterValueAs() does not work on parameters holding objects");

  auto *p = parameter(name);
  if (!p || !p->value().is<T>())
    return {};
  return p->value().get<T>();
}

} // namespace tsd::core

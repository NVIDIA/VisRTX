// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/containers/FlatMap.hpp"
#include "tsd/containers/IndexedVector.hpp"
#include "tsd/core/AnariObjectCache.hpp"
#include "tsd/core/Parameter.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/core/UpdateDelegate.hpp"
// std
#include <iostream>
#include <optional>
#include <type_traits>

namespace tsd {

using namespace literals;
struct Context;

// Token declarations /////////////////////////////////////////////////////////

namespace tokens {

extern Token none;
extern Token unknown;

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

  Object(anari::DataType type = ANARI_UNKNOWN, Token subtype = tokens::none);
  virtual ~Object() = default;

  // Movable, not copyable
  Object(const Object &) = delete;
  Object &operator=(const Object &) = delete;
  Object(Object &&);
  Object &operator=(Object &&);

  virtual anari::DataType type() const;
  Token subtype() const;
  size_t index() const;
  Context *context() const;

  //// Metadata ////

  const std::string &name() const;
  void setName(const char *n);

  //// Parameters ////

  // Token-based access
  Parameter &addParameter(Token name);
  template <typename T>
  void setParameter(Token name, T value);
  void setParameter(Token name, ANARIDataType type, const void *v);
  void setParameterObject(Token name, const Object &obj);

  Parameter *parameter(Token name);
  template <typename T>
  std::optional<T> parameterValueAs(Token name);
  template <typename T = Object>
  T *parameterValueAsObject(Token name);

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
  virtual void parameterChanged(const Parameter *p) override;
  BaseUpdateDelegate *updateDelegate() const;

 private:
  friend struct Context;

  Context *m_context{nullptr};
  ParameterMap m_parameters;
  anari::DataType m_type{ANARI_UNKNOWN};
  Token m_subtype;
  std::string m_name;
  std::string m_description;
  size_t m_index{0};
  BaseUpdateDelegate *m_updateDelegate{nullptr};
};

void print(const Object &obj, std::ostream &out = std::cout);

// Type trait-like helper functions //

template <typename T>
constexpr bool isObject()
{
  return std::is_same<Object, T>::value || std::is_base_of<Object, T>::value;
}

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline void Object::setParameter(Token name, T value)
{
  auto *p = m_parameters.at(name);
  if (p)
    p->setValue(value);
  else
    addParameter(name).setValue(value);
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

} // namespace tsd

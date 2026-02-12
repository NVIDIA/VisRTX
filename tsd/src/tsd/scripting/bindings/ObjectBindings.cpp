// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ArrayHelpers.hpp"
#include "ObjectMethodBindings.hpp"
#include "ParameterHelpers.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/core/scene/Scene.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include "tsd/core/scene/objects/Camera.hpp"
#include "tsd/core/scene/objects/Geometry.hpp"
#include "tsd/core/scene/objects/Light.hpp"
#include "tsd/core/scene/objects/Material.hpp"
#include "tsd/core/scene/objects/Sampler.hpp"
#include "tsd/core/scene/objects/SpatialField.hpp"
#include "tsd/core/scene/objects/Surface.hpp"
#include "tsd/core/scene/objects/Volume.hpp"
#include "tsd/scripting/LuaBindings.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <fmt/format.h>
#include <functional>
#include <sol/sol.hpp>
#include <vector>

namespace tsd::scripting {

template <typename T>
class ScopedArrayMap
{
 public:
  ScopedArrayMap(core::Array &arr) : m_array(arr), m_ptr(arr.mapAs<T>()) {}
  ~ScopedArrayMap()
  {
    if (m_ptr)
      m_array.unmap();
  }

  ScopedArrayMap(const ScopedArrayMap &) = delete;
  ScopedArrayMap &operator=(const ScopedArrayMap &) = delete;

  T *data()
  {
    return m_ptr;
  }
  T &operator[](size_t i)
  {
    return m_ptr[i];
  }

 private:
  core::Array &m_array;
  T *m_ptr;
};

ANARIDataType arrayTypeFromString(const std::string &typeStr)
{
  if (typeStr == "float")
    return ANARI_FLOAT32;
  if (typeStr == "float2")
    return ANARI_FLOAT32_VEC2;
  if (typeStr == "float3")
    return ANARI_FLOAT32_VEC3;
  if (typeStr == "float4")
    return ANARI_FLOAT32_VEC4;
  if (typeStr == "int")
    return ANARI_INT32;
  if (typeStr == "int2")
    return ANARI_INT32_VEC2;
  if (typeStr == "int3")
    return ANARI_INT32_VEC3;
  if (typeStr == "int4")
    return ANARI_INT32_VEC4;
  if (typeStr == "uint")
    return ANARI_UINT32;
  if (typeStr == "uint2")
    return ANARI_UINT32_VEC2;
  if (typeStr == "uint3")
    return ANARI_UINT32_VEC3;
  if (typeStr == "uint4")
    return ANARI_UINT32_VEC4;
  if (typeStr == "mat4")
    return ANARI_FLOAT32_MAT4;

  throw std::runtime_error(fmt::format(
      "Unknown array type: '{}'. Valid types: float, float2, float3, float4, "
      "int, int2, int3, int4, uint, uint2, uint3, uint4, mat4",
      typeStr));
}

static size_t elementArity(ANARIDataType elemType)
{
  switch (elemType) {
  case ANARI_FLOAT32_VEC2:
  case ANARI_INT32_VEC2:
  case ANARI_UINT32_VEC2:
    return 2;
  case ANARI_FLOAT32_VEC3:
  case ANARI_INT32_VEC3:
  case ANARI_UINT32_VEC3:
    return 3;
  case ANARI_FLOAT32_VEC4:
  case ANARI_INT32_VEC4:
  case ANARI_UINT32_VEC4:
    return 4;
  default:
    return 1;
  }
}

static bool isElementTable(sol::table t, ANARIDataType elemType)
{
  return t.size() == elementArity(elemType);
}

static bool isVecUserdata(sol::object o, ANARIDataType elemType)
{
  switch (elemType) {
  case ANARI_FLOAT32_VEC2:
    return o.is<math::float2>();
  case ANARI_FLOAT32_VEC3:
    return o.is<math::float3>();
  case ANARI_FLOAT32_VEC4:
    return o.is<math::float4>();
  case ANARI_INT32_VEC2:
    return o.is<math::int2>();
  case ANARI_INT32_VEC3:
    return o.is<math::int3>();
  case ANARI_INT32_VEC4:
    return o.is<math::int4>();
  case ANARI_UINT32_VEC2:
    return o.is<math::uint2>();
  case ANARI_UINT32_VEC3:
    return o.is<math::uint3>();
  case ANARI_UINT32_VEC4:
    return o.is<math::uint4>();
  default:
    return false;
  }
}

static bool isElementLike(sol::object o, ANARIDataType elemType)
{
  if (elemType == ANARI_FLOAT32_MAT4)
    return o.is<math::mat4>();
  const size_t arity = elementArity(elemType);
  if (arity >= 2) {
    return isVecUserdata(o, elemType)
        || (o.is<sol::table>() && isElementTable(o.as<sol::table>(), elemType));
  }
  // Scalar types
  return o.is<double>() || o.is<float>() || o.is<int>();
}

template <typename Vec, typename Elem, size_t N>
static Vec decodeVec(sol::object o, const char *typeName)
{
  if (o.is<Vec>())
    return o.as<Vec>();
  if (o.is<sol::table>()) {
    sol::table t = o.as<sol::table>();
    if (t.size() != N)
      throw std::runtime_error(
          fmt::format("expected {} or table of {} numbers", typeName, N));
    if constexpr (N == 2)
      return Vec(t[1].get<Elem>(), t[2].get<Elem>());
    else if constexpr (N == 3)
      return Vec(t[1].get<Elem>(), t[2].get<Elem>(), t[3].get<Elem>());
    else if constexpr (N == 4)
      return Vec(t[1].get<Elem>(),
          t[2].get<Elem>(),
          t[3].get<Elem>(),
          t[4].get<Elem>());
  }
  throw std::runtime_error(
      fmt::format("expected {} or table of {} numbers", typeName, N));
}

template <typename Vec, typename Elem, size_t N>
static void fillArrayFromFlattened(ScopedArrayMap<Vec> &map,
    size_t copyCount,
    std::vector<sol::object> &flattened,
    const char *typeName)
{
  for (size_t i = 0; i < copyCount; i++)
    map[i] = decodeVec<Vec, Elem, N>(flattened[i], typeName);
}

template <typename Vec, typename Elem, size_t N>
static void fillArrayFromTable(ScopedArrayMap<Vec> &map,
    size_t copyCount,
    sol::table &data,
    const char *typeName)
{
  for (size_t i = 0; i < copyCount; i++)
    map[i] = decodeVec<Vec, Elem, N>(data[i + 1], typeName);
}

template <typename Vec, size_t N>
static void getVecArrayAsLua(sol::state_view &lua,
    ScopedArrayMap<Vec> &map,
    size_t count,
    sol::table &out)
{
  for (size_t i = 0; i < count; i++) {
    const auto &v = map[i];
    auto t = lua.create_table(static_cast<int>(N), 0);
    if constexpr (N >= 1)
      t[1] = v.x;
    if constexpr (N >= 2)
      t[2] = v.y;
    if constexpr (N >= 3)
      t[3] = v.z;
    if constexpr (N >= 4)
      t[4] = v.w;
    out[i + 1] = t;
  }
}

static void inferArrayDimsFromLuaData(sol::table data,
    ANARIDataType elemType,
    size_t &items0,
    size_t &items1,
    size_t &items2)
{
  items0 = 0;
  items1 = 0;
  items2 = 0;

  const size_t n = data.size();
  if (n == 0)
    throw std::runtime_error(
        "setParameterArray: cannot infer dimensions from empty data table");

  // 1D candidate
  bool is1D = true;
  for (size_t i = 0; i < n; i++) {
    if (!isElementLike(data[i + 1], elemType)) {
      is1D = false;
      break;
    }
  }
  if (is1D) {
    items0 = n;
    return;
  }

  // 2D candidate: data[y][x]
  bool is2D = true;
  size_t width = 0;
  for (size_t y = 0; y < n && is2D; y++) {
    sol::object rowObj = data[y + 1];
    if (!rowObj.is<sol::table>()) {
      is2D = false;
      break;
    }
    sol::table row = rowObj.as<sol::table>();
    if (y == 0)
      width = row.size();
    if (row.size() != width || width == 0) {
      is2D = false;
      break;
    }
    for (size_t x = 0; x < width; x++) {
      if (!isElementLike(row[x + 1], elemType)) {
        is2D = false;
        break;
      }
    }
  }
  if (is2D) {
    items0 = width;
    items1 = n;
    return;
  }

  // 3D candidate: data[z][y][x]
  bool is3D = true;
  size_t rows = 0;
  width = 0;
  for (size_t z = 0; z < n && is3D; z++) {
    sol::object planeObj = data[z + 1];
    if (!planeObj.is<sol::table>()) {
      is3D = false;
      break;
    }
    sol::table plane = planeObj.as<sol::table>();
    if (z == 0)
      rows = plane.size();
    if (plane.size() != rows || rows == 0) {
      is3D = false;
      break;
    }
    for (size_t y = 0; y < rows && is3D; y++) {
      sol::object rowObj = plane[y + 1];
      if (!rowObj.is<sol::table>()) {
        is3D = false;
        break;
      }
      sol::table row = rowObj.as<sol::table>();
      if (z == 0 && y == 0)
        width = row.size();
      if (row.size() != width || width == 0) {
        is3D = false;
        break;
      }
      for (size_t x = 0; x < width; x++) {
        if (!isElementLike(row[x + 1], elemType)) {
          is3D = false;
          break;
        }
      }
    }
  }
  if (is3D) {
    items0 = width;
    items1 = rows;
    items2 = n;
    return;
  }

  throw std::runtime_error(
      "setParameterArray: failed to infer dimensions from data shape; "
      "provide explicit dimensions");
}

void arraySetDataFromLua(core::Array &arr, sol::table data, sol::this_state s)
{
  const size_t dim0 = arr.dim(0);
  const size_t dim1 = arr.dim(1);
  const size_t dim2 = arr.dim(2);

  std::vector<sol::object> flattened;

  // 2D nested form: data[y][x], with size [dim1][dim0]
  const bool canBeNested2D = dim2 == 1 && dim1 > 1 && data.size() == dim1;
  if (canBeNested2D) {
    bool valid = true;
    for (size_t y = 0; y < dim1 && valid; y++) {
      sol::object rowObj = data[y + 1];
      if (!rowObj.is<sol::table>()) {
        valid = false;
        break;
      }
      sol::table row = rowObj.as<sol::table>();
      if (row.size() != dim0) {
        valid = false;
        break;
      }
    }
    if (valid) {
      flattened.reserve(arr.size());
      for (size_t y = 0; y < dim1; y++) {
        sol::table row = data[y + 1];
        for (size_t x = 0; x < dim0; x++)
          flattened.push_back(row[x + 1]);
      }
    }
  }

  // 3D nested form: data[z][y][x], with size [dim2][dim1][dim0]
  if (flattened.empty() && dim2 > 1 && data.size() == dim2) {
    bool valid = true;
    for (size_t z = 0; z < dim2 && valid; z++) {
      sol::object planeObj = data[z + 1];
      if (!planeObj.is<sol::table>()) {
        valid = false;
        break;
      }
      sol::table plane = planeObj.as<sol::table>();
      if (plane.size() != dim1) {
        valid = false;
        break;
      }
      for (size_t y = 0; y < dim1; y++) {
        sol::object rowObj = plane[y + 1];
        if (!rowObj.is<sol::table>()) {
          valid = false;
          break;
        }
        sol::table row = rowObj.as<sol::table>();
        if (row.size() != dim0) {
          valid = false;
          break;
        }
      }
    }
    if (valid) {
      flattened.reserve(arr.size());
      for (size_t z = 0; z < dim2; z++) {
        sol::table plane = data[z + 1];
        for (size_t y = 0; y < dim1; y++) {
          sol::table row = plane[y + 1];
          for (size_t x = 0; x < dim0; x++)
            flattened.push_back(row[x + 1]);
        }
      }
    }
  }

  const bool usingNestedForm = !flattened.empty();
  const size_t count = usingNestedForm ? flattened.size() : data.size();
  if (count == 0)
    return;

  if (!usingNestedForm && count != arr.size()) {
    sol::state_view lua(s);
    std::string msg = fmt::format(
        "Array.setData: table size ({}) differs from array size ({})",
        count,
        arr.size());
    if (count > arr.size()) {
      msg += "; truncating";
    } else {
      msg += "; remaining elements unchanged";
    }
    lua["print"](msg);
  }

  auto elemType = arr.elementType();
  size_t copyCount = std::min(count, arr.size());

#define FILL_VEC(ANARI_TYPE, Vec, Elem, N, name)                               \
  if (elemType == ANARI_TYPE) {                                                \
    ScopedArrayMap<Vec> map(arr);                                              \
    if (usingNestedForm)                                                       \
      fillArrayFromFlattened<Vec, Elem, N>(map, copyCount, flattened, name);   \
    else                                                                       \
      fillArrayFromTable<Vec, Elem, N>(map, copyCount, data, name);            \
  }

  try {
    if (elemType == ANARI_FLOAT32) {
      ScopedArrayMap<float> map(arr);
      if (usingNestedForm) {
        for (size_t i = 0; i < copyCount; i++)
          map[i] = flattened[i].as<float>();
      } else {
        for (size_t i = 0; i < copyCount; i++)
          map[i] = data[i + 1].get<float>();
      }
    } else if (elemType == ANARI_INT32) {
      ScopedArrayMap<int32_t> map(arr);
      if (usingNestedForm) {
        for (size_t i = 0; i < copyCount; i++)
          map[i] = flattened[i].as<int32_t>();
      } else {
        for (size_t i = 0; i < copyCount; i++)
          map[i] = data[i + 1].get<int32_t>();
      }
    } else if (elemType == ANARI_UINT32) {
      ScopedArrayMap<uint32_t> map(arr);
      if (usingNestedForm) {
        for (size_t i = 0; i < copyCount; i++)
          map[i] = flattened[i].as<uint32_t>();
      } else {
        for (size_t i = 0; i < copyCount; i++)
          map[i] = data[i + 1].get<uint32_t>();
      }
    } else
      FILL_VEC(ANARI_FLOAT32_VEC2, math::float2, float, 2, "float2")
    else FILL_VEC(ANARI_FLOAT32_VEC3,
        math::float3,
        float,
        3,
        "float3") else FILL_VEC(ANARI_FLOAT32_VEC4,
        math::float4,
        float,
        4,
        "float4") else FILL_VEC(ANARI_INT32_VEC2,
        math::int2,
        int32_t,
        2,
        "int2") else FILL_VEC(ANARI_INT32_VEC3,
        math::int3,
        int32_t,
        3,
        "int3") else FILL_VEC(ANARI_INT32_VEC4,
        math::int4,
        int32_t,
        4,
        "int4") else FILL_VEC(ANARI_UINT32_VEC2,
        math::uint2,
        uint32_t,
        2,
        "uint2") else FILL_VEC(ANARI_UINT32_VEC3,
        math::uint3,
        uint32_t,
        3,
        "uint3") else FILL_VEC(ANARI_UINT32_VEC4,
        math::uint4,
        uint32_t,
        4,
        "uint4") else if (elemType == ANARI_FLOAT32_MAT4)
    {
      ScopedArrayMap<math::mat4> map(arr);
      if (usingNestedForm) {
        for (size_t i = 0; i < copyCount; i++)
          map[i] = flattened[i].as<math::mat4>();
      } else {
        for (size_t i = 0; i < copyCount; i++)
          map[i] = data[i + 1].get<math::mat4>();
      }
    }
    else
    {
      throw std::runtime_error("Array.setData: unsupported element type");
    }
  } catch (const sol::error &e) {
    throw std::runtime_error("Array.setData: failed to convert table element - "
        + std::string(e.what()));
  }

#undef FILL_VEC
}

core::ArrayRef setParameterArrayFromLua(core::Object &obj,
    const std::string &name,
    const std::string &typeStr,
    sol::table data,
    sol::this_state s)
{
  auto *scene = obj.scene();
  if (!scene)
    throw std::runtime_error(
        "setParameterArray: object is not attached to a scene");

  const auto elemType = arrayTypeFromString(typeStr);
  size_t items0 = 0, items1 = 0, items2 = 0;
  inferArrayDimsFromLuaData(data, elemType, items0, items1, items2);

  auto arr = scene->createArray(elemType, items0, items1, items2);
  if (!arr.valid())
    throw std::runtime_error("setParameterArray: failed to create array");

  arraySetDataFromLua(*arr.data(), data, s);
  obj.setParameterObject(core::Token(name), *arr.data());
  return arr;
}

core::ArrayRef setParameterArrayFromLua(core::Object &obj,
    const std::string &name,
    const std::string &typeStr,
    size_t items0,
    size_t items1,
    size_t items2,
    sol::table data,
    sol::this_state s)
{
  auto *scene = obj.scene();
  if (!scene)
    throw std::runtime_error(
        "setParameterArray: object is not attached to a scene");

  auto arr =
      scene->createArray(arrayTypeFromString(typeStr), items0, items1, items2);
  if (!arr.valid())
    throw std::runtime_error("setParameterArray: failed to create array");

  arraySetDataFromLua(*arr.data(), data, s);
  obj.setParameterObject(core::Token(name), *arr.data());
  return arr;
}

static sol::table arrayGetDataAsLua(core::Array &arr, sol::this_state s)
{
  sol::state_view lua(s);
  const auto count = arr.size();
  auto out = lua.create_table(static_cast<int>(count), 0);

  if (count == 0)
    return out;

  const auto elemType = arr.elementType();

#define GET_VEC(ANARI_TYPE, Vec, N)                                            \
  if (elemType == ANARI_TYPE) {                                                \
    ScopedArrayMap<Vec> map(arr);                                              \
    getVecArrayAsLua<Vec, N>(lua, map, count, out);                            \
  }

  if (elemType == ANARI_FLOAT32) {
    ScopedArrayMap<float> map(arr);
    for (size_t i = 0; i < count; i++)
      out[i + 1] = map[i];
  } else if (elemType == ANARI_INT32) {
    ScopedArrayMap<int32_t> map(arr);
    for (size_t i = 0; i < count; i++)
      out[i + 1] = map[i];
  } else if (elemType == ANARI_UINT32) {
    ScopedArrayMap<uint32_t> map(arr);
    for (size_t i = 0; i < count; i++)
      out[i + 1] = map[i];
  } else
    GET_VEC(ANARI_FLOAT32_VEC2, math::float2, 2)
  else GET_VEC(ANARI_FLOAT32_VEC3, math::float3, 3) else GET_VEC(
      ANARI_FLOAT32_VEC4, math::float4, 4) else GET_VEC(ANARI_INT32_VEC2,
      math::int2,
      2) else GET_VEC(ANARI_INT32_VEC3,
      math::int3,
      3) else GET_VEC(ANARI_INT32_VEC4,
      math::int4,
      4) else GET_VEC(ANARI_UINT32_VEC2,
      math::uint2,
      2) else GET_VEC(ANARI_UINT32_VEC3,
      math::uint3,
      3) else GET_VEC(ANARI_UINT32_VEC4, math::uint4, 4) else if (elemType
      == ANARI_FLOAT32_MAT4)
  {
    ScopedArrayMap<math::mat4> map(arr);
    for (size_t i = 0; i < count; i++)
      out[i + 1] = map[i];
  }
  else
  {
    throw std::runtime_error("Array.getData: unsupported element type");
  }

#undef GET_VEC

  return out;
}

// Register an ObjectPoolRef<T> usertype with forwarded Object methods.
//
// All Object methods are available directly on refs:
//   ref:setParameter("radius", 1.5)
//   ref.name = "myGeom"
//   ref:removeParameter("radius")
template <typename T>
auto registerObjectPoolRef(sol::table &tsd, const char *name)
{
  using Ref = core::ObjectPoolRef<T>;
  auto refType = tsd.new_usertype<Ref>(
      name,
      sol::no_constructor,
      "valid",
      &Ref::valid,
      "index",
      [](const Ref &r) -> size_t { return r.index(); },
      sol::meta_function::to_string,
      [name](const Ref &r) {
        if (!r.valid())
          return fmt::format("{}(invalid)", name);
        return fmt::format("{}({})", name, r.index());
      });

  registerObjectMethodsOn(refType,
      [](Ref &r) -> core::Object * { return r.valid() ? r.data() : nullptr; });

  return refType;
}

void registerObjectBindings(sol::state &lua)
{
  sol::table tsd = lua["tsd"];

  // Register concrete Object types first (metatables keyed by C++ type_index)
  // Then register Ref types which overwrite tsd["Name"] table entries.

  tsd.new_usertype<core::Geometry>("Geometry",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<core::Object>());

  tsd.new_usertype<core::Material>("Material",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<core::Object>());

  tsd.new_usertype<core::Light>("Light",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<core::Object>());

  tsd.new_usertype<core::Camera>("Camera",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<core::Object>());

  tsd.new_usertype<core::Surface>(
      "Surface",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<core::Object>(),
      sol::meta_function::equal_to,
      [](const core::Surface &a, const core::Surface &b) { return &a == &b; },
      sol::meta_function::less_than,
      [](const core::Surface &a, const core::Surface &b) {
        return std::less<const core::Surface *>{}(&a, &b);
      },
      "geometry",
      [](core::Surface &s) {
        return s.parameterValueAsObject<core::Geometry>("geometry");
      },
      "material",
      [](core::Surface &s) {
        return s.parameterValueAsObject<core::Material>("material");
      });

  tsd.new_usertype<core::Volume>("Volume",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<core::Object>(),
      "spatialField",
      [](core::Volume &v) {
        return v.parameterValueAsObject<core::SpatialField>("value");
      });

  tsd.new_usertype<core::Sampler>("Sampler",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<core::Object>());

  tsd.new_usertype<core::SpatialField>(
      "SpatialField",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<core::Object>(),
      sol::meta_function::equal_to,
      [](const core::SpatialField &a, const core::SpatialField &b) {
        return &a == &b;
      },
      sol::meta_function::less_than,
      [](const core::SpatialField &a, const core::SpatialField &b) {
        return std::less<const core::SpatialField *>{}(&a, &b);
      },
      "computeValueRange",
      &core::SpatialField::computeValueRange);

  tsd.new_usertype<core::Array>(
      "Array",
      sol::no_constructor,
      sol::base_classes,
      sol::bases<core::Object>(),
      sol::meta_function::equal_to,
      [](const core::Array &a, const core::Array &b) { return &a == &b; },
      sol::meta_function::less_than,
      [](const core::Array &a, const core::Array &b) {
        return std::less<const core::Array *>{}(&a, &b);
      },
      "elementType",
      &core::Array::elementType,
      "size",
      &core::Array::size,
      "elementSize",
      &core::Array::elementSize,
      "isEmpty",
      &core::Array::isEmpty,
      "dim",
      &core::Array::dim,
      "setData",
      [](core::Array &arr, sol::table data, sol::this_state s) {
        arraySetDataFromLua(arr, data, s);
      },
      "getData",
      [](core::Array &arr, sol::this_state s) {
        return arrayGetDataAsLua(arr, s);
      });

  registerObjectPoolRef<core::Geometry>(tsd, "Geometry");
  registerObjectPoolRef<core::Material>(tsd, "Material");
  registerObjectPoolRef<core::Light>(tsd, "Light");
  registerObjectPoolRef<core::Camera>(tsd, "Camera");
  registerObjectPoolRef<core::Sampler>(tsd, "Sampler");

  auto surfaceRefType = registerObjectPoolRef<core::Surface>(tsd, "Surface");
  surfaceRefType["geometry"] = +[](core::SurfaceRef &r) -> core::Geometry * {
    if (!r.valid())
      return nullptr;
    return r.data()->parameterValueAsObject<core::Geometry>("geometry");
  };
  surfaceRefType["material"] = +[](core::SurfaceRef &r) -> core::Material * {
    if (!r.valid())
      return nullptr;
    return r.data()->parameterValueAsObject<core::Material>("material");
  };

  auto volumeRefType = registerObjectPoolRef<core::Volume>(tsd, "Volume");
  volumeRefType["spatialField"] =
      +[](core::VolumeRef &r) -> core::SpatialField * {
    if (!r.valid())
      return nullptr;
    return r.data()->parameterValueAsObject<core::SpatialField>("value");
  };

  auto fieldRefType =
      registerObjectPoolRef<core::SpatialField>(tsd, "SpatialField");
  fieldRefType["computeValueRange"] =
      +[](core::SpatialFieldRef &r) -> math::float2 {
    if (!r.valid())
      return math::float2(0.f);
    return r.data()->computeValueRange();
  };

  auto arrayRefType = registerObjectPoolRef<core::Array>(tsd, "Array");
  arrayRefType["elementType"] = +[](const core::ArrayRef &r) -> ANARIDataType {
    return r.valid() ? r.data()->elementType() : ANARI_UNKNOWN;
  };
  arrayRefType["size"] = +[](const core::ArrayRef &r) -> size_t {
    return r.valid() ? r.data()->size() : 0;
  };
  arrayRefType["elementSize"] = +[](const core::ArrayRef &r) -> size_t {
    return r.valid() ? r.data()->elementSize() : 0;
  };
  arrayRefType["isEmpty"] = +[](const core::ArrayRef &r) -> bool {
    return r.valid() ? r.data()->isEmpty() : true;
  };
  arrayRefType["dim"] = +[](const core::ArrayRef &r, size_t d) -> size_t {
    return r.valid() ? r.data()->dim(d) : 0;
  };
  arrayRefType["setData"] =
      [](core::ArrayRef &r, sol::table data, sol::this_state s) {
        if (!r.valid())
          throw std::runtime_error("attempt to setData on invalid Array");
        arraySetDataFromLua(*r.data(), data, s);
      };
  arrayRefType["getData"] = [](core::ArrayRef &r, sol::this_state s) {
    if (!r.valid())
      throw std::runtime_error("attempt to getData on invalid Array");
    return arrayGetDataAsLua(*r.data(), s);
  };
}

} // namespace tsd::scripting

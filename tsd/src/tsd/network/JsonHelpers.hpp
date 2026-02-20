// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Any.hpp"
#include "tsd/core/scene/Object.hpp"
#include <nlohmann/json.hpp>
#include <optional>

namespace tsd::network {

inline std::optional<nlohmann::json> anyToJson(const tsd::core::Any &v)
{
  auto t = v.type();
  if (t == ANARI_BOOL)
    return v.get<bool>();
  if (t == ANARI_INT32)
    return v.get<int32_t>();
  if (t == ANARI_UINT32)
    return v.get<uint32_t>();
  if (t == ANARI_FLOAT32)
    return v.get<float>();
  if (t == ANARI_FLOAT64)
    return v.get<double>();
  if (t == ANARI_STRING)
    return v.getString();
  if (t == ANARI_FLOAT32_VEC2) {
    auto w = v.getAs<tsd::math::float2>(ANARI_FLOAT32_VEC2);
    return nlohmann::json{w.x, w.y};
  }
  if (t == ANARI_FLOAT32_VEC3) {
    auto w = v.getAs<tsd::math::float3>(ANARI_FLOAT32_VEC3);
    return nlohmann::json{w.x, w.y, w.z};
  }
  if (t == ANARI_FLOAT32_VEC4) {
    auto w = v.getAs<tsd::math::float4>(ANARI_FLOAT32_VEC4);
    return nlohmann::json{w.x, w.y, w.z, w.w};
  }
  if (t == ANARI_FLOAT32_BOX1) {
    auto w = v.getAs<tsd::math::box1>(ANARI_FLOAT32_BOX1);
    return nlohmann::json{w.lower, w.upper};
  }
  return std::nullopt;
}

inline void objectParamsToJson(
    nlohmann::json &j, const tsd::core::Object &obj)
{
  for (size_t i = 0; i < obj.numParameters(); i++) {
    const auto &val = obj.parameterAt(i).value();
    if (!val.valid() || val.holdsObject())
      continue;
    if (auto jv = anyToJson(val))
      j[obj.parameterNameAt(i)] = *jv;
  }
}

inline void objectMetadataToJson(
    nlohmann::json &j, const tsd::core::Object &obj)
{
  for (size_t i = 0; i < obj.numMetadata(); i++) {
    const char *name = obj.getMetadataName(i);
    if (!name || !name[0])
      continue;

    auto mv = obj.getMetadataValue(name);
    if (mv.valid() && !mv.holdsObject()) {
      if (auto jv = anyToJson(mv))
        j[name] = *jv;
      continue;
    }

    anari::DataType aType = ANARI_UNKNOWN;
    const void *aPtr = nullptr;
    size_t aSize = 0;
    obj.getMetadataArray(name, &aType, &aPtr, &aSize);
    if (!aPtr || aSize == 0)
      continue;

    auto arr = nlohmann::json::array();
    if (aType == ANARI_FLOAT32_VEC2) {
      const auto *pts = static_cast<const tsd::math::float2 *>(aPtr);
      for (size_t k = 0; k < aSize; k++)
        arr.push_back({pts[k].x, pts[k].y});
    } else if (aType == ANARI_FLOAT32_VEC4) {
      const auto *pts = static_cast<const tsd::math::float4 *>(aPtr);
      for (size_t k = 0; k < aSize; k++)
        arr.push_back({pts[k].x, pts[k].y, pts[k].z, pts[k].w});
    } else if (aType == ANARI_FLOAT32) {
      const auto *vals = static_cast<const float *>(aPtr);
      for (size_t k = 0; k < aSize; k++)
        arr.push_back(vals[k]);
    }
    if (!arr.empty())
      j[name] = arr;
  }
}

} // namespace tsd::network

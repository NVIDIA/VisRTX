// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/algorithms/computeScalarRange.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/Context.hpp"

#include "tsd/core/algorithms/detail/computeScalarRangeImpl.hpp"

namespace tsd::core {

tsd::math::float2 computeScalarRange(const Array &a)
{
  constexpr float maxFloat = std::numeric_limits<float>::max();
  tsd::math::float2 retval{maxFloat, -maxFloat};

  const anari::DataType type = a.elementType();
  const bool elementsAreArrays = anari::isArray(type);
  const bool elementsAreScalars =
      !anari::isObject(type) && anari::componentsOf(type) == 1;

  if (auto *ctx = a.context(); elementsAreArrays && ctx) {
    const auto *begin = (uint64_t *)a.data();
    const auto *end = begin + a.size();
    std::for_each(begin, end, [&](uint64_t idx) {
      tsd::math::float2 subRange{maxFloat, -maxFloat};
      if (auto subArray = ctx->getObject<Array>(idx); subArray)
        subRange = computeScalarRange(*subArray);
      retval.x = std::min(retval.x, subRange.x);
      retval.y = std::max(retval.y, subRange.y);
    });
  } else if (elementsAreScalars) {
    switch (type) {
    case ANARI_UFIXED8:
      retval = detail::computeScalarRange_ufixed8(a);
      break;
    case ANARI_UFIXED16:
      retval = detail::computeScalarRange_ufixed16(a);
      break;
    case ANARI_FIXED8:
      retval = detail::computeScalarRange_fixed8(a);
      break;
    case ANARI_FIXED16:
      retval = detail::computeScalarRange_fixed16(a);
      break;
    case ANARI_FLOAT32:
      retval = detail::computeScalarRange_float32(a);
      break;
    case ANARI_FLOAT64:
      retval = detail::computeScalarRange_float64(a);
      break;
    default:
      logWarning(
          "computeScalarRange() called on an "
          "array with incompatible element type '%s'",
          anari::toString(type));
      break;
    }
  }

  return retval;
}

} // namespace tsd::core

// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/objects/SpatialField.hpp"

#include "tsd/core/Logging.hpp"
#include "tsd/core/algorithms/computeScalarRange.hpp"
#include "tsd/core/scene/Context.hpp"
// std
#include <optional>

namespace tsd::core {

SpatialField::SpatialField(Token stype) : Object(ANARI_SPATIAL_FIELD, stype)
{
  if (stype == tokens::spatial_field::structuredRegular) {
    addParameter("origin")
        .setValue(tsd::math::float3(0.f, 0.f, 0.f))
        .setDescription("bottom-left corner of the field");
    addParameter("spacing")
        .setValue(tsd::math::float3(1.f, 1.f, 1.f))
        .setMin(tsd::math::float3(0.f, 0.f, 0.f))
        .setDescription("voxel size in object-space units");
    addParameter("filter")
        .setValue("linear")
        .setStringValues({"nearest", "linear"})
        .setStringSelection(1);
    addParameter("data")
        .setValue({ANARI_ARRAY3D, INVALID_INDEX})
        .setDescription("vertex-centered voxel data");
  } else if (stype == tokens::spatial_field::amr) {
    addParameter("gridOrigin")
        .setValue(tsd::math::float3(0.f, 0.f, 0.f))
        .setDescription("bottom-left corner of the field");
    addParameter("gridSpacing")
        .setValue(tsd::math::float3(1.f, 1.f, 1.f))
        .setMin(tsd::math::float3(0.f, 0.f, 0.f))
        .setDescription("voxel size in object-space units");
    addParameter("refinementRatio")
        .setValue({ANARI_ARRAY1D, INVALID_INDEX})
        .setDescription("array of refinement ratios between levels");
    addParameter("block.bounds")
        .setValue({ANARI_ARRAY1D, INVALID_INDEX})
        .setDescription("array of grid bounds, in voxel space, for each AMR block");
    addParameter("block.level")
        .setValue({ANARI_ARRAY1D, INVALID_INDEX})
        .setDescription("array of each block's refinement level");
    addParameter("block.data")
        .setValue({ANARI_ARRAY1D, INVALID_INDEX})
        .setDescription("array of 3D arrays containing the scalar voxel data (deprecated");
    addParameter("data")
        .setValue({ANARI_ARRAY1D, INVALID_INDEX})
        .setDescription("array of floats containing the scalar voxel data");
  } else if (stype == tokens::spatial_field::nanovdb) {
    addParameter("gridData")
        .setValue({ANARI_ARRAY1D, INVALID_INDEX})
        .setDescription("array containing serialzed NanoVDB grid");
    addParameter("filter").setValue("linear").setStringValues(
        {"linear", "nearest"});
  }
}

anari::Object SpatialField::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::SpatialField>(d, subtype().c_str());
}

tsd::math::float2 SpatialField::computeValueRange()
{
  tsd::math::float2 retval{0.f, 1.f};
  auto *ctx = this->context();
  if (!ctx)
    return retval;

  auto getDataRangeFromParameter =
      [&](Parameter *p) -> std::optional<tsd::math::float2> {
    if (!p || !anari::isArray(p->value().type()))
      return {};
    else if (auto a = ctx->getObject<Array>(p->value().getAsObjectIndex()); a)
      return computeScalarRange(*a);
    else
      return {};
  };

  if (subtype() == tokens::spatial_field::structuredRegular) {
    if (auto range = getDataRangeFromParameter(parameter("data")); range)
      retval = *range;
  } else if (subtype() == tokens::spatial_field::unstructured) {
    if (auto r = getDataRangeFromParameter(parameter("vertex.data")); r)
      retval = *r;
    else if (auto r = getDataRangeFromParameter(parameter("cell.data")); r)
      retval = *r;
  } else if (subtype() == tokens::spatial_field::nanovdb) {
    if (auto *range = parameter("range"); range)
      retval = range->value().get<tsd::math::float2>();
  } else if (subtype() == tokens::spatial_field::amr) {
    if (auto range = getDataRangeFromParameter(parameter("block.data")); range)
      retval = *range;
    else if (auto range = getDataRangeFromParameter(parameter("data")); range)
      retval = *range;
  } else {
    logWarning(
        "implementation not yet provided for computing the data "
        "range of '%s' spatial fields",
        subtype().c_str());
  }

  return retval;
}

namespace tokens::spatial_field {

const Token structuredRegular = "structuredRegular";
const Token unstructured = "unstructured";
const Token amr = "amr";
const Token nanovdb = "nanovdb";

} // namespace tokens::spatial_field

} // namespace tsd::core

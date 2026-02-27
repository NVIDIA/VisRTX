// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "computeVorticity.hpp"
// tsd_core
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/algorithms/vort.h"
// nanovdb
#include <nanovdb/NanoVDB.h>
// anari
#include <anari/frontend/anari_enums.h>
// std
#include <cstring>
#include <string>

namespace tsd::io {

using namespace tsd::core;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct FieldData
{
  std::vector<double> values;
  std::vector<double> x, y, z;
  size_t nx{0}, ny{0}, nz{0};
};

static bool extractStructuredRegular(
    Scene &scene, SpatialField *field, FieldData &out)
{
  auto *p = field->parameter("data");
  if (!p || !anari::isArray(p->value().type())) {
    logError("[computeVorticity] structuredRegular field has no 'data' array");
    return false;
  }

  auto arr = scene.getObject<Array>(p->value().getAsObjectIndex());
  if (!arr) {
    logError("[computeVorticity] structuredRegular 'data' array is invalid");
    return false;
  }

  out.nx = arr->dim(0);
  out.ny = arr->dim(1);
  out.nz = arr->dim(2);

  if (out.nx < 2 || out.ny < 2 || out.nz < 2) {
    logError("[computeVorticity] field dimensions must be >= 2 in each axis");
    return false;
  }

  if (arr->elementType() != ANARI_FLOAT32) {
    logError(
        "[computeVorticity] structuredRegular field element type must be "
        "ANARI_FLOAT32 (got %d)",
        (int)arr->elementType());
    return false;
  }

  auto *p_orig = field->parameter("origin");
  auto *p_spc = field->parameter("spacing");
  if (!p_orig || !p_spc) {
    logError("[computeVorticity] structuredRegular field missing origin/spacing");
    return false;
  }

  auto origin = p_orig->value().get<math::float3>();
  auto spacing = p_spc->value().get<math::float3>();

  // Build coordinate arrays
  out.x.resize(out.nx);
  out.y.resize(out.ny);
  out.z.resize(out.nz);
  for (size_t i = 0; i < out.nx; ++i)
    out.x[i] = origin.x + i * (double)spacing.x;
  for (size_t j = 0; j < out.ny; ++j)
    out.y[j] = origin.y + j * (double)spacing.y;
  for (size_t k = 0; k < out.nz; ++k)
    out.z[k] = origin.z + k * (double)spacing.z;

  // Convert float → double
  size_t total = out.nx * out.ny * out.nz;
  out.values.resize(total);
  const float *src = arr->dataAs<float>();
  for (size_t i = 0; i < total; ++i)
    out.values[i] = src[i];

  return true;
}

static bool extractNanoVDB(Scene &scene, SpatialField *field, FieldData &out)
{
  auto *p = field->parameter("data");
  if (!p || !anari::isArray(p->value().type())) {
    logError("[computeVorticity] nanovdb field has no 'data' array");
    return false;
  }

  auto arr = scene.getObject<Array>(p->value().getAsObjectIndex());
  if (!arr || !arr->data()) {
    logError("[computeVorticity] nanovdb 'data' array is invalid");
    return false;
  }

  const uint8_t *rawData = static_cast<const uint8_t *>(arr->data());
  const auto *meta =
      reinterpret_cast<const nanovdb::GridMetaData *>(rawData);

  if (meta->gridType() != nanovdb::GridType::Float) {
    logError(
        "[computeVorticity] nanovdb field must be float32 (GridType::Float)");
    return false;
  }

  const auto *grid =
      reinterpret_cast<const nanovdb::NanoGrid<float> *>(rawData);
  auto bbox = grid->indexBBox();
  auto lo = bbox.min();
  auto hi = bbox.max();

  out.nx = (size_t)(hi[0] - lo[0] + 1);
  out.ny = (size_t)(hi[1] - lo[1] + 1);
  out.nz = (size_t)(hi[2] - lo[2] + 1);

  if (out.nx < 2 || out.ny < 2 || out.nz < 2) {
    logError("[computeVorticity] nanovdb field dimensions must be >= 2");
    return false;
  }

  // Derive world-space coordinates from the grid map
  const auto &map = grid->map();
  auto worldLo = map.applyMap(nanovdb::Vec3d(lo[0], lo[1], lo[2]));
  auto worldHi = map.applyMap(nanovdb::Vec3d(hi[0], hi[1], hi[2]));

  double dx = (out.nx > 1) ? (worldHi[0] - worldLo[0]) / (out.nx - 1) : 1.0;
  double dy = (out.ny > 1) ? (worldHi[1] - worldLo[1]) / (out.ny - 1) : 1.0;
  double dz = (out.nz > 1) ? (worldHi[2] - worldLo[2]) / (out.nz - 1) : 1.0;

  out.x.resize(out.nx);
  out.y.resize(out.ny);
  out.z.resize(out.nz);
  for (size_t i = 0; i < out.nx; ++i)
    out.x[i] = worldLo[0] + i * dx;
  for (size_t j = 0; j < out.ny; ++j)
    out.y[j] = worldLo[1] + j * dy;
  for (size_t k = 0; k < out.nz; ++k)
    out.z[k] = worldLo[2] + k * dz;

  // Rasterize into a dense buffer using the accessor
  size_t total = out.nx * out.ny * out.nz;
  out.values.resize(total, 0.0);
  auto acc = grid->getAccessor();

  for (int k = lo[2]; k <= hi[2]; ++k) {
    size_t kk = (size_t)(k - lo[2]);
    for (int j = lo[1]; j <= hi[1]; ++j) {
      size_t jj = (size_t)(j - lo[1]);
      for (int i = lo[0]; i <= hi[0]; ++i) {
        size_t ii = (size_t)(i - lo[0]);
        out.values[kk * out.ny * out.nx + jj * out.nx + ii] =
            (double)acc.getValue(nanovdb::Coord(i, j, k));
      }
    }
  }

  return true;
}

static bool extractFieldData(Scene &scene, SpatialField *field, FieldData &out)
{
  if (!field) {
    logError("[computeVorticity] null SpatialField pointer");
    return false;
  }

  if (field->subtype() == tokens::spatial_field::structuredRegular) {
    return extractStructuredRegular(scene, field, out);
  } else if (field->subtype() == tokens::spatial_field::nanovdb) {
    return extractNanoVDB(scene, field, out);
  } else {
    logError(
        "[computeVorticity] unsupported SpatialField subtype '%s'; "
        "only structuredRegular and nanovdb are supported",
        field->subtype().c_str());
    return false;
  }
}

static VolumeRef makeOutputVolume(Scene &scene,
    const std::string &name,
    const std::vector<double> &data,
    size_t nx,
    size_t ny,
    size_t nz,
    const math::float3 &origin,
    const math::float3 &spacing,
    LayerNodeRef location)
{
  // Create output SpatialField
  auto field =
      scene.createObject<SpatialField>(tokens::spatial_field::structuredRegular);
  field->setName(name.c_str());
  field->setParameter("origin", origin);
  field->setParameter("spacing", spacing);

  // Fill output array
  auto dataArr = scene.createArray(ANARI_FLOAT32, nx, ny, nz);
  float *dst = dataArr->mapAs<float>();
  for (size_t i = 0; i < data.size(); ++i)
    dst[i] = (float)data[i];
  dataArr->unmap();
  field->setParameterObject("data", *dataArr);

  // Compute value range
  float2 valueRange = field->computeValueRange();

  // Create Volume node in the scene tree
  auto tx = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root());

  auto [inst, vol] = scene.insertNewChildObjectNode<Volume>(
      tx, tokens::volume::transferFunction1D);
  vol->setName(name.c_str());
  vol->setParameterObject("value", *field);
  vol->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);

  // Default colormap
  auto colorArr = scene.createArray(ANARI_FLOAT32_VEC4, 256);
  colorArr->setData(makeDefaultColorMap(256).data());
  vol->setParameterObject("color", *colorArr);

  return vol;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

VorticityResult computeVorticity(Scene &scene,
    SpatialField *u,
    SpatialField *v,
    SpatialField *w,
    LayerNodeRef location,
    VorticityOptions opts)
{
  VorticityResult result;

  logStatus("[computeVorticity] extracting field data...");

  FieldData uData, vData, wData;
  if (!extractFieldData(scene, u, uData))
    return result;
  if (!extractFieldData(scene, v, vData))
    return result;
  if (!extractFieldData(scene, w, wData))
    return result;

  // Verify matching dimensions
  if (uData.nx != vData.nx || uData.nx != wData.nx || uData.ny != vData.ny
      || uData.ny != wData.ny || uData.nz != vData.nz
      || uData.nz != wData.nz) {
    logError(
        "[computeVorticity] U/V/W fields have mismatched dimensions: "
        "U=(%zu,%zu,%zu) V=(%zu,%zu,%zu) W=(%zu,%zu,%zu)",
        uData.nx,
        uData.ny,
        uData.nz,
        vData.nx,
        vData.ny,
        vData.nz,
        wData.nx,
        wData.ny,
        wData.nz);
    return result;
  }

  size_t nx = uData.nx, ny = uData.ny, nz = uData.nz;
  size_t total = nx * ny * nz;

  logStatus(
      "[computeVorticity] computing vortical quantities on %zux%zux%zu grid...",
      nx,
      ny,
      nz);

  // Allocate output buffers
  std::vector<double> vorticityBuf(total, 0.0);
  std::vector<double> helicityBuf(total, 0.0);
  std::vector<double> lambda2Buf(total, 0.0);
  std::vector<double> qCritBuf(total, 0.0);

  // Run the vorticity computation
  vort(uData.values,
      vData.values,
      wData.values,
      uData.x,
      uData.y,
      uData.z,
      vorticityBuf,
      helicityBuf,
      lambda2Buf,
      qCritBuf,
      nx,
      ny,
      nz);

  logStatus("[computeVorticity] creating output volumes...");

  // Derive origin and spacing from the U field coordinates
  math::float3 origin{(float)uData.x[0], (float)uData.y[0], (float)uData.z[0]};
  math::float3 spacing{(float)(uData.x[1] - uData.x[0]),
      (float)(uData.y[1] - uData.y[0]),
      (float)(uData.z[1] - uData.z[0])};

  if (opts.lambda2)
    result.lambda2 = makeOutputVolume(
        scene, "lambda2", lambda2Buf, nx, ny, nz, origin, spacing, location);
  if (opts.qCriterion)
    result.qCriterion = makeOutputVolume(
        scene, "q_criterion", qCritBuf, nx, ny, nz, origin, spacing, location);
  if (opts.vorticity)
    result.vorticity = makeOutputVolume(
        scene, "vorticity", vorticityBuf, nx, ny, nz, origin, spacing, location);
  if (opts.helicity)
    result.helicity = makeOutputVolume(
        scene, "helicity", helicityBuf, nx, ny, nz, origin, spacing, location);

  logStatus("[computeVorticity] done.");
  return result;
}

} // namespace tsd::io

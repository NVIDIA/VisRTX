// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "computeVorticity.hpp"
// tsd_core
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/algorithms/vort.h"
#if TSD_USE_CUDA
#include "tsd/core/algorithms/vort_cuda.h"
#endif
// nanovdb
#include <nanovdb/NanoVDB.h>
// anari
#include <anari/frontend/anari_enums.h>
// std
#include <cstring>
#include <string>
#if TSD_USE_VTK
#include <vtkCellArray.h>
#include <vtkDataObject.h>
#include <vtkFloatArray.h>
#include <vtkGradientFilter.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkProbeFilter.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#endif

namespace tsd::io {

using namespace tsd::core;

// ---------------------------------------------------------------------------
// FieldData — holds a pointer to float velocity data plus coordinate arrays.
//
// For structuredRegular fields the data pointer is non-owning (points directly
// into the ANARI Array storage — no copy).  For NanoVDB fields the sparse
// grid is rasterized into ownedData (float, not double) and ptr is set to
// point at that buffer.
// ---------------------------------------------------------------------------

struct FieldData
{
  const float *ptr{
      nullptr}; // non-owning (structuredRegular) or alias into ownedData
  std::vector<float> ownedData; // owns data for NanoVDB rasterization
  std::vector<double> x, y, z; // world-space coordinate arrays
  size_t nx{0}, ny{0}, nz{0};
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

static void fillLinearCoords(
    std::vector<double> &dst, size_t n, double origin, double spacing)
{
  dst.resize(n);
  for (size_t i = 0; i < n; ++i)
    dst[i] = origin + i * spacing;
}

static bool readCoordsParam(Scene &scene,
    const SpatialField *field,
    const char *name,
    std::vector<double> &dst,
    size_t n)
{
  auto *cp = field->parameter(name);
  if (!cp || !anari::isArray(cp->value().type())) {
    logError("[computeVorticity] field missing coord parameter '%s'", name);
    return false;
  }
  auto ca = scene.getObject<Array>(cp->value().getAsObjectIndex());
  if (!ca) {
    logError("[computeVorticity] field coord parameter '%s' is invalid", name);
    return false;
  }
  if (ca->size() < n) {
    logError(
        "[computeVorticity] coord array '%s' has %zu element(s) but field "
        "dimension requires %zu",
        name,
        ca->size(),
        n);
    return false;
  }
  dst.resize(n);
  switch (ca->elementType()) {
  case ANARI_FLOAT32: {
    const float *src = ca->dataAs<float>();
    for (size_t i = 0; i < n; ++i)
      dst[i] = src[i];
    break;
  }
  case ANARI_FLOAT64: {
    const double *src = ca->dataAs<double>();
    for (size_t i = 0; i < n; ++i)
      dst[i] = src[i];
    break;
  }
  default:
    logError(
        "[computeVorticity] coord array '%s' has unsupported element type %d "
        "(expected ANARI_FLOAT32 or ANARI_FLOAT64)",
        name,
        (int)ca->elementType());
    return false;
  }
  return true;
}

static void rasterizeNanoVDB(const nanovdb::NanoGrid<float> *grid,
    nanovdb::Coord lo,
    nanovdb::Coord hi,
    FieldData &out)
{
  out.ownedData.resize(out.nx * out.ny * out.nz, 0.0f);
  out.ptr = out.ownedData.data();
  auto acc = grid->getAccessor();
  for (int k = lo[2]; k <= hi[2]; ++k) {
    for (int j = lo[1]; j <= hi[1]; ++j) {
      float *row = out.ownedData.data() + (size_t)(k - lo[2]) * out.ny * out.nx
          + (size_t)(j - lo[1]) * out.nx;
      for (int i = lo[0]; i <= hi[0]; ++i)
        row[i - lo[0]] = acc.getValue(nanovdb::Coord(i, j, k));
    }
  }
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

static bool extractStructuredRegular(
    Scene &scene, const SpatialField *field, FieldData &out)
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
    logError(
        "[computeVorticity] structuredRegular field missing origin/spacing");
    return false;
  }

  auto origin = p_orig->value().get<math::float3>();
  auto spacing = p_spc->value().get<math::float3>();

  // Point directly at the ANARI array's storage — no copy needed.
  out.ptr = arr->dataAs<float>();

  fillLinearCoords(out.x, out.nx, origin.x, spacing.x);
  fillLinearCoords(out.y, out.ny, origin.y, spacing.y);
  fillLinearCoords(out.z, out.nz, origin.z, spacing.z);

  return true;
}

static bool extractNanoVDB(
    Scene &scene, const SpatialField *field, FieldData &out)
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
  const auto *meta = reinterpret_cast<const nanovdb::GridMetaData *>(rawData);

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

  const auto &map = grid->map();
  auto worldLo = map.applyMap(nanovdb::Vec3d(lo[0], lo[1], lo[2]));
  auto worldHi = map.applyMap(nanovdb::Vec3d(hi[0], hi[1], hi[2]));

  double dx = (out.nx > 1) ? (worldHi[0] - worldLo[0]) / (out.nx - 1) : 1.0;
  double dy = (out.ny > 1) ? (worldHi[1] - worldLo[1]) / (out.ny - 1) : 1.0;
  double dz = (out.nz > 1) ? (worldHi[2] - worldLo[2]) / (out.nz - 1) : 1.0;

  fillLinearCoords(out.x, out.nx, worldLo[0], dx);
  fillLinearCoords(out.y, out.ny, worldLo[1], dy);
  fillLinearCoords(out.z, out.nz, worldLo[2], dz);
  rasterizeNanoVDB(grid, lo, hi, out);

  return true;
}

static bool extractStructuredRectilinear(
    Scene &scene, const SpatialField *field, FieldData &out)
{
  auto *p = field->parameter("data");
  if (!p || !anari::isArray(p->value().type())) {
    logError(
        "[computeVorticity] structuredRectilinear field has no 'data' array");
    return false;
  }

  auto arr = scene.getObject<Array>(p->value().getAsObjectIndex());
  if (!arr) {
    logError(
        "[computeVorticity] structuredRectilinear 'data' array is invalid");
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
        "[computeVorticity] structuredRectilinear field element type must be "
        "ANARI_FLOAT32 (got %d)",
        (int)arr->elementType());
    return false;
  }

  out.ptr = arr->dataAs<float>();

  return readCoordsParam(scene, field, "coordsX", out.x, out.nx)
      && readCoordsParam(scene, field, "coordsY", out.y, out.ny)
      && readCoordsParam(scene, field, "coordsZ", out.z, out.nz);
}

static bool extractNanoVDBRectilinear(
    Scene &scene, const SpatialField *field, FieldData &out)
{
  auto *p = field->parameter("data");
  if (!p || !anari::isArray(p->value().type())) {
    logError("[computeVorticity] nanovdbRectilinear field has no 'data' array");
    return false;
  }

  auto arr = scene.getObject<Array>(p->value().getAsObjectIndex());
  if (!arr || !arr->data()) {
    logError("[computeVorticity] nanovdbRectilinear 'data' array is invalid");
    return false;
  }

  const uint8_t *rawData = static_cast<const uint8_t *>(arr->data());
  const auto *meta = reinterpret_cast<const nanovdb::GridMetaData *>(rawData);

  if (meta->gridType() != nanovdb::GridType::Float) {
    logError(
        "[computeVorticity] nanovdbRectilinear field must be float32 "
        "(GridType::Float)");
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
    logError(
        "[computeVorticity] nanovdbRectilinear field dimensions must be >= 2");
    return false;
  }

  if (!readCoordsParam(scene, field, "coordsX", out.x, out.nx)
      || !readCoordsParam(scene, field, "coordsY", out.y, out.ny)
      || !readCoordsParam(scene, field, "coordsZ", out.z, out.nz))
    return false;

  rasterizeNanoVDB(grid, lo, hi, out);
  return true;
}

static bool extractFieldData(
    Scene &scene, const SpatialField *field, FieldData &out)
{
  if (!field) {
    logError("[computeVorticity] null SpatialField pointer");
    return false;
  }

  if (field->subtype() == tokens::spatial_field::structuredRegular) {
    return extractStructuredRegular(scene, field, out);
  } else if (field->subtype() == tokens::spatial_field::structuredRectilinear) {
    return extractStructuredRectilinear(scene, field, out);
  } else if (field->subtype() == tokens::spatial_field::nanovdb) {
    return extractNanoVDB(scene, field, out);
  } else if (field->subtype() == tokens::spatial_field::nanovdbRectilinear) {
    return extractNanoVDBRectilinear(scene, field, out);
  } else {
    logError(
        "[computeVorticity] unsupported SpatialField subtype '%s'; "
        "only structuredRegular, structuredRectilinear, nanovdb, and "
        "nanovdbRectilinear are supported (unstructured and amr require "
        "mesh interpolation first)",
        field->subtype().c_str());
    return false;
  }
}

// ---------------------------------------------------------------------------
// wrapFieldAsVolume — insert a transferFunction1D Volume node into the scene
// around an already-configured SpatialField.
// ---------------------------------------------------------------------------

static VolumeRef wrapFieldAsVolume(Scene &scene,
    const std::string &name,
    SpatialFieldRef &field,
    LayerNodeRef location)
{
  float2 valueRange = field->computeValueRange();

  auto tx = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root());

  auto [inst, vol] = scene.insertNewChildObjectNode<Volume>(
      tx, tokens::volume::transferFunction1D);
  vol->setName(name.c_str());
  vol->setParameterObject("value", *field);
  vol->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);

  auto colorArr = scene.createArray(ANARI_FLOAT32_VEC4, 256);
  colorArr->setData(makeDefaultColorMap(256).data());
  vol->setParameterObject("color", *colorArr);

  return vol;
}

// ---------------------------------------------------------------------------
// wrapAsVolume — structuredRegular variant (used by the VTK unstructured path
// which always produces a uniform resampled grid).
// ---------------------------------------------------------------------------

static VolumeRef wrapAsVolume(Scene &scene,
    const std::string &name,
    ArrayRef &dataArr,
    size_t nx,
    size_t ny,
    size_t nz,
    const math::float3 &origin,
    const math::float3 &spacing,
    LayerNodeRef location)
{
  auto field = scene.createObject<SpatialField>(
      tokens::spatial_field::structuredRegular);
  field->setName(name.c_str());
  field->setParameter("origin", origin);
  field->setParameter("spacing", spacing);
  field->setParameterObject("data", *dataArr);
  return wrapFieldAsVolume(scene, name, field, location);
}

// ---------------------------------------------------------------------------
// isUniformCoords — true when consecutive spacings agree to within 1e-5
// relative to the first interval (or the array has fewer than 2 entries).
// ---------------------------------------------------------------------------

static bool isUniformCoords(const std::vector<double> &coords)
{
  if (coords.size() < 2)
    return true;
  const double h = coords[1] - coords[0];
  const double tol = (h < 0 ? -h : h) * 1e-5;
  for (size_t i = 2; i < coords.size(); ++i) {
    const double d = (coords[i] - coords[i - 1]) - h;
    if (d > tol || d < -tol)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// wrapAsVolume — FieldData variant: preserves world-space mapping by choosing
// structuredRegular for uniform grids and structuredRectilinear for non-uniform
// ones (e.g. input was structuredRectilinear or nanovdbRectilinear).
// ---------------------------------------------------------------------------

static VolumeRef wrapAsVolume(Scene &scene,
    const std::string &name,
    ArrayRef &dataArr,
    const FieldData &src,
    LayerNodeRef location)
{
  if (isUniformCoords(src.x) && isUniformCoords(src.y)
      && isUniformCoords(src.z)) {
    const math::float3 origin{
        (float)src.x[0], (float)src.y[0], (float)src.z[0]};
    const math::float3 spacing{(float)(src.x[1] - src.x[0]),
        (float)(src.y[1] - src.y[0]),
        (float)(src.z[1] - src.z[0])};
    return wrapAsVolume(scene,
        name,
        dataArr,
        src.nx,
        src.ny,
        src.nz,
        origin,
        spacing,
        location);
  }

  // Non-uniform grid: build a structuredRectilinear SpatialField so the full
  // coordinate mapping is preserved.
  auto field = scene.createObject<SpatialField>(
      tokens::spatial_field::structuredRectilinear);
  field->setName(name.c_str());
  field->setParameterObject("data", *dataArr);

  auto makeCoordArray = [&](const std::vector<double> &coords) -> ArrayRef {
    auto arr = scene.createArray(ANARI_FLOAT32, coords.size());
    float *p = arr->mapAs<float>();
    for (size_t i = 0; i < coords.size(); ++i)
      p[i] = (float)coords[i];
    arr->unmap();
    return arr;
  };

  auto arrX = makeCoordArray(src.x);
  auto arrY = makeCoordArray(src.y);
  auto arrZ = makeCoordArray(src.z);
  field->setParameterObject("coordsX", *arrX);
  field->setParameterObject("coordsY", *arrY);
  field->setParameterObject("coordsZ", *arrZ);

  return wrapFieldAsVolume(scene, name, field, location);
}

// ---------------------------------------------------------------------------
// Unstructured path — CUDA (Green-Gauss) or VTK CPU fallback
// ---------------------------------------------------------------------------

#if TSD_USE_CUDA || TSD_USE_VTK

static VorticityResult computeVorticityUnstructured(Scene &scene,
    const SpatialField *u,
    const SpatialField *v,
    const SpatialField *w,
    LayerNodeRef location,
    VorticityOptions opts)
{
  VorticityResult result;

  // Step 1: numPoints from u's vertex.position array
  auto *posParam = u->parameter("vertex.position");
  if (!posParam || !anari::isArray(posParam->value().type())) {
    logError("[computeVorticity] unstructured field missing 'vertex.position'");
    return result;
  }
  auto posArr = scene.getObject<Array>(posParam->value().getAsObjectIndex());
  if (!posArr) {
    logError(
        "[computeVorticity] unstructured field 'vertex.position' is invalid");
    return result;
  }
  const size_t numPoints = posArr->dim(0);

  // Validate v and w have matching point count
  auto getNumPoints = [&](const SpatialField *field) -> size_t {
    auto *p = field->parameter("vertex.position");
    if (!p || !anari::isArray(p->value().type()))
      return 0;
    auto a = scene.getObject<Array>(p->value().getAsObjectIndex());
    return a ? a->dim(0) : 0;
  };
  if (getNumPoints(v) != numPoints || getNumPoints(w) != numPoints) {
    logError(
        "[computeVorticity] U/V/W unstructured fields have mismatched point "
        "counts");
    return result;
  }

  // Step 2: scalar velocity data from vertex.data
  auto getDataPtr = [&](const SpatialField *field) -> const float * {
    auto *p = field->parameter("vertex.data");
    if (!p || !anari::isArray(p->value().type()))
      return nullptr;
    auto a = scene.getObject<Array>(p->value().getAsObjectIndex());
    return a ? a->dataAs<float>() : nullptr;
  };
  const float *uPtr = getDataPtr(u);
  const float *vPtr = getDataPtr(v);
  const float *wPtr = getDataPtr(w);
  if (!uPtr || !vPtr || !wPtr) {
    logError(
        "[computeVorticity] unstructured fields missing 'vertex.data' scalar");
    return result;
  }

  // Step 3: get topology arrays
  auto *indexParam = u->parameter("index");
  auto *cellIdxParam = u->parameter("cell.index");
  auto *cellTypeParam = u->parameter("cell.type");
  if (!indexParam || !cellIdxParam || !cellTypeParam
      || !anari::isArray(indexParam->value().type())
      || !anari::isArray(cellIdxParam->value().type())
      || !anari::isArray(cellTypeParam->value().type())) {
    logError(
        "[computeVorticity] unstructured field missing topology parameters");
    return result;
  }
  auto connArr = scene.getObject<Array>(indexParam->value().getAsObjectIndex());
  auto cellIdxArr =
      scene.getObject<Array>(cellIdxParam->value().getAsObjectIndex());
  auto cellTypeArr =
      scene.getObject<Array>(cellTypeParam->value().getAsObjectIndex());

  const uint32_t *connectivity = connArr->dataAs<uint32_t>();
  const uint32_t *cellIndex = cellIdxArr->dataAs<uint32_t>();
  const uint8_t *cellTypeData = cellTypeArr->dataAs<uint8_t>();
  const size_t numCells = cellIdxArr->dim(0);
  const size_t connSize = connArr->dim(0);

  logStatus(
      "[computeVorticity] computing vortical quantities on %zu-point "
      "unstructured grid...",
      numPoints);

#if TSD_USE_CUDA

  // CUDA path: Green-Gauss gradient + full-resolution unstructured output.

  // Build vertex-to-cell CSR adjacency on the CPU.
  std::vector<uint32_t> vtxCellOffsets(numPoints + 1, 0);
  for (size_t ci = 0; ci < numCells; ++ci) {
    size_t start = cellIndex[ci];
    size_t end = (ci + 1 < numCells) ? cellIndex[ci + 1] : connSize;
    for (size_t j = start; j < end; ++j)
      vtxCellOffsets[connectivity[j] + 1]++;
  }
  for (size_t vv = 1; vv <= numPoints; ++vv)
    vtxCellOffsets[vv] += vtxCellOffsets[vv - 1];
  std::vector<uint32_t> vtxCellList(vtxCellOffsets[numPoints]);
  {
    std::vector<uint32_t> tmp(vtxCellOffsets.begin(), vtxCellOffsets.end());
    for (size_t ci = 0; ci < numCells; ++ci) {
      size_t start = cellIndex[ci];
      size_t end = (ci + 1 < numCells) ? cellIndex[ci + 1] : connSize;
      for (size_t j = start; j < end; ++j) {
        uint32_t vv = connectivity[j];
        vtxCellList[tmp[vv]++] = (uint32_t)ci;
      }
    }
  }

  // Allocate ANARI output arrays and map host pointers for in-place fill.
  auto makeOutBuf = [&](bool enabled) -> std::pair<ArrayRef, float *> {
    if (!enabled)
      return {};
    auto arr = scene.createArray(ANARI_FLOAT32, numPoints);
    return {arr, arr->mapAs<float>()};
  };
  auto [vorticityArr, vorticityOut] = makeOutBuf(opts.vorticity);
  auto [helicityArr, helicityOut] = makeOutBuf(opts.helicity);
  auto [lambda2Arr, lambda2Out] = makeOutBuf(opts.lambda2);
  auto [qCritArr, qCritOut] = makeOutBuf(opts.qCriterion);

  // positions: posArr stores float3 elements (x,y,z interleaved).
  const float *positions = reinterpret_cast<const float *>(posArr->data());

  vort_cuda_unstructured(positions,
      uPtr,
      vPtr,
      wPtr,
      connectivity,
      cellIndex,
      cellTypeData,
      vtxCellOffsets.data(),
      vtxCellList.data(),
      numPoints,
      numCells,
      connSize,
      vorticityOut,
      helicityOut,
      lambda2Out,
      qCritOut);

  if (vorticityArr)
    vorticityArr->unmap();
  if (helicityArr)
    helicityArr->unmap();
  if (lambda2Arr)
    lambda2Arr->unmap();
  if (qCritArr)
    qCritArr->unmap();

  // Wrap each output as an unstructured SpatialField sharing the topology
  // arrays from the u-field.  wrapFieldAsVolume adds the transfer function.
  auto makeUnstrField = [&](const std::string &name,
                            ArrayRef &arr) -> SpatialFieldRef {
    if (!arr)
      return {};
    auto f =
        scene.createObject<SpatialField>(tokens::spatial_field::unstructured);
    f->setName(name.c_str());
    f->setParameterObject("vertex.position", *posArr);
    f->setParameterObject("index", *connArr);
    f->setParameterObject("cell.index", *cellIdxArr);
    f->setParameterObject("cell.type", *cellTypeArr);
    f->setParameterObject("vertex.data", *arr);
    return f;
  };

  logStatus("[computeVorticity] creating output volumes...");
  if (vorticityArr) {
    auto f = makeUnstrField("vorticity", vorticityArr);
    result.vorticity = wrapFieldAsVolume(scene, "vorticity", f, location);
  }
  if (helicityArr) {
    auto f = makeUnstrField("helicity", helicityArr);
    result.helicity = wrapFieldAsVolume(scene, "helicity", f, location);
  }
  if (lambda2Arr) {
    auto f = makeUnstrField("lambda2", lambda2Arr);
    result.lambda2 = wrapFieldAsVolume(scene, "lambda2", f, location);
  }
  if (qCritArr) {
    auto f = makeUnstrField("q_criterion", qCritArr);
    result.qCriterion = wrapFieldAsVolume(scene, "q_criterion", f, location);
  }
  logStatus("[computeVorticity] done.");

#elif TSD_USE_VTK

  // VTK CPU fallback: vtkGradientFilter + vtkProbeFilter resample to 64³.

  // Step 4: reconstruct vtkUnstructuredGrid
  auto vgrid = vtkSmartPointer<vtkUnstructuredGrid>::New();

  auto points = vtkSmartPointer<vtkPoints>::New();
  points->SetDataTypeToFloat();
  points->SetNumberOfPoints(static_cast<vtkIdType>(numPoints));
  const auto *rawPos = posArr->dataAs<tsd::math::float3>();
  for (size_t i = 0; i < numPoints; ++i)
    points->SetPoint(
        static_cast<vtkIdType>(i), rawPos[i].x, rawPos[i].y, rawPos[i].z);
  vgrid->SetPoints(points);

  vgrid->Allocate(static_cast<vtkIdType>(numCells));
  for (size_t ci = 0; ci < numCells; ++ci) {
    size_t start = cellIndex[ci];
    size_t end = (ci + 1 < numCells) ? cellIndex[ci + 1] : connSize;
    size_t npts = end - start;
    std::vector<vtkIdType> ptIds(npts);
    for (size_t j = 0; j < npts; ++j)
      ptIds[j] = static_cast<vtkIdType>(connectivity[start + j]);
    vgrid->InsertNextCell(static_cast<int>(cellTypeData[ci]),
        static_cast<vtkIdType>(npts),
        ptIds.data());
  }

  // Add interleaved velocity array (u,v,w per point)
  auto velArr = vtkSmartPointer<vtkFloatArray>::New();
  velArr->SetName("velocity");
  velArr->SetNumberOfComponents(3);
  velArr->SetNumberOfTuples(static_cast<vtkIdType>(numPoints));
  for (size_t i = 0; i < numPoints; ++i) {
    float tuple[3] = {uPtr[i], vPtr[i], wPtr[i]};
    velArr->SetTuple(static_cast<vtkIdType>(i), tuple);
  }
  vgrid->GetPointData()->AddArray(velArr);

  // Step 5: run vtkGradientFilter to get the 9-component Jacobian
  auto gradFilter = vtkSmartPointer<vtkGradientFilter>::New();
  gradFilter->SetInputData(vgrid);
  gradFilter->SetInputArrayToProcess(
      0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "velocity");
  gradFilter->SetResultArrayName("gradients");
  gradFilter->SetComputeGradient(1);
  gradFilter->Update();

  auto *gradOutput = vtkUnstructuredGrid::SafeDownCast(gradFilter->GetOutput());
  if (!gradOutput) {
    logError("[computeVorticity] vtkGradientFilter failed");
    return result;
  }
  vtkDataArray *gradData = gradOutput->GetPointData()->GetArray("gradients");
  if (!gradData || gradData->GetNumberOfComponents() != 9) {
    logError("[computeVorticity] gradient filter output has unexpected format");
    return result;
  }

  // Step 6: unpack Jacobian into 9 double vectors
  // VTK ordering: d(comp0)/dx, d(comp0)/dy, d(comp0)/dz,
  //               d(comp1)/dx, d(comp1)/dy, d(comp1)/dz,
  //               d(comp2)/dx, d(comp2)/dy, d(comp2)/dz
  // → dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz
  std::vector<double> dux(numPoints), duy(numPoints), duz(numPoints);
  std::vector<double> dvx(numPoints), dvy(numPoints), dvz(numPoints);
  std::vector<double> dwx(numPoints), dwy(numPoints), dwz(numPoints);
  for (size_t i = 0; i < numPoints; ++i) {
    double t[9];
    gradData->GetTuple(static_cast<vtkIdType>(i), t);
    dux[i] = t[0];
    duy[i] = t[1];
    duz[i] = t[2];
    dvx[i] = t[3];
    dvy[i] = t[4];
    dvz[i] = t[5];
    dwx[i] = t[6];
    dwy[i] = t[7];
    dwz[i] = t[8];
  }

  // Step 7: allocate output arrays and compute
  auto makeOutBuf = [&](bool enabled) -> std::pair<ArrayRef, float *> {
    if (!enabled)
      return {};
    auto arr = scene.createArray(ANARI_FLOAT32, numPoints);
    return {arr, arr->mapAs<float>()};
  };
  auto [lambda2Arr, lambda2Out] = makeOutBuf(opts.lambda2);
  auto [qCritArr, qCritOut] = makeOutBuf(opts.qCriterion);
  auto [vorticityArr, vorticityOut] = makeOutBuf(opts.vorticity);
  auto [helicityArr, helicityOut] = makeOutBuf(opts.helicity);

  vort_from_jacobians(uPtr,
      vPtr,
      wPtr,
      dux.data(),
      dvx.data(),
      dwx.data(),
      duy.data(),
      dvy.data(),
      dwy.data(),
      duz.data(),
      dvz.data(),
      dwz.data(),
      vorticityOut,
      helicityOut,
      lambda2Out,
      qCritOut,
      numPoints);

  // Step 8: unmap
  if (lambda2Arr)
    lambda2Arr->unmap();
  if (qCritArr)
    qCritArr->unmap();
  if (vorticityArr)
    vorticityArr->unmap();
  if (helicityArr)
    helicityArr->unmap();

  // Step 9: Resample onto a structured regular grid and create volumes.
  // VisRTX does not support the "unstructured" SpatialField subtype, so we
  // interpolate the per-point output values onto an axis-aligned regular grid
  // using vtkProbeFilter, then wrap as structuredRegular which VisRTX does
  // support.

  // 9a. Add vortical output arrays back to vgrid as VTK point data.
  auto addToGrid = [&](const char *name, ArrayRef &arr, bool enabled) {
    if (!enabled || !arr)
      return;
    const float *src = arr->dataAs<float>(); // ANARI_FLOAT32, safe
    auto vtkarr = vtkSmartPointer<vtkFloatArray>::New();
    vtkarr->SetName(name);
    vtkarr->SetNumberOfComponents(1);
    vtkarr->SetNumberOfTuples(static_cast<vtkIdType>(numPoints));
    for (size_t i = 0; i < numPoints; ++i)
      vtkarr->SetValue(static_cast<vtkIdType>(i), src[i]);
    vgrid->GetPointData()->AddArray(vtkarr);
  };
  addToGrid("lambda2", lambda2Arr, opts.lambda2);
  addToGrid("q_criterion", qCritArr, opts.qCriterion);
  addToGrid("vorticity", vorticityArr, opts.vorticity);
  addToGrid("helicity", helicityArr, opts.helicity);

  // 9b. Compute output grid resolution proportional to bounding box extents.
  double bounds[6];
  vgrid->GetBounds(bounds);
  const double bx = bounds[1] - bounds[0];
  const double by = bounds[3] - bounds[2];
  const double bz = bounds[5] - bounds[4];
  const double maxB = std::max({bx, by, bz});
  if (maxB < 1e-10) {
    logError("[computeVorticity] degenerate mesh bounds, aborting resample");
    return result;
  }
  const int MAX_RES = 64;
  const size_t resX =
      std::max(size_t(2), size_t(std::round(MAX_RES * bx / maxB)));
  const size_t resY =
      std::max(size_t(2), size_t(std::round(MAX_RES * by / maxB)));
  const size_t resZ =
      std::max(size_t(2), size_t(std::round(MAX_RES * bz / maxB)));

  logStatus("[computeVorticity] resampling to %zu×%zu×%zu structured grid...",
      resX,
      resY,
      resZ);

  auto imData = vtkSmartPointer<vtkImageData>::New();
  imData->SetDimensions(
      static_cast<int>(resX), static_cast<int>(resY), static_cast<int>(resZ));
  imData->SetOrigin(bounds[0], bounds[2], bounds[4]);
  imData->SetSpacing(bx / (resX - 1), by / (resY - 1), bz / (resZ - 1));

  auto probe = vtkSmartPointer<vtkProbeFilter>::New();
  probe->SetSourceData(vgrid);
  probe->SetInputData(imData);
  probe->Update();

  auto *probed = vtkImageData::SafeDownCast(probe->GetOutput());
  if (!probed) {
    logError("[computeVorticity] vtkProbeFilter resampling failed");
    return result;
  }

  // 9c. Extract resampled arrays and wrap as structuredRegular volumes.
  const size_t resTotal = resX * resY * resZ;
  const math::float3 origin{
      (float)bounds[0], (float)bounds[2], (float)bounds[4]};
  const math::float3 spacing{(float)(bx / (resX - 1)),
      (float)(by / (resY - 1)),
      (float)(bz / (resZ - 1))};

  auto wrapResampled = [&](const char *name, bool enabled) -> VolumeRef {
    if (!enabled)
      return {};
    vtkDataArray *arr = probed->GetPointData()->GetArray(name);
    if (!arr)
      return {};
    auto outArr = scene.createArray(ANARI_FLOAT32, resX, resY, resZ);
    auto *buf = outArr->mapAs<float>();
    for (size_t i = 0; i < resTotal; ++i)
      buf[i] = static_cast<float>(arr->GetTuple1(static_cast<vtkIdType>(i)));
    outArr->unmap();
    return wrapAsVolume(
        scene, name, outArr, resX, resY, resZ, origin, spacing, location);
  };

  logStatus("[computeVorticity] creating output volumes...");
  result.lambda2 = wrapResampled("lambda2", opts.lambda2);
  result.qCriterion = wrapResampled("q_criterion", opts.qCriterion);
  result.vorticity = wrapResampled("vorticity", opts.vorticity);
  result.helicity = wrapResampled("helicity", opts.helicity);

  logStatus("[computeVorticity] done.");

#endif // TSD_USE_CUDA || TSD_USE_VTK (inner)

  return result;
}

#endif // TSD_USE_CUDA || TSD_USE_VTK

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

VorticityResult computeVorticity(Scene &scene,
    const SpatialField *u,
    const SpatialField *v,
    const SpatialField *w,
    LayerNodeRef location,
    VorticityOptions opts)
{
  VorticityResult result;

#if TSD_USE_CUDA || TSD_USE_VTK
  if (u && u->subtype() == tokens::spatial_field::unstructured)
    return computeVorticityUnstructured(scene, u, v, w, location, opts);
#endif

  logStatus("[computeVorticity] extracting field data...");

  FieldData uData, vData, wData;
  if (!extractFieldData(scene, u, uData))
    return result;
  if (!extractFieldData(scene, v, vData))
    return result;
  if (!extractFieldData(scene, w, wData))
    return result;

  if (uData.nx != vData.nx || uData.nx != wData.nx || uData.ny != vData.ny
      || uData.ny != wData.ny || uData.nz != vData.nz || uData.nz != wData.nz) {
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

  const size_t nx = uData.nx, ny = uData.ny, nz = uData.nz;

  logStatus(
      "[computeVorticity] computing vortical quantities on %zux%zux%zu grid...",
      nx,
      ny,
      nz);

  // Pre-allocate one ANARI float32 array per selected output and map it so
  // vort() can write directly into the final storage — no intermediate buffers.
  auto makeOutBuf = [&](bool enabled) -> std::pair<ArrayRef, float *> {
    if (!enabled)
      return {};
    auto arr = scene.createArray(ANARI_FLOAT32, nx, ny, nz);
    return {arr, arr->mapAs<float>()};
  };
  auto [lambda2Arr, lambda2Out] = makeOutBuf(opts.lambda2);
  auto [qCritArr, qCritOut] = makeOutBuf(opts.qCriterion);
  auto [vorticityArr, vorticityOut] = makeOutBuf(opts.vorticity);
  auto [helicityArr, helicityOut] = makeOutBuf(opts.helicity);

  // Compute — writes float* outputs directly; null outputs are skipped.
#if TSD_USE_CUDA
  vort_cuda(uData.ptr,
      vData.ptr,
      wData.ptr,
      uData.x.data(),
      uData.y.data(),
      uData.z.data(),
      vorticityOut,
      helicityOut,
      lambda2Out,
      qCritOut,
      nx,
      ny,
      nz);
#else
  vort(uData.ptr,
      vData.ptr,
      wData.ptr,
      uData.x.data(),
      uData.y.data(),
      uData.z.data(),
      vorticityOut,
      helicityOut,
      lambda2Out,
      qCritOut,
      nx,
      ny,
      nz);
#endif

  // Unmap output arrays before handing them to SpatialField
  if (lambda2Arr)
    lambda2Arr->unmap();
  if (qCritArr)
    qCritArr->unmap();
  if (vorticityArr)
    vorticityArr->unmap();
  if (helicityArr)
    helicityArr->unmap();

  logStatus("[computeVorticity] creating output volumes...");

  if (lambda2Arr)
    result.lambda2 =
        wrapAsVolume(scene, "lambda2", lambda2Arr, uData, location);
  if (qCritArr)
    result.qCriterion =
        wrapAsVolume(scene, "q_criterion", qCritArr, uData, location);
  if (vorticityArr)
    result.vorticity =
        wrapAsVolume(scene, "vorticity", vorticityArr, uData, location);
  if (helicityArr)
    result.helicity =
        wrapAsVolume(scene, "helicity", helicityArr, uData, location);

  logStatus("[computeVorticity] done.");
  return result;
}

} // namespace tsd::io

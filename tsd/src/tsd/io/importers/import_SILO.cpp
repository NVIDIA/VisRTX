// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_SILO
#define TSD_USE_SILO 1
#endif

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"

// anari
#include <anari/anari_cpp/Traits.h>

#if TSD_USE_SILO
// silo
#include <silo.h>
// std
#include <algorithm>
#include <cmath>
#include <filesystem>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#endif

namespace tsd::io {

using namespace tsd::core;

#if TSD_USE_SILO

// Silo ghost zone type constants (from silo.h):
// DB_GHOSTTYPE_NOGHOST (0) - real data, not a ghost/halo cell
// DB_GHOSTTYPE_INTDUP (1)  - duplicated internal ghost zone (halo cell)

// Helper: map Silo data types to ANARI types
static anari::DataType siloTypeToANARIType(int siloType)
{
  switch (siloType) {
  case DB_FLOAT:
    return ANARI_FLOAT32;
  case DB_DOUBLE:
    return ANARI_FLOAT64;
  case DB_INT:
    return ANARI_INT32;
  case DB_SHORT:
    return ANARI_INT16;
  case DB_LONG:
    return ANARI_INT64;
  case DB_CHAR:
    return ANARI_INT8;
  default:
    logWarning(
        "[import_SILO] unknown Silo data type %d, using float", siloType);
    return ANARI_FLOAT32;
  }
}

// ============================================================================
// Derived field computation functions
// ============================================================================

// Compute gradient in X direction (one-sided differences at boundaries)
static void computeGradX(const std::vector<float> &u,
    std::vector<float> &uGrad,
    int nx,
    int ny,
    int nz,
    float dx)
{
  if (std::abs(dx) < 1e-10f) {
    std::fill(uGrad.begin(), uGrad.end(), 0.0f);
    return;
  }

  for (int z = 0; z < nz; ++z) {
    size_t off = z * nx * ny;
    for (int row = 0; row < ny; ++row) {
      // One-sided forward difference at lower boundary
      if (nx > 1)
        uGrad[off + row * nx] =
            (u[off + row * nx + 1] - u[off + row * nx]) / dx;
      // One-sided backward difference at upper boundary
      if (nx > 1)
        uGrad[off + row * nx + nx - 1] =
            (u[off + row * nx + nx - 1] - u[off + row * nx + nx - 2]) / dx;
      // Interior points (central difference)
      for (int col = 1; col < nx - 1; ++col) {
        uGrad[off + row * nx + col] = 0.5f
            * (u[off + row * nx + col + 1] - u[off + row * nx + col - 1]) / dx;
      }
    }
  }
}

// Compute gradient in Y direction (one-sided differences at boundaries)
static void computeGradY(const std::vector<float> &v,
    std::vector<float> &vGrad,
    int nx,
    int ny,
    int nz,
    float dy)
{
  if (std::abs(dy) < 1e-10f) {
    std::fill(vGrad.begin(), vGrad.end(), 0.0f);
    return;
  }

  for (int z = 0; z < nz; ++z) {
    size_t off = z * nx * ny;
    for (int col = 0; col < nx; ++col) {
      // One-sided forward difference at lower boundary
      if (ny > 1)
        vGrad[0 * nx + col + off] =
            (v[1 * nx + col + off] - v[0 * nx + col + off]) / dy;
      // One-sided backward difference at upper boundary
      if (ny > 1)
        vGrad[(ny - 1) * nx + col + off] =
            (v[(ny - 1) * nx + col + off] - v[(ny - 2) * nx + col + off]) / dy;
      // Interior points (central difference)
      for (int row = 1; row < ny - 1; ++row) {
        vGrad[row * nx + col + off] = 0.5f
            * (v[(row + 1) * nx + col + off] - v[(row - 1) * nx + col + off])
            / dy;
      }
    }
  }
}

// Compute gradient in Z direction (non-uniform spacing)
// Note: w is zone-centered data (nz zones), z is node positions (nz+1 nodes)
// For zone i, we use the spacing between surrounding nodes
static void computeGradZ(const std::vector<float> &w,
    std::vector<float> &wGrad,
    int nx,
    int ny,
    int nz,
    const std::vector<float> &z)
{
  size_t off = nx * ny;
  for (int row = 0; row < ny; ++row) {
    for (int col = 0; col < nx; ++col) {
      // Boundaries (forward/backward difference)
      // Last zone
      if (nz > 1) {
        float dz = (z[nz] - z[nz - 1]);
        if (std::abs(dz) > 1e-10f)
          wGrad[(nz - 1) * off + row * nx + col] =
              (w[(nz - 1) * off + row * nx + col]
                  - w[(nz - 2) * off + row * nx + col])
              / dz;
        else
          wGrad[(nz - 1) * off + row * nx + col] = 0.0f;
      }

      // First zone
      if (nz > 1) {
        float dz = (z[1] - z[0]);
        if (std::abs(dz) > 1e-10f)
          wGrad[0 * off + row * nx + col] =
              (w[1 * off + row * nx + col] - w[0 * off + row * nx + col]) / dz;
        else
          wGrad[0 * off + row * nx + col] = 0.0f;
      }

      // Interior points (central difference)
      for (int zi = 1; zi < nz - 1; ++zi) {
        float dz = (z[zi + 1] - z[zi - 1]);
        if (std::abs(dz) > 1e-10f)
          wGrad[zi * off + row * nx + col] = 0.5f
              * (w[(zi + 1) * off + row * nx + col]
                  - w[(zi - 1) * off + row * nx + col])
              / dz;
        else
          wGrad[zi * off + row * nx + col] = 0.0f;
      }
    }
  }
}

// Compute all 9 velocity gradients
static void computeVelocityGradients(const std::vector<float> &vel1,
    const std::vector<float> &vel2,
    const std::vector<float> &vel3,
    int nx,
    int ny,
    int nz,
    float dx,
    float dy,
    const std::vector<float> &z,
    std::vector<float> &dux,
    std::vector<float> &duy,
    std::vector<float> &duz,
    std::vector<float> &dvx,
    std::vector<float> &dvy,
    std::vector<float> &dvz,
    std::vector<float> &dwx,
    std::vector<float> &dwy,
    std::vector<float> &dwz)
{
  computeGradX(vel1, dux, nx, ny, nz, dx);
  computeGradX(vel2, dvx, nx, ny, nz, dx);
  computeGradX(vel3, dwx, nx, ny, nz, dx);

  computeGradY(vel1, duy, nx, ny, nz, dy);
  computeGradY(vel2, dvy, nx, ny, nz, dy);
  computeGradY(vel3, dwy, nx, ny, nz, dy);

  computeGradZ(vel1, duz, nx, ny, nz, z);
  computeGradZ(vel2, dvz, nx, ny, nz, z);
  computeGradZ(vel3, dwz, nx, ny, nz, z);
}

// Compute lambda2 criterion from velocity gradients
static void computeLambda2(const std::vector<float> &dux,
    const std::vector<float> &duy,
    const std::vector<float> &duz,
    const std::vector<float> &dvx,
    const std::vector<float> &dvy,
    const std::vector<float> &dvz,
    const std::vector<float> &dwx,
    const std::vector<float> &dwy,
    const std::vector<float> &dwz,
    std::vector<float> &result)
{
  size_t len = dux.size();
  for (size_t i = 0; i < len; ++i) {
    // Strain rate tensor S = 0.5*(J + J^T)
    float s11 = dux[i];
    float s12 = 0.5f * (duy[i] + dvx[i]);
    float s13 = 0.5f * (duz[i] + dwx[i]);
    float s22 = dvy[i];
    float s23 = 0.5f * (dvz[i] + dwy[i]);
    float s33 = dwz[i];

    // Antisymmetric part Omega = 0.5*(J - J^T)
    float o12 = 0.5f * (duy[i] - dvx[i]);
    float o13 = 0.5f * (duz[i] - dwx[i]);
    float o23 = 0.5f * (dvz[i] - dwy[i]);

    // S^2 + Omega^2
    float m11 = s11 * s11 + s12 * s12 + s13 * s13 - o12 * o12 - o13 * o13;
    float m12 =
        s11 * s12 + s12 * s22 + s13 * s23 + o12 * (s11 - s22) + o13 * o23;
    float m13 =
        s11 * s13 + s12 * s23 + s13 * s33 + o13 * (s11 - s33) - o12 * o23;
    float m22 = s12 * s12 + s22 * s22 + s23 * s23 - o12 * o12 - o23 * o23;
    float m23 =
        s12 * s13 + s22 * s23 + s23 * s33 + o23 * (s22 - s33) + o12 * o13;
    float m33 = s13 * s13 + s23 * s23 + s33 * s33 - o13 * o13 - o23 * o23;

    // Compute eigenvalues of 3x3 symmetric matrix
    float p1 = m12 * m12 + m13 * m13 + m23 * m23;
    float q = (m11 + m22 + m33) / 3.0f;
    float p2 = (m11 - q) * (m11 - q) + (m22 - q) * (m22 - q)
        + (m33 - q) * (m33 - q) + 2.0f * p1;
    float p = std::sqrt(p2 / 6.0f);

    // Handle degenerate cases
    if (p < 1e-10f) {
      result[i] = 0.0f;
      continue;
    }

    float b11 = (m11 - q) / p;
    float b12 = m12 / p;
    float b13 = m13 / p;
    float b22 = (m22 - q) / p;
    float b23 = m23 / p;
    float b33 = (m33 - q) / p;

    float r = (b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13)
                  + b13 * (b12 * b23 - b22 * b13))
        / 2.0f;
    r = std::max(-1.0f, std::min(1.0f, r));

    float phi = std::acos(r) / 3.0f;
    float eig1 = q + 2.0f * p * std::cos(phi);
    float eig3 = q + 2.0f * p * std::cos(phi + (2.0f * M_PI / 3.0f));
    float eig2 = 3.0f * q - eig1 - eig3; // middle eigenvalue

    result[i] = -std::min(eig2, 0.0f);
  }
}

// Compute velocity magnitude from three velocity components
static void computeVelocityMagnitude(const std::vector<float> &vel1,
    const std::vector<float> &vel2,
    const std::vector<float> &vel3,
    std::vector<float> &result)
{
  size_t len = vel1.size();
  for (size_t i = 0; i < len; ++i) {
    result[i] =
        std::sqrt(vel1[i] * vel1[i] + vel2[i] * vel2[i] + vel3[i] * vel3[i]);
  }
}

// Compute vorticity magnitude from velocity gradients
// Vorticity ω = curl(v) = (dwdy - dvdz, dudz - dwdx, dvdx - dudy)
// Returns |ω|
static void computeVorticity(const std::vector<float> &dux,
    const std::vector<float> &duy,
    const std::vector<float> &duz,
    const std::vector<float> &dvx,
    const std::vector<float> &dvy,
    const std::vector<float> &dvz,
    const std::vector<float> &dwx,
    const std::vector<float> &dwy,
    const std::vector<float> &dwz,
    std::vector<float> &result)
{
  size_t len = dux.size();
  for (size_t i = 0; i < len; ++i) {
    float wx = dwy[i] - dvz[i]; // ∂w/∂y - ∂v/∂z
    float wy = duz[i] - dwx[i]; // ∂u/∂z - ∂w/∂x
    float wz = dvx[i] - duy[i]; // ∂v/∂x - ∂u/∂y
    result[i] = std::sqrt(wx * wx + wy * wy + wz * wz);
  }
}

// Compute Q-criterion from velocity gradients
// Q = 0.5 * (||Ω||² - ||S||²)
// where S is strain rate tensor and Ω is vorticity tensor
static void computeQCriterion(const std::vector<float> &dux,
    const std::vector<float> &duy,
    const std::vector<float> &duz,
    const std::vector<float> &dvx,
    const std::vector<float> &dvy,
    const std::vector<float> &dvz,
    const std::vector<float> &dwx,
    const std::vector<float> &dwy,
    const std::vector<float> &dwz,
    std::vector<float> &result)
{
  size_t len = dux.size();
  for (size_t i = 0; i < len; ++i) {
    // Strain rate tensor S = 0.5*(J + J^T)
    float s11 = dux[i];
    float s12 = 0.5f * (duy[i] + dvx[i]);
    float s13 = 0.5f * (duz[i] + dwx[i]);
    float s22 = dvy[i];
    float s23 = 0.5f * (dvz[i] + dwy[i]);
    float s33 = dwz[i];

    // Vorticity tensor Ω = 0.5*(J - J^T)
    float o12 = 0.5f * (duy[i] - dvx[i]);
    float o13 = 0.5f * (duz[i] - dwx[i]);
    float o23 = 0.5f * (dvz[i] - dwy[i]);

    // ||S||² (Frobenius norm squared)
    float s_norm_sq = s11 * s11 + s22 * s22 + s33 * s33
        + 2.0f * (s12 * s12 + s13 * s13 + s23 * s23);

    // ||Ω||² (Frobenius norm squared)
    float o_norm_sq = 2.0f * (o12 * o12 + o13 * o13 + o23 * o23);

    // Q = 0.5 * (||Ω||² - ||S||²)
    result[i] = 0.5f * (o_norm_sq - s_norm_sq);
  }
}

// Compute helicity from velocity and vorticity
// Helicity H = v · ω = v · (∇ × v)
static void computeHelicity(const std::vector<float> &vel1,
    const std::vector<float> &vel2,
    const std::vector<float> &vel3,
    const std::vector<float> &dux,
    const std::vector<float> &duy,
    const std::vector<float> &duz,
    const std::vector<float> &dvx,
    const std::vector<float> &dvy,
    const std::vector<float> &dvz,
    const std::vector<float> &dwx,
    const std::vector<float> &dwy,
    const std::vector<float> &dwz,
    std::vector<float> &result)
{
  size_t len = vel1.size();
  for (size_t i = 0; i < len; ++i) {
    // Vorticity components: ω = ∇ × v
    float wx = dwy[i] - dvz[i]; // ∂w/∂y - ∂v/∂z
    float wy = duz[i] - dwx[i]; // ∂u/∂z - ∂w/∂x
    float wz = dvx[i] - duy[i]; // ∂v/∂x - ∂u/∂y

    // Helicity: H = v · ω
    result[i] = vel1[i] * wx + vel2[i] * wy + vel3[i] * wz;
  }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Helper: create a SpatialField from a Silo quadmesh + quadvar
static SpatialFieldRef createFieldFromQuadMesh(Scene &scene,
    DBfile *dbfile,
    const char *meshName,
    const char *varName,
    const std::string &fieldName)
{
  // Read the mesh
  DBquadmesh *mesh = DBGetQuadmesh(dbfile, meshName);
  if (!mesh) {
    logError("[import_SILO] failed to read quad mesh '%s'", meshName);
    return {};
  }

  // Read the variable
  DBquadvar *var = DBGetQuadvar(dbfile, varName);
  if (!var) {
    logError("[import_SILO] failed to read quad var '%s'", varName);
    DBFreeQuadmesh(mesh);
    return {};
  }

  // Check environment variable to force uniform grid approximation
  bool useUniformGrid = false;
  if (const char *env = std::getenv("TSD_SILO_IMPORT_RECTILINEAR_AS_UNIFORM")) {
    auto s = std::string_view(env);
    useUniformGrid = s == "1" || s == "true";
  }

  // Create spatial field - use structuredRectilinear by default, unless env var
  // is set
  SpatialFieldRef field;
  if (useUniformGrid) {
    field = scene.createObject<SpatialField>(
        tokens::spatial_field::structuredRegular);
    logStatus(
        "[import_SILO] using uniform grid approximation (TSD_SILO_IMPORT_RECTILINEAR_AS_UNIFORM set)");
  } else {
    field = scene.createObject<SpatialField>(
        tokens::spatial_field::structuredRectilinear);
    logStatus(
        "[import_SILO] using rectilinear grid with explicit axis coordinates");
  }
  field->setName(fieldName.c_str());

  // Zones are defined between nodes, so zone count = node_count - 1
  int3 nodes(mesh->dims[0], mesh->dims[1], mesh->dims[2]);
  int3 zones(mesh->dims[0] - 1, mesh->dims[1] - 1, mesh->dims[2] - 1);

  logStatus(
      "[import_SILO] total node dims:"
      " %d x %d x %d, total zone dims: %d x %d x %d",
      nodes.x,
      nodes.y,
      nodes.z,
      zones.x,
      zones.y,
      zones.z);

  // Check for ghost zones, keeping only phony cells.
  // Phony is a term coming from the SILO headers.
  int3 minIndex(mesh->min_index[0], mesh->min_index[1], mesh->min_index[2]);
  int3 maxIndex(mesh->max_index[0], mesh->max_index[1], mesh->max_index[2]);
  bool hasGhostZones = minIndex != int3(0)
      || maxIndex
          != int3(mesh->dims[0] - 1, mesh->dims[1] - 1, mesh->dims[2] - 1);

  // Check if data is node-centered or cell-centered
  bool isNodeCentered = (var->centering == DB_NODECENT);

  if (hasGhostZones) {
    logStatus("[import_SILO] phony %s: [%d:%d, %d:%d, %d:%d]",
        isNodeCentered ? "nodes" : "cells",
        minIndex.x,
        maxIndex.x,
        minIndex.y,
        maxIndex.y,
        minIndex.z,
        maxIndex.z);
  }

  // Get origin and spacing from coordinates (use full extent including ghosts)
  float3 origin(0.f);
  float3 spacing(1.f);
  tsd::math::box3 roi;

  // Compute origin and spacing from the full grid extent
  for (int d = 0; d < 3; d++) {
    if (mesh->coords[d]) {
      if (mesh->datatype == DB_FLOAT) {
        float *c = (float *)mesh->coords[d];
        origin[d] = c[0]; // First node in full grid
        if (mesh->dims[d] > 1) {
          // Spacing from full grid extent
          spacing[d] = (c[mesh->dims[d] - 1] - c[0]) / nodes[d];
        }
        roi.lower[d] = c[minIndex[d]];
        roi.upper[d] = c[maxIndex[d]];
      } else if (mesh->datatype == DB_DOUBLE) {
        double *c = (double *)mesh->coords[d];
        origin[d] = c[0]; // First node in full grid
        if (mesh->dims[d] > 1) {
          // Spacing from full grid extent
          spacing[d] = (c[mesh->dims[d] - 1] - c[0]) / nodes[d];
        }
        roi.lower[d] = c[minIndex[d]];
        roi.upper[d] = c[maxIndex[d]];
      }
    }
  }

  logStatus(
      "[import_SILO] origin: %.9f %.9f %.9f, spacing: %.9f %.9f %.9f, %s values",
      origin.x,
      origin.y,
      origin.z,
      spacing.x,
      spacing.y,
      spacing.z,
      isNodeCentered ? "nodes" : "cells");

  if (useUniformGrid) {
    // Set uniform grid parameters
    field->setParameter("origin", origin);
    field->setParameter("spacing", spacing);
  } else {
    // Extract axis coordinates for rectilinear grid
    for (int d = 0; d < 3; d++) {
      const char *axisNames[] = {"coordsX", "coordsY", "coordsZ"};
      if (mesh->coords[d]) {
        auto axisArray = scene.createArray(ANARI_FLOAT32, mesh->dims[d]);
        float *axisData = (float *)axisArray->map();

        // Copy coordinates to float array
        if (mesh->datatype == DB_FLOAT) {
          float *c = (float *)mesh->coords[d];
          std::copy(c, c + mesh->dims[d], axisData);
        } else if (mesh->datatype == DB_DOUBLE) {
          double *c = (double *)mesh->coords[d];
          for (int i = 0; i < mesh->dims[d]; i++) {
            axisData[i] = static_cast<float>(c[i]);
          }
        }

        axisArray->unmap();
        field->setParameterObject(axisNames[d], *axisArray);
      }
    }
  }

  field->setParameter("dataCentering", isNodeCentered ? "node" : "cell");

  // Set ROI to exclude ghost zones if they exist
  if (hasGhostZones) {
    field->setParameter("roi", roi);
    logStatus(
        "[import_SILO] ROI set to %.9f => %.9f; %.9f => %.9f; %.9f => %.9f to exclude ghost zones",
        roi.lower.x,
        roi.upper.x,
        roi.lower.y,
        roi.upper.y,
        roi.lower.z,
        roi.upper.z);
  }

  // Load all variable data (including ghost zones)
  auto dataType = siloTypeToANARIType(var->datatype);
  ArrayRef dataArray;
  if (isNodeCentered) {
    dataArray = scene.createArray(dataType, nodes.x, nodes.y, nodes.z);
  } else {
    dataArray = scene.createArray(dataType, zones.x, zones.y, zones.z);
  }
  dataArray->setData(var->vals[0]);
  field->setParameterObject("data", *dataArray);

  DBFreeQuadmesh(mesh);
  DBFreeQuadvar(var);

  return field;
}

// Helper: create a SpatialField from a Silo ucdmesh + ucdvar
static SpatialFieldRef createFieldFromUcdMesh(Scene &scene,
    DBfile *dbfile,
    const char *meshName,
    const char *varName,
    const std::string &fieldName)
{
  // Read the mesh
  DBucdmesh *mesh = DBGetUcdmesh(dbfile, meshName);
  if (!mesh) {
    logError("[import_SILO] failed to read UCD mesh '%s'", meshName);
    return {};
  }

  // Read the variable
  DBucdvar *var = DBGetUcdvar(dbfile, varName);
  if (!var) {
    logError("[import_SILO] failed to read UCD var '%s'", varName);
    DBFreeUcdmesh(mesh);
    return {};
  }

  // Check for ghost zones using ghost_zone_labels
  bool hasGhostZones = false;
  std::vector<int> realZoneIndices;

  if (mesh->zones && mesh->zones->ghost_zone_labels) {
    hasGhostZones = true;
    logStatus("[import_SILO] UCD mesh '%s' has ghost zone labels", meshName);

    // Build list of real (non-ghost) zones
    for (int i = 0; i < mesh->zones->nzones; i++) {
      if (mesh->zones->ghost_zone_labels[i] == DB_GHOSTTYPE_NOGHOST) {
        realZoneIndices.push_back(i);
      }
    }

    logStatus("[import_SILO] %zu real zones out of %d total",
        realZoneIndices.size(),
        mesh->zones->nzones);
  }

  // Create unstructured spatial field
  auto field =
      scene.createObject<SpatialField>(tokens::spatial_field::unstructured);
  field->setName(fieldName.c_str());

  // Copy vertex positions
  size_t numVerts = mesh->nnodes;
  auto vertexPositions = scene.createArray(ANARI_FLOAT32_VEC3, numVerts);
  float3 *positions = (float3 *)vertexPositions->map();

  for (size_t i = 0; i < numVerts; i++) {
    float x = 0.f, y = 0.f, z = 0.f;
    if (mesh->datatype == DB_FLOAT) {
      x = ((float *)mesh->coords[0])[i];
      y = ((float *)mesh->coords[1])[i];
      if (mesh->ndims > 2)
        z = ((float *)mesh->coords[2])[i];
    } else if (mesh->datatype == DB_DOUBLE) {
      x = ((double *)mesh->coords[0])[i];
      y = ((double *)mesh->coords[1])[i];
      if (mesh->ndims > 2)
        z = ((double *)mesh->coords[2])[i];
    }
    positions[i] = float3(x, y, z);
  }
  vertexPositions->unmap();
  field->setParameterObject("vertex.position", *vertexPositions);

  // Copy cell data (indices and types) - only real zones
  if (mesh->zones && mesh->zones->nzones > 0) {
    DBzonelist *zl = mesh->zones;
    size_t numCells = hasGhostZones ? realZoneIndices.size() : zl->nzones;

    // For now, just pass through the nodelist and shape arrays
    // This is simplified - proper handling would convert to ANARI cell types
    auto cellIndex = scene.createArray(ANARI_UINT32, numCells);
    uint32_t *indices = (uint32_t *)cellIndex->map();

    if (hasGhostZones) {
      // Only include real zones
      for (size_t i = 0; i < realZoneIndices.size(); i++) {
        indices[i] = realZoneIndices[i] * 8; // Assuming hex cells for now
      }
    } else {
      for (size_t i = 0; i < numCells; i++) {
        indices[i] = i * 8; // Assuming hex cells for now
      }
    }
    cellIndex->unmap();
    field->setParameterObject("cell.index", *cellIndex);

    // Cell type (assume hexahedron = 12 in ANARI)
    auto cellType = scene.createArray(ANARI_UINT8, numCells);
    uint8_t *types = (uint8_t *)cellType->map();
    for (size_t i = 0; i < numCells; i++) {
      types[i] = 12; // ANARI_HEXAHEDRON
    }
    cellType->unmap();
    field->setParameterObject("cell.type", *cellType);
  }

  // Copy variable data - handle ghost zones for zone-centered variables
  auto dataType = siloTypeToANARIType(var->datatype);

  if (var->centering == DB_ZONECENT && hasGhostZones) {
    // Zone-centered variable: filter out ghost zones
    size_t numRealZones = realZoneIndices.size();
    auto dataArray = scene.createArray(dataType, numRealZones);
    void *dst = dataArray->map();
    size_t elementSize = anari::sizeOf(dataType);

    for (size_t i = 0; i < realZoneIndices.size(); i++) {
      int srcIdx = realZoneIndices[i];
      std::memcpy((char *)dst + i * elementSize,
          (char *)var->vals[0] + srcIdx * elementSize,
          elementSize);
    }
    dataArray->unmap();
    field->setParameterObject("cell.data", *dataArray);

    logStatus("[import_SILO] copied zone-centered data for %zu real zones",
        numRealZones);
  } else if (var->centering == DB_NODECENT) {
    // Node-centered variable: copy all node data
    // Note: could also filter ghost nodes if mesh->ghost_node_labels exists
    auto dataArray = scene.createArray(dataType, var->nvals);
    void *dst = dataArray->map();
    void *src = var->vals[0];
    std::memcpy(dst, src, var->nvals * anari::sizeOf(dataType));
    dataArray->unmap();
    field->setParameterObject("vertex.data", *dataArray);
  } else {
    // No ghost zones or other centering, copy as-is
    auto dataArray = scene.createArray(dataType, var->nvals);
    void *dst = dataArray->map();
    void *src = var->vals[0];
    std::memcpy(dst, src, var->nvals * anari::sizeOf(dataType));
    dataArray->unmap();
    field->setParameterObject("vertex.data", *dataArray);
  }

  DBFreeUcdmesh(mesh);
  DBFreeUcdvar(var);

  return field;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static SpatialFieldRef import_SILO_singleMesh(
    Scene &scene, DBfile *dbfile, DBtoc *toc, const std::string &varName)
{
  // Find the variable to visualize
  const char *targetVarName = nullptr;
  int targetVarType = DB_INVALID_OBJECT;

  if (!varName.empty()) {
    // Try to find the requested variable
    targetVarType = DBInqVarType(dbfile, varName.c_str());
    if (targetVarType == DB_QUADVAR || targetVarType == DB_UCDVAR) {
      targetVarName = varName.c_str();
    }
  }

  // If no variable specified or not found, use first available
  if (!targetVarName) {
    if (toc->nqvar > 0) {
      targetVarName = toc->qvar_names[0];
      targetVarType = DB_QUADVAR;
    } else if (toc->nucdvar > 0) {
      targetVarName = toc->ucdvar_names[0];
      targetVarType = DB_UCDVAR;
    }
  }

  if (!targetVarName) {
    logError("[import_SILO] no variables found in file");
    return {};
  }

  SpatialFieldRef field;
  if (targetVarType == DB_QUADVAR) {
    DBquadvar *qv = DBGetQuadvar(dbfile, targetVarName);
    if (qv && qv->meshname) {
      field = createFieldFromQuadMesh(
          scene, dbfile, qv->meshname, targetVarName, targetVarName);
      DBFreeQuadvar(qv);
    }
  } else if (targetVarType == DB_UCDVAR) {
    DBucdvar *uv = DBGetUcdvar(dbfile, targetVarName);
    if (uv && uv->meshname) {
      field = createFieldFromUcdMesh(
          scene, dbfile, uv->meshname, targetVarName, targetVarName);
      DBFreeUcdvar(uv);
    }
  }

  return field;
}

static SpatialFieldRef import_SILO_multiMesh(Scene &scene,
    const std::string &file,
    DBfile *dbfile,
    DBtoc *toc,
    const std::string &varName,
    bool isDerivedField,
    const std::string &derivedFieldType)
{
  // Find the multivar to use
  const char *multivarName = nullptr;
  DBmultivar *mv_vel1 = nullptr, *mv_vel2 = nullptr, *mv_vel3 = nullptr;

  if (isDerivedField) {
    // For derived fields, find vel1, vel2, vel3 multivars
    for (int i = 0; i < toc->nmultivar; i++) {
      std::string mvName = toc->multivar_names[i];
      if (mvName == "vel1") {
        mv_vel1 = DBGetMultivar(dbfile, "vel1");
        multivarName = "vel1"; // Use vel1 as reference for structure
      } else if (mvName == "vel2") {
        mv_vel2 = DBGetMultivar(dbfile, "vel2");
      } else if (mvName == "vel3") {
        mv_vel3 = DBGetMultivar(dbfile, "vel3");
      }
    }

    if (!mv_vel1 || !mv_vel2 || !mv_vel3) {
      logError(
          "[import_SILO] derived field '%s' requires vel1, vel2, vel3 multivars",
          derivedFieldType.c_str());
      if (mv_vel1)
        DBFreeMultivar(mv_vel1);
      if (mv_vel2)
        DBFreeMultivar(mv_vel2);
      if (mv_vel3)
        DBFreeMultivar(mv_vel3);
      return {};
    }

    logStatus("[import_SILO] found velocity components for derived field");
  } else if (!varName.empty() && toc->nmultivar > 0) {
    // Search for the requested multivar
    for (int i = 0; i < toc->nmultivar; i++) {
      if (varName == toc->multivar_names[i]) {
        multivarName = toc->multivar_names[i];
        break;
      }
    }
  } else if (toc->nmultivar > 0) {
    // Use first available multivar
    multivarName = toc->multivar_names[0];
  }

  if (!multivarName) {
    logError(
        "[import_SILO] no multivar found (requested: %s)", varName.c_str());
    return {};
  }

  DBmultivar *mv = nullptr;
  if (!isDerivedField) {
    mv = DBGetMultivar(dbfile, multivarName);
    if (!mv) {
      logError("[import_SILO] failed to read multivar '%s'", multivarName);
      return {};
    }
  } else {
    // For derived fields, use vel1 as the reference structure
    mv = mv_vel1;
  }

  // First pass: collect block metadata to determine global grid
  struct BlockInfo
  {
    std::string file;
    std::string varName;
    float3 origin;
    float3 spacing;
    int3 dims;
    int varType;
  };

  std::vector<BlockInfo> blocks;
  float3 globalSpacing(1.f, 1.f, 1.f);
  float3 globalMin(std::numeric_limits<float>::max());
  float3 globalMax(std::numeric_limits<float>::lowest());
  // Derived fields are always computed as float32
  anari::DataType globalDataType = ANARI_FLOAT32;
  bool firstBlock = true;
  bool isNodeCentered = true; // Default to node-centered

  std::filesystem::path basePath = std::filesystem::path(file).parent_path();

  logStatus(
      "[import_SILO] collecting %d blocks to create unified volume", mv->nvars);

  for (int i = 0; i < mv->nvars; i++) {
    const char *varPath = mv->varnames[i];
    if (!varPath || strlen(varPath) == 0)
      continue;

    // Parse block name (format: "../dir/file.silo:varname" or "varname")
    std::string blockPath(varPath);
    std::string blockFile = file;
    std::string blockVarName;

    size_t blockColon = blockPath.find(':');
    if (blockColon != std::string::npos) {
      std::string relPath = blockPath.substr(0, blockColon);
      blockVarName = blockPath.substr(blockColon + 1);

      // Resolve relative path
      std::filesystem::path fullPath = basePath / relPath;
      blockFile = fullPath.string();
    } else {
      blockVarName = blockPath;
    }

    // Open block file
    DBfile *blockDbfile = nullptr;
    if (blockFile == file) {
      blockDbfile = dbfile; // Same file
    } else {
      blockDbfile = DBOpen(blockFile.c_str(), DB_UNKNOWN, DB_READ);
      if (!blockDbfile) {
        logWarning(
            "[import_SILO] failed to open block file '%s'", blockFile.c_str());
        continue;
      }
    }

    // Determine variable type
    int varType = DBInqVarType(blockDbfile, blockVarName.c_str());

    // Get block metadata
    BlockInfo info;
    info.file = blockFile;
    info.varName = blockVarName;
    info.varType = varType;

    if (varType == DB_QUADVAR) {
      DBquadvar *qv = DBGetQuadvar(blockDbfile, blockVarName.c_str());
      DBquadmesh *qm = nullptr;

      if (qv && qv->meshname) {
        qm = DBGetQuadmesh(blockDbfile, qv->meshname);

        if (qm) {
          // Get real node bounds (accounting for ghost zones)
          int3 realNodeFirst(0, 0, 0);
          int3 realNodeLast(qm->dims[0] - 1, qm->dims[1] - 1, qm->dims[2] - 1);

          if (qm->ghost_zone_labels) {
            for (int j = 0; j < qm->ndims && j < 3; j++) {
              realNodeFirst[j] = qm->min_index[j];
              realNodeLast[j] = qm->max_index[j];
            }
          }

          // Get spacing from the first real block
          if (firstBlock) {
            // Compute spacing from consecutive real coordinates
            if (qm->coords[0] && qm->dims[0] > realNodeFirst.x + 1) {
              if (qm->datatype == DB_FLOAT) {
                float *x = (float *)qm->coords[0];
                globalSpacing.x = (x[realNodeLast.x] - x[realNodeFirst.x])
                    / (qm->dims[0] + (isNodeCentered ? -1 : 0));
              } else if (qm->datatype == DB_DOUBLE) {
                double *x = (double *)qm->coords[0];
                globalSpacing.x = (x[realNodeLast.x] - x[realNodeFirst.x])
                    / (qm->dims[0] + (isNodeCentered ? -1 : 0));
              }
            }
            if (qm->coords[1] && qm->dims[1] > realNodeFirst.y + 1) {
              if (qm->datatype == DB_FLOAT) {
                float *y = (float *)qm->coords[1];
                globalSpacing.y = (y[realNodeLast.y] - y[realNodeFirst.y])
                    / (qm->dims[1] + (isNodeCentered ? -1 : 0));
              } else if (qm->datatype == DB_DOUBLE) {
                double *y = (double *)qm->coords[1];
                globalSpacing.y = (y[realNodeLast.y] - y[realNodeFirst.y])
                    / (qm->dims[1] + (isNodeCentered ? -1 : 0));
              }
            }
            if (qm->ndims > 2 && qm->coords[2]
                && qm->dims[2] > realNodeFirst.z + 1) {
              if (qm->datatype == DB_FLOAT) {
                float *z = (float *)qm->coords[2];
                globalSpacing.z = (z[realNodeLast.z] - z[realNodeFirst.z])
                    / (qm->dims[2] + (isNodeCentered ? -1 : 0));
              } else if (qm->datatype == DB_DOUBLE) {
                double *z = (double *)qm->coords[2];
                globalSpacing.z = (z[realNodeLast.z] - z[realNodeFirst.z])
                    / (qm->dims[2] + (isNodeCentered ? -1 : 0));
              }
            }
            if (!isDerivedField) {
              globalDataType = siloTypeToANARIType(qv->datatype);
            }
            firstBlock = false;
          }

          // Calculate origin and extents
          // The origin is at the first real node, use realNodeFirst as index
          float3 blockOrigin(0.f);
          if (qm->coords[0]) {
            if (qm->datatype == DB_FLOAT)
              blockOrigin.x = ((float *)qm->coords[0])[realNodeFirst.x];
            else if (qm->datatype == DB_DOUBLE)
              blockOrigin.x = ((double *)qm->coords[0])[realNodeFirst.x];
          }
          if (qm->coords[1]) {
            if (qm->datatype == DB_FLOAT)
              blockOrigin.y = ((float *)qm->coords[1])[realNodeFirst.y];
            else if (qm->datatype == DB_DOUBLE)
              blockOrigin.y = ((double *)qm->coords[1])[realNodeFirst.y];
          }
          if (qm->ndims > 2 && qm->coords[2]) {
            if (qm->datatype == DB_FLOAT)
              blockOrigin.z = ((float *)qm->coords[2])[realNodeFirst.z];
            else if (qm->datatype == DB_DOUBLE)
              blockOrigin.z = ((double *)qm->coords[2])[realNodeFirst.z];
          }

          info.origin = blockOrigin;
          info.spacing = globalSpacing;

          // Capture centering from first block
          if (firstBlock) {
            isNodeCentered = (qv->centering == DB_NODECENT);
          }

          // Dims are in zones (nodes - 1)
          info.dims = int3(realNodeLast.x - realNodeFirst.x,
              realNodeLast.y - realNodeFirst.y,
              realNodeLast.z - realNodeFirst.z);

          // Update global bounds
          float3 blockMax = blockOrigin + float3(info.dims) * globalSpacing;
          globalMin = min(globalMin, blockOrigin);
          globalMax = max(globalMax, blockMax);

          DBFreeQuadmesh(qm);
        }
        DBFreeQuadvar(qv);
      }
    }

    // Close block file if it's different
    if (blockDbfile != dbfile)
      DBClose(blockDbfile);

    if (info.dims.x > 0) {
      blocks.push_back(info);
    }
  }

  // Don't free multivar yet - we need it for the second pass
  // It will be freed after the second pass completes

  if (blocks.empty()) {
    logError("[import_SILO] no valid blocks found");
    if (isDerivedField) {
      if (mv_vel1)
        DBFreeMultivar(mv_vel1);
      if (mv_vel2)
        DBFreeMultivar(mv_vel2);
      if (mv_vel3)
        DBFreeMultivar(mv_vel3);
    } else {
      DBFreeMultivar(mv);
    }
    return {};
  }

  // Calculate global grid dimensions
  float3 globalExtent = globalMax - globalMin;
  int3 globalDims((int)std::round(globalExtent.x / globalSpacing.x),
      (int)std::round(globalExtent.y / globalSpacing.y),
      (int)std::round(globalExtent.z / globalSpacing.z));

  if (globalDims.x < 1)
    globalDims.x = 1;
  if (globalDims.y < 1)
    globalDims.y = 1;
  if (globalDims.z < 1)
    globalDims.z = 1;

  logStatus(
      "[import_SILO] unified volume:"
      " origin=(%.2f,%.2f,%.2f) spacing=(%.2f,%.2f,%.2f) dims=%dx%dx%d",
      globalMin.x,
      globalMin.y,
      globalMin.z,
      globalSpacing.x,
      globalSpacing.y,
      globalSpacing.z,
      globalDims.x,
      globalDims.y,
      globalDims.z);

  // Create unified data array
  size_t totalElements = (size_t)globalDims.x * globalDims.y * globalDims.z;
  auto unifiedDataArray = scene.createArray(
      globalDataType, globalDims.x, globalDims.y, globalDims.z);
  void *unifiedData = unifiedDataArray->map();

  // Initialize to zero/background value
  std::memset(unifiedData, 0, totalElements * anari::sizeOf(globalDataType));

  // Second pass: copy each block's data into the unified grid
  for (size_t blockIdx = 0; blockIdx < blocks.size(); blockIdx++) {
    const BlockInfo &info = blocks[blockIdx];

    // Open block file
    DBfile *blockDbfile = nullptr;
    if (info.file == file) {
      blockDbfile = dbfile;
    } else {
      blockDbfile = DBOpen(info.file.c_str(), DB_UNKNOWN, DB_READ);
      if (!blockDbfile)
        continue;
    }

    if (info.varType == DB_QUADVAR) {
      // Load velocity components (either single var or vel1/vel2/vel3 for
      // derived fields)
      DBquadvar *qv_vel1 = nullptr, *qv_vel2 = nullptr, *qv_vel3 = nullptr;
      DBquadmesh *qm = nullptr;

      if (isDerivedField) {
        // Load all three velocity components
        qv_vel1 = DBGetQuadvar(blockDbfile, info.varName.c_str()); // vel1
        if (qv_vel1 && qv_vel1->meshname) {
          qm = DBGetQuadmesh(blockDbfile, qv_vel1->meshname);
          // Find corresponding vel2 and vel3 blocks
          std::string vel2Name = info.varName;
          std::string vel3Name = info.varName;
          size_t pos = vel2Name.find("vel1");
          if (pos != std::string::npos) {
            vel2Name.replace(pos, 4, "vel2");
            vel3Name.replace(pos, 4, "vel3");
            qv_vel2 = DBGetQuadvar(blockDbfile, vel2Name.c_str());
            qv_vel3 = DBGetQuadvar(blockDbfile, vel3Name.c_str());
          }
        }

        if (!qv_vel1 || !qv_vel2 || !qv_vel3 || !qm) {
          logWarning(
              "[import_SILO] failed to load velocity components for block %zu",
              blockIdx);
          if (qv_vel1)
            DBFreeQuadvar(qv_vel1);
          if (qv_vel2)
            DBFreeQuadvar(qv_vel2);
          if (qv_vel3)
            DBFreeQuadvar(qv_vel3);
          if (qm)
            DBFreeQuadmesh(qm);
          if (blockDbfile != dbfile)
            DBClose(blockDbfile);
          continue;
        }
      } else {
        // Load single variable
        qv_vel1 = DBGetQuadvar(blockDbfile, info.varName.c_str());
        if (qv_vel1 && qv_vel1->meshname) {
          qm = DBGetQuadmesh(blockDbfile, qv_vel1->meshname);
        }
      }

      if (qv_vel1 && qm) {
        // Calculate this block's position in the global grid
        float3 offset = info.origin - globalMin;
        int3 globalOffset((int)std::round(offset.x / globalSpacing.x),
            (int)std::round(offset.y / globalSpacing.y),
            (int)std::round(offset.z / globalSpacing.z));

        // Get real node bounds for this block
        int3 realNodeFirst(0, 0, 0);
        int3 realNodeLast(qm->dims[0] - 1, qm->dims[1] - 1, qm->dims[2] - 1);

        if (qm->ghost_zone_labels && qm->min_index && qm->max_index) {
          for (int j = 0; j < qm->ndims && j < 3; j++) {
            realNodeFirst[j] = qm->min_index[j];
            realNodeLast[j] = qm->max_index[j];
          }
        }

        int3 realZoneFirst = realNodeFirst;
        int3 realZoneLast = int3(std::max(realNodeLast.x - 1, realNodeFirst.x),
            std::max(realNodeLast.y - 1, realNodeFirst.y),
            std::max(realNodeLast.z - 1, realNodeFirst.z));

        int3 blockTotalDims(qm->dims[0] - 1, qm->dims[1] - 1, qm->dims[2] - 1);
        size_t elementSize = anari::sizeOf(globalDataType);

        // Prepare data for derived field computation if needed
        std::vector<float> blockData;
        if (isDerivedField) {
          size_t blockElements = (realZoneLast.x - realZoneFirst.x + 1)
              * (realZoneLast.y - realZoneFirst.y + 1)
              * (realZoneLast.z - realZoneFirst.z + 1);
          blockData.resize(blockElements);

          // Load velocity data for all derived fields
          // Check the actual data size from the quadvar
          std::vector<float> vel1Data;
          std::vector<float> vel2Data;
          std::vector<float> vel3Data;

          // Copy full block data
          auto copyToFloat = [](DBquadvar *qv, std::vector<float> &dest) {
            dest.resize(qv->nels);
            if (qv->datatype == DB_FLOAT) {
              std::copy((float *)qv->vals[0],
                  (float *)qv->vals[0] + qv->nels,
                  dest.begin());
            } else if (qv->datatype == DB_DOUBLE) {
              double *src = (double *)qv->vals[0];
              for (size_t i = 0; i < qv->nels; ++i) {
                dest[i] = static_cast<float>(src[i]);
              }
            }
          };

          copyToFloat(qv_vel1, vel1Data);
          copyToFloat(qv_vel2, vel2Data);
          copyToFloat(qv_vel3, vel3Data);

          logStatus(
              "[import_SILO] block %zu:"
              " loaded velocity data, size=%zu, expected=%d",
              blockIdx,
              size_t(qv_vel1->nels),
              blockTotalDims.x * blockTotalDims.y * blockTotalDims.z);

          if (derivedFieldType == "vel_mag") {
            // Compute velocity magnitude for real zones only
            size_t idx = 0;
            for (int k = realZoneFirst.z; k <= realZoneLast.z; k++) {
              for (int j = realZoneFirst.y; j <= realZoneLast.y; j++) {
                for (int i = realZoneFirst.x; i <= realZoneLast.x; i++) {
                  int srcIdx = k * blockTotalDims.y * blockTotalDims.x
                      + j * blockTotalDims.x + i;
                  float v1 = vel1Data[srcIdx];
                  float v2 = vel2Data[srcIdx];
                  float v3 = vel3Data[srcIdx];
                  blockData[idx++] = std::sqrt(v1 * v1 + v2 * v2 + v3 * v3);
                }
              }
            }
          } else if (derivedFieldType == "lambda2" || derivedFieldType == "w"
              || derivedFieldType == "qcrit" || derivedFieldType == "hel") {
            // These derived fields all require velocity gradients

            // Verify data size matches expectations
            size_t expectedSize =
                blockTotalDims.x * blockTotalDims.y * blockTotalDims.z;
            size_t actualDataSize = vel1Data.size();
            if (actualDataSize != expectedSize) {
              logWarning(
                  "[import_SILO] block %zu: data size mismatch %zu != %zu",
                  blockIdx,
                  actualDataSize,
                  expectedSize);
              // Skip this block
              DBFreeQuadmesh(qm);
              if (qv_vel1)
                DBFreeQuadvar(qv_vel1);
              if (qv_vel2)
                DBFreeQuadvar(qv_vel2);
              if (qv_vel3)
                DBFreeQuadvar(qv_vel3);
              if (blockDbfile != dbfile)
                DBClose(blockDbfile);
              continue;
            }

            // Compute velocity gradients
            std::vector<float> dux(vel1Data.size()), duy(vel1Data.size()),
                duz(vel1Data.size());
            std::vector<float> dvx(vel1Data.size()), dvy(vel1Data.size()),
                dvz(vel1Data.size());
            std::vector<float> dwx(vel1Data.size()), dwy(vel1Data.size()),
                dwz(vel1Data.size());

            // Extract Z coordinates for non-uniform spacing
            std::vector<float> zCoords(qm->dims[2]);
            for (int zi = 0; zi < qm->dims[2]; ++zi) {
              if (qm->datatype == DB_FLOAT)
                zCoords[zi] = ((float *)qm->coords[2])[zi];
              else if (qm->datatype == DB_DOUBLE)
                zCoords[zi] = ((double *)qm->coords[2])[zi];
            }

            computeVelocityGradients(vel1Data,
                vel2Data,
                vel3Data,
                blockTotalDims.x,
                blockTotalDims.y,
                blockTotalDims.z,
                globalSpacing.x,
                globalSpacing.y,
                zCoords,
                dux,
                duy,
                duz,
                dvx,
                dvy,
                dvz,
                dwx,
                dwy,
                dwz);

            // Debug: check gradient magnitudes
            float maxGrad = 0.0f;
            for (size_t ii = 0; ii < dux.size(); ++ii) {
              maxGrad = std::max(maxGrad, std::abs(dux[ii]));
              maxGrad = std::max(maxGrad, std::abs(duy[ii]));
              maxGrad = std::max(maxGrad, std::abs(duz[ii]));
            }
            logStatus(
                "[import_SILO] block %zu: "
                "max velocity gradient = %.6e, spacing=(%.3e,%.3e)",
                blockIdx,
                maxGrad,
                globalSpacing.x,
                globalSpacing.y);

            // Compute the requested derived field
            std::vector<float> derivedFull(vel1Data.size());

            if (derivedFieldType == "lambda2") {
              computeLambda2(
                  dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz, derivedFull);
            } else if (derivedFieldType == "w") {
              computeVorticity(
                  dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz, derivedFull);
            } else if (derivedFieldType == "qcrit") {
              computeQCriterion(
                  dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz, derivedFull);
            } else if (derivedFieldType == "hel") {
              computeHelicity(vel1Data,
                  vel2Data,
                  vel3Data,
                  dux,
                  duy,
                  duz,
                  dvx,
                  dvy,
                  dvz,
                  dwx,
                  dwy,
                  dwz,
                  derivedFull);
            }

            // Debug: check for NaN or constant values
            float minVal = derivedFull[0], maxVal = derivedFull[0];
            int nanCount = 0;
            for (size_t ii = 0; ii < derivedFull.size(); ++ii) {
              if (std::isnan(derivedFull[ii]) || std::isinf(derivedFull[ii])) {
                nanCount++;
                derivedFull[ii] = 0.0f; // Replace NaN/Inf with 0
              } else {
                minVal = std::min(minVal, derivedFull[ii]);
                maxVal = std::max(maxVal, derivedFull[ii]);
              }
            }
            if (nanCount > 0) {
              logWarning("[import_SILO] block %zu: %s had %d NaN/Inf values",
                  blockIdx,
                  derivedFieldType.c_str(),
                  nanCount);
            }
            logStatus(
                "[import_SILO] block %zu:"
                " %s computed, range in full block: %.6e to %.6e",
                blockIdx,
                derivedFieldType.c_str(),
                minVal,
                maxVal);

            // Extract only real zones
            size_t idx = 0;
            for (int k = realZoneFirst.z; k <= realZoneLast.z; k++) {
              for (int j = realZoneFirst.y; j <= realZoneLast.y; j++) {
                for (int i = realZoneFirst.x; i <= realZoneLast.x; i++) {
                  int srcIdx = k * blockTotalDims.y * blockTotalDims.x
                      + j * blockTotalDims.x + i;
                  blockData[idx++] = derivedFull[srcIdx];
                }
              }
            }
          }
        }

        // Copy data to unified grid
        size_t dstIdx = 0;
        for (int k = realZoneFirst.z; k <= realZoneLast.z; k++) {
          for (int j = realZoneFirst.y; j <= realZoneLast.y; j++) {
            for (int i = realZoneFirst.x; i <= realZoneLast.x; i++) {
              // Destination index in global array
              int gi = globalOffset.x + (i - realZoneFirst.x);
              int gj = globalOffset.y + (j - realZoneFirst.y);
              int gk = globalOffset.z + (k - realZoneFirst.z);
              int globalIdx =
                  gk * globalDims.y * globalDims.x + gj * globalDims.x + gi;

              if (isDerivedField) {
                // Copy from computed derived field
                ((float *)unifiedData)[globalIdx] = blockData[dstIdx];
              } else {
                // Copy directly from source variable
                int srcIdx = k * blockTotalDims.y * blockTotalDims.x
                    + j * blockTotalDims.x + i;
                std::memcpy((char *)unifiedData + globalIdx * elementSize,
                    (char *)qv_vel1->vals[0] + srcIdx * elementSize,
                    elementSize);
              }
              dstIdx++;
            }
          }
        }

        logStatus(
            "[import_SILO] block %d: "
            "%s offset=(%d,%d,%d) dims=%dx%dx%d",
            (int)blockIdx,
            info.varName.c_str(),
            globalOffset.x,
            globalOffset.y,
            globalOffset.z,
            info.dims.x,
            info.dims.y,
            info.dims.z);

        DBFreeQuadmesh(qm);

        // Free velocity variables
        if (qv_vel1)
          DBFreeQuadvar(qv_vel1);
        if (qv_vel2)
          DBFreeQuadvar(qv_vel2);
        if (qv_vel3)
          DBFreeQuadvar(qv_vel3);
      }
    }

    // Close block file if it's different
    if (blockDbfile != dbfile)
      DBClose(blockDbfile);
  }

  unifiedDataArray->unmap();

  // Free multivars
  if (isDerivedField) {
    if (mv_vel1)
      DBFreeMultivar(mv_vel1);
    if (mv_vel2)
      DBFreeMultivar(mv_vel2);
    if (mv_vel3)
      DBFreeMultivar(mv_vel3);
  } else {
    DBFreeMultivar(mv);
  }

  // Create unified spatial field
  auto field = scene.createObject<SpatialField>(
      tokens::spatial_field::structuredRegular);
  std::string fieldName =
      isDerivedField ? derivedFieldType : std::string(multivarName);
  field->setName(fieldName.c_str());
  field->setParameter("origin", globalMin);
  field->setParameter("spacing", globalSpacing);
  field->setParameter("dataCentering", isNodeCentered ? "node" : "cell");
  field->setParameterObject("data", *unifiedDataArray);

  // Create single volume with unified field
  logStatus("[import_SILO] creating unified volume");

  return field;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Main import function
SpatialFieldRef import_SILO(Scene &scene, const char *filepath)
{
  // Parse variable name if present (format: file.silo or file.silo:varname)
  std::string file(filepath);
  std::string varName;
  size_t colonPos = file.find(':');
  if (colonPos != std::string::npos) {
    varName = file.substr(colonPos + 1);
    file = file.substr(0, colonPos);
  }

  // Check if this is a derived field request
  bool isDerivedField = false;
  std::string derivedFieldType;
  if (varName == "vel_mag" || varName == "lambda2" || varName == "hel"
      || varName == "w" || varName == "qcrit") {
    isDerivedField = true;
    derivedFieldType = varName;
    logStatus(
        "[import_SILO] derived field requested: %s", derivedFieldType.c_str());
    // Clear varName so we can handle the base velocity variables
    varName.clear();
  }

  // Open Silo file
  DBfile *dbfile = DBOpen(file.c_str(), DB_UNKNOWN, DB_READ);
  if (!dbfile) {
    logError("[import_SILO] failed to open file '%s'", file.c_str());
    return {};
  }

  // Get table of contents
  DBtoc *toc = DBGetToc(dbfile);
  if (!toc) {
    logError("[import_SILO] failed to get table of contents");
    DBClose(dbfile);
    return {};
  }

  logStatus("[import_SILO] loading '%s'", file.c_str());

  SpatialFieldRef field;
  if (toc->nmultimesh > 0) {
    logStatus("[import_SILO] found %d meshes", toc->nmultimesh);
    field = import_SILO_multiMesh(
        scene, file, dbfile, toc, varName, isDerivedField, derivedFieldType);
  } else {
    logStatus("[import_SILO] importing single mesh");
    field = import_SILO_singleMesh(scene, dbfile, toc, varName);
  }

  DBClose(dbfile);

  logStatus("[import_SILO] done!");
  return field;
}

void import_SILO(Scene &scene, const char *filename, LayerNodeRef location)
{
  SpatialFieldRef field = import_SILO(scene, filename);
  if (field) {
    // Create one transform node for the unified volume
    auto tx = scene.insertChildTransformNode(location);

    // Create shared color map
    auto colorArray = scene.createArray(ANARI_FLOAT32_VEC4, 256);
    colorArray->setData(makeDefaultColorMap(colorArray->size()).data());

    auto valueRange = field->computeValueRange();

    auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
        tx, tokens::volume::transferFunction1D);
    volume->setName(fileOf(filename).c_str());
    volume->setParameterObject("value", *field);
    volume->setParameterObject("color", *colorArray);
    volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
  }
}

#else

SpatialFieldRef import_SILO(Scene &scene, const char *filepath)
{
  logError("[import_SILO] Silo support not enabled in this build");

  return {};
}

void import_SILO(Scene &scene, const char *filename, LayerNodeRef location)
{
  logError("[import_SILO] Silo support not enabled in this build");
}

#endif

} // namespace tsd::io

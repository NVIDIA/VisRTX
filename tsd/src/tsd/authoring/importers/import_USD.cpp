// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_USD
#define TSD_USE_USD 1
#endif

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/TSDMath.hpp"
#if TSD_USE_USD
// usd
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCache.h>
#endif
// std
#include <string>
#include <vector>
#include <limits>

namespace tsd {

#if TSD_USE_USD

// Helper: Convert pxr::GfMatrix4d to tsd::mat4 (float4x4)
inline tsd::mat4 to_tsd_mat4(const pxr::GfMatrix4d &m)
{
  tsd::mat4 out;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      out[i][j] = static_cast<float>(m[i][j]);
  return out;
}

// Add these helpers at file or namespace scope if not already present:
inline float3 min(const float3 &a, const float3 &b) {
  return float3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}
inline float3 max(const float3 &a, const float3 &b) {
  return float3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

// Helper: Import a UsdGeomMesh prim as a TSD mesh under the given parent node
static void import_usd_mesh(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent, const pxr::GfMatrix4d &usdXform, float3 &sceneMin, float3 &sceneMax)
{
  pxr::UsdGeomMesh mesh(prim);
  pxr::VtArray<pxr::GfVec3f> points;
  mesh.GetPointsAttr().Get(&points);

  // Update scene bounding box with transformed points
  for (size_t i = 0; i < points.size(); ++i) {
    pxr::GfVec3f p = points[i];
    pxr::GfVec4d p4(p[0], p[1], p[2], 1.0);
    pxr::GfVec4d wp4 = usdXform * p4;
    float3 wp{float(wp4[0]), float(wp4[1]), float(wp4[2])};
    sceneMin = tsd::min(sceneMin, wp);
    sceneMax = tsd::max(sceneMax, wp);
  }

  pxr::VtArray<int> faceVertexIndices;
  mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);
  pxr::VtArray<int> faceVertexCounts;
  mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts);
  pxr::VtArray<pxr::GfVec3f> normals;
  mesh.GetNormalsAttr().Get(&normals);

  logInfo("[import_USD] Mesh '%s': %zu points, %zu faces, %zu normals",
          prim.GetName().GetString().c_str(),
          points.size(),
          faceVertexCounts.size(),
          normals.size());
  printf("[import_USD] Mesh '%s': %zu points, %zu faces, %zu normals\n",
          prim.GetName().GetString().c_str(),
          points.size(),
          faceVertexCounts.size(),
          normals.size());

  std::vector<float3> outVertices;
  std::vector<float3> outNormals;
  size_t index = 0;
  for (size_t face = 0; face < faceVertexCounts.size(); ++face) {
    int vertsInFace = faceVertexCounts[face];
    for (int v = 2; v < vertsInFace; ++v) {
      int idx0 = faceVertexIndices[index];
      int idx1 = faceVertexIndices[index + v - 1];
      int idx2 = faceVertexIndices[index + v];
      outVertices.push_back(float3(points[idx0][0], points[idx0][1], points[idx0][2]));
      outVertices.push_back(float3(points[idx1][0], points[idx1][1], points[idx1][2]));
      outVertices.push_back(float3(points[idx2][0], points[idx2][1], points[idx2][2]));
      if (normals.size() == points.size()) {
        outNormals.push_back(float3(normals[idx0][0], normals[idx0][1], normals[idx0][2]));
        outNormals.push_back(float3(normals[idx1][0], normals[idx1][1], normals[idx1][2]));
        outNormals.push_back(float3(normals[idx2][0], normals[idx2][1], normals[idx2][2]));
      }
    }
    index += vertsInFace;
  }

  auto meshObj = ctx.createObject<Geometry>(tokens::geometry::triangle);
  auto vertexPositionArray = ctx.createArray(ANARI_FLOAT32_VEC3, outVertices.size());
  vertexPositionArray->setData(outVertices.data(), outVertices.size());
  meshObj->setParameterObject("vertex.position", *vertexPositionArray);
  if (!outNormals.empty()) {
    auto normalsArray = ctx.createArray(ANARI_FLOAT32_VEC3, outNormals.size());
    normalsArray->setData(outNormals.data(), outNormals.size());
    meshObj->setParameterObject("vertex.normal", *normalsArray);
  }
  std::string primName = prim.GetName().GetString();
  if (primName.empty()) primName = "<unnamed_mesh>";
  meshObj->setName(primName.c_str());
  auto mat = ctx.defaultMaterial();
  auto surface = ctx.createSurface(primName.c_str(), meshObj, mat);
  ctx.insertChildObjectNode(parent, surface);
}

// Recursive import function for prims and their children
static void import_usd_prim_recursive(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent, pxr::UsdGeomXformCache &xformCache, float3 &sceneMin, float3 &sceneMax)
{
  // Get the world transform for this prim
  pxr::GfMatrix4d usdXform = xformCache.GetLocalToWorldTransform(prim);
  tsd::mat4 tsdXform = to_tsd_mat4(usdXform);
  std::string primName = prim.GetName().GetString();
  if (primName.empty()) primName = "<unnamed_xform>";
  auto xformNode = ctx.insertChildTransformNode(parent, tsdXform, primName.c_str());

  if (prim.IsA<pxr::UsdGeomMesh>()) {
    import_usd_mesh(ctx, prim, xformNode, usdXform, sceneMin, sceneMax);
  }

  // Recurse into children
  for (const auto &child : prim.GetChildren()) {
    import_usd_prim_recursive(ctx, child, xformNode, xformCache, sceneMin, sceneMax);
  }
}

void import_USD(Context &ctx,
    const char *filepath,
    LayerNodeRef location,
    bool useDefaultMaterial)
{
  pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filepath);
  if (!stage) {
    logError("[import_USD] failed to open stage '%s'", filepath);
    return;
  }
  logInfo("[import_USD] Opened USD stage: %s", filepath);
  printf("[import_USD] Opened USD stage: %s\n", filepath);
  auto defaultPrim = stage->GetDefaultPrim();
  if (defaultPrim) {
    logInfo("[import_USD] Default prim: %s", defaultPrim.GetPath().GetString().c_str());
    printf("[import_USD] Default prim: %s\n", defaultPrim.GetPath().GetString().c_str());
  } else {
    logInfo("[import_USD] No default prim set.");
    printf("[import_USD] No default prim set.\n");
  }
  size_t primCount = 0;
  for (auto it = stage->Traverse().begin(); it != stage->Traverse().end(); ++it)
    ++primCount;
  logInfo("[import_USD] Number of prims in stage: %zu", primCount);
  printf("[import_USD] Number of prims in stage: %zu\n", primCount);
  float3 sceneMin(std::numeric_limits<float>::max());
  float3 sceneMax(std::numeric_limits<float>::lowest());
  auto usd_root = ctx.insertChildNode(
      location ? location : ctx.defaultLayer()->root(), filepath);
  pxr::UsdGeomXformCache xformCache(pxr::UsdTimeCode::Default());

  // Traverse all prims in the USD file
  for (pxr::UsdPrim const &prim : stage->Traverse()) {
    if (prim.IsInstance()) {
      pxr::UsdPrim prototype = prim.GetPrototype();
      if (prototype) {
        pxr::GfMatrix4d usdXform = xformCache.GetLocalToWorldTransform(prim);
        tsd::mat4 tsdXform = to_tsd_mat4(usdXform);
        std::string primName = prim.GetName().GetString();
        if (primName.empty()) primName = "<unnamed_instance>";
        auto xformNode = ctx.insertChildTransformNode(usd_root, tsdXform, primName.c_str());
        // Recursively import the prototype under this transform node
        import_usd_prim_recursive(ctx, prototype, xformNode, xformCache, sceneMin, sceneMax);

      } else {
        logWarning("[import_USD] Instance has no prototype: %s", prim.GetName().GetString().c_str());
        printf("[import_USD] Instance has no prototype: %s\n", prim.GetName().GetString().c_str());
      }
      continue;
    }
    if (prim.IsInstanceable()) {
      logInfo("[import_USD] Prim is instanceable: %s (type: %s, path: %s)",
              prim.GetName().GetString().c_str(),
              prim.GetTypeName().GetString().c_str(),
              prim.GetPath().GetString().c_str());
      printf("[import_USD] Prim is instanceable: %s (type: %s, path: %s)",
              prim.GetName().GetString().c_str(),
              prim.GetTypeName().GetString().c_str(),
              prim.GetPath().GetString().c_str());
    }
    // Only import top-level prims (not children, which will be handled recursively)
    if (prim.GetParent() && prim.GetParent().IsPseudoRoot()) {
      import_usd_prim_recursive(ctx, prim, usd_root, xformCache, sceneMin, sceneMax);
    }
  }
  logInfo("[import_USD] Scene bounds: min=(%f, %f, %f) max=(%f, %f, %f)",
          sceneMin.x, sceneMin.y, sceneMin.z,
          sceneMax.x, sceneMax.y, sceneMax.z);
  printf("[import_USD] Scene bounds: min=(%f, %f, %f) max=(%f, %f, %f)\n",
          sceneMin.x, sceneMin.y, sceneMin.z,
          sceneMax.x, sceneMax.y, sceneMax.z);
}
#else
void import_USD(Context &ctx,
    const char *filepath,
    LayerNodeRef location,
    bool useDefaultMaterial)
{
  logError("[import_USD] USD not enabled in TSD build.");
}
#endif

} // namespace tsd

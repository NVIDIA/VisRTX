// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_USD
#define TSD_USE_USD 1
#endif

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
#if TSD_USE_USD
// usd
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#endif
// std
#include <string>
#include <vector>

namespace tsd {

#if TSD_USE_USD

void import_USD(Context &ctx,
    const char *filepath,
    LayerNodeRef location,
    bool useDefaultMaterial)
{
  // Open the USD stage
  pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filepath);
  if (!stage) {
    // Handle error
    logError("[import_USD] failed to open stage '%s'", filepath);
    return;
  }

  // Create a root node in your scene
  auto usd_root = ctx.insertChildNode(
      location ? location : ctx.defaultLayer()->root(), filepath);

  // Traverse all prims in the USD file
  for (pxr::UsdPrim const &prim : stage->Traverse()) {
    if (!prim.IsA<pxr::UsdGeomMesh>()) {
      logWarning("[import_USD] skipping prim (not UsdGeomMesh)");
      continue;
    }

    pxr::UsdGeomMesh mesh(prim);

    // Get points (vertices)
    pxr::VtArray<pxr::GfVec3f> points;
    mesh.GetPointsAttr().Get(&points);

    // Get face vertex indices
    pxr::VtArray<int> faceVertexIndices;
    mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);

    // Get face vertex counts (number of vertices per face)
    pxr::VtArray<int> faceVertexCounts;
    mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts);

    // Optionally get normals
    pxr::VtArray<pxr::GfVec3f> normals;
    mesh.GetNormalsAttr().Get(&normals);

    // TODO: Get texcoords if present

    // Triangulate faces (fan method)
    std::vector<float3> outVertices;
    std::vector<float3> outNormals;
    // TODO: Add texcoord extraction if needed

    size_t index = 0;
    for (size_t face = 0; face < faceVertexCounts.size(); ++face) {
      int vertsInFace = faceVertexCounts[face];
      // Triangulate n-gons (fan method)
      for (int v = 2; v < vertsInFace; ++v) {
        int idx0 = faceVertexIndices[index];
        int idx1 = faceVertexIndices[index + v - 1];
        int idx2 = faceVertexIndices[index + v];
        outVertices.push_back(
            float3(points[idx0][0], points[idx0][1], points[idx0][2]));
        outVertices.push_back(
            float3(points[idx1][0], points[idx1][1], points[idx1][2]));
        outVertices.push_back(
            float3(points[idx2][0], points[idx2][1], points[idx2][2]));
        // Normals (if present and per-vertex)
        if (normals.size() == points.size()) {
          outNormals.push_back(
              float3(normals[idx0][0], normals[idx0][1], normals[idx0][2]));
          outNormals.push_back(
              float3(normals[idx1][0], normals[idx1][1], normals[idx1][2]));
          outNormals.push_back(
              float3(normals[idx2][0], normals[idx2][1], normals[idx2][2]));
        }
      }
      index += vertsInFace;
    }

    // Create mesh object in your engine
    auto meshObj = ctx.createObject<Geometry>(tokens::geometry::triangle);
    auto vertexPositionArray =
        ctx.createArray(ANARI_FLOAT32_VEC3, outVertices.size());
    auto *outVerts = vertexPositionArray->mapAs<float3>();
    for (size_t i = 0; i < outVertices.size(); ++i)
      outVerts[i] = outVertices[i];
    vertexPositionArray->unmap();
    meshObj->setParameterObject("vertex.position", *vertexPositionArray);

    if (!outNormals.empty()) {
      auto normalsArray =
          ctx.createArray(ANARI_FLOAT32_VEC3, outNormals.size());
      auto *outNorms = normalsArray->mapAs<float3>();
      for (size_t i = 0; i < outNormals.size(); ++i)
        outNorms[i] = outNormals[i];
      normalsArray->unmap();
      meshObj->setParameterObject("vertex.normal", *normalsArray);
    }

    // TODO: Handle texcoords, colors, and materials if present

    // Set mesh name
    std::string name = prim.GetName().GetString();
    if (name.empty())
      name = "<unnamed_mesh>";
    meshObj->setName(name.c_str());

    // Use default material for now
    auto mat = ctx.defaultMaterial();
    auto surface = ctx.createSurface(name.c_str(), meshObj, mat);
    ctx.insertChildObjectNode(usd_root, surface);
  }
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

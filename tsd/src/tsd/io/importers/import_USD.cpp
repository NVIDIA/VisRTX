// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <anari/anari_cpp/ext/linalg.h>
#ifndef TSD_USE_USD
#define TSD_USE_USD 1
#endif

#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/scene/objects/Array.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/HDRImage.h"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/io/importers/detail/usd/OmniPbrMaterial.h"
#if TSD_USE_USD
// usd
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/basisCurves.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/points.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdShade/input.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/output.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdVol/volume.h>
#endif
// std
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace tsd::io {

using namespace tsd::core;

#if TSD_USE_USD

// -----------------------------------------------------------------------------
// Material-related helpers
// -----------------------------------------------------------------------------

// Template helpers for setting material parameters from USD shader inputs
static void setShaderInputIfPresent(MaterialRef &mat,
    pxr::UsdShadeShader &shader,
    const char *inputName,
    const char *paramName)
{
  pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(inputName));
  pxr::GfVec3f colorVal;
  if (input && input.Get(&colorVal)) {
    logStatus("[import_USD] Setting %s: %f %f %f\n",
        paramName,
        colorVal[0],
        colorVal[1],
        colorVal[2]);
    mat->setParameter(tsd::core::Token(paramName),
        tsd::math::float3(colorVal[0], colorVal[1], colorVal[2]));
  }
}

static void setShaderInputIfPresent(MaterialRef &mat,
    pxr::UsdShadeShader &shader,
    const char *inputName,
    const char *paramName,
    float)
{
  pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(inputName));
  float floatVal;
  if (input && input.Get(&floatVal)) {
    logStatus("[import_USD] Setting %s: %f\n", paramName, floatVal);
    mat->setParameter(tsd::core::Token(paramName), floatVal);
  }
}

// Helper: Import a UsdPreviewSurface material as a physicallyBased TSD material
static MaterialRef importUsdPreviewSurfaceMaterial(Scene &scene,
    const pxr::UsdShadeMaterial &usdMat,
    const std::string &basePath)
{
  // Find the UsdPreviewSurface shader
  pxr::UsdShadeShader surfaceShader;
  pxr::TfToken outputName("surface");
  pxr::UsdShadeOutput surfaceOutput = usdMat.GetOutput(outputName);

  if (surfaceOutput && surfaceOutput.HasConnectedSource()) {
    logStatus("[import_USD] Surface output has connected source\n");
    pxr::UsdShadeConnectableAPI source;
    pxr::TfToken sourceName;
    pxr::UsdShadeAttributeType sourceType;
    surfaceOutput.GetConnectedSource(&source, &sourceName, &sourceType);
    surfaceShader = pxr::UsdShadeShader(source.GetPrim());
  }

  if (!surfaceShader)
    return scene.defaultMaterial();

  auto mat = scene.createObject<Material>(tokens::material::physicallyBased);

  setShaderInputIfPresent(mat, surfaceShader, "diffuseColor", "baseColor");
  setShaderInputIfPresent(mat, surfaceShader, "emissiveColor", "emissive");
  setShaderInputIfPresent(mat, surfaceShader, "metallic", "metallic", 0.0f);
  setShaderInputIfPresent(mat, surfaceShader, "roughness", "roughness", 0.0f);
  setShaderInputIfPresent(mat, surfaceShader, "clearcoat", "clearcoat", 0.0f);
  setShaderInputIfPresent(
      mat, surfaceShader, "clearcoatRoughness", "clearcoatRoughness", 0.0f);
  setShaderInputIfPresent(mat, surfaceShader, "opacity", "opacity", 0.0f);
  setShaderInputIfPresent(mat, surfaceShader, "ior", "ior", 0.0f);

  // Set name
  std::string matName = usdMat.GetPrim().GetPath().GetString();
  if (matName.empty())
    matName = "USDPreviewSurface";
  mat->setName(matName.c_str());
  logStatus("[import_USD] Created material: %s\n", matName.c_str());

  return mat;
}

// Caches material refs by USD prim path to avoid duplicate imports
using MaterialCache = std::unordered_map<std::string, MaterialRef>;

// Try to import the bound material for a prim.  Checks the cache first, then
// tries OmniPBR (MDL), then UsdPreviewSurface, then falls back to default.
static MaterialRef getBoundMaterial(Scene &scene,
    const pxr::UsdPrim &prim,
    const std::string &basePath,
    MaterialCache &matCache,
    TextureCache &texCache)
{
  pxr::UsdShadeMaterialBindingAPI binding(prim);
  pxr::UsdShadeMaterial usdMat = binding.ComputeBoundMaterial();
  if (!usdMat)
    return scene.defaultMaterial();

  std::string matPath = usdMat.GetPath().GetString();
  auto it = matCache.find(matPath);
  if (it != matCache.end())
    return it->second;

  MaterialRef mat;

  // Try OmniPBR via MDL surface output
  auto mdlOutput = usdMat.GetSurfaceOutput(pxr::TfToken("mdl"));
  for (auto &src : mdlOutput.GetConnectedSources()) {
    pxr::UsdShadeShader shader(src.source);
    pxr::TfToken subId;
    shader.GetSourceAssetSubIdentifier(&subId, pxr::TfToken("mdl"));
    if (subId == pxr::TfToken("OmniPBR")) {
      mat = materials::importOmniPBRMaterial(
          scene, usdMat, shader, basePath, texCache);
      break;
    }
  }

  // Fall back to UsdPreviewSurface
  if (!mat)
    mat = importUsdPreviewSurfaceMaterial(scene, usdMat, basePath);

  if (!mat)
    mat = scene.defaultMaterial();

  matCache[matPath] = mat;
  return mat;
}


// Helper to extract volume transfer function from USD material
struct VolumeTransferFunction
{
  std::vector<math::float4> colors;
  std::vector<float> xPoints;
  math::float2 domain{0.0f, 1.0f};
  bool hasTransferFunction = false;
};

static VolumeTransferFunction getVolumeTransferFunction(
    const pxr::UsdPrim &prim)
{
  VolumeTransferFunction tf;

  // Check if MaterialBindingAPI can be applied to this prim type
  if (!pxr::UsdShadeMaterialBindingAPI::CanApply(prim)) {
    return tf;
  }

  // Try to get material binding
  pxr::UsdShadeMaterialBindingAPI binding(prim);
  pxr::UsdShadeMaterial usdMat;

  // First, try to get the direct material binding relationship
  pxr::UsdRelationship materialRel =
      prim.GetRelationship(pxr::TfToken("material:binding"));
  if (materialRel) {
    pxr::SdfPathVector targets;
    materialRel.GetTargets(&targets);
    if (!targets.empty()) {
      // Try to get the material prim directly
      pxr::UsdPrim materialPrim = prim.GetStage()->GetPrimAtPath(targets[0]);
      if (materialPrim) {
        usdMat = pxr::UsdShadeMaterial(materialPrim);
      }
    }
  }

  // If direct resolution didn't work, try ComputeBoundMaterial
  if (!usdMat && binding) {
    usdMat = binding.ComputeBoundMaterial();
  }

  if (!usdMat) {
    return tf;
  }

  // Look for volume output connection
  pxr::TfToken volumeOutputName("nvindex:volume");
  pxr::UsdShadeOutput volumeOutput = usdMat.GetOutput(volumeOutputName);

  if (!volumeOutput || !volumeOutput.HasConnectedSource()) {
    return tf;
  }

  // Get the VolumeShader
  pxr::UsdShadeConnectableAPI volumeSource;
  pxr::TfToken volumeSourceName;
  pxr::UsdShadeAttributeType volumeSourceType;
  volumeOutput.GetConnectedSource(
      &volumeSource, &volumeSourceName, &volumeSourceType);
  pxr::UsdShadeShader volumeShader(volumeSource.GetPrim());

  if (!volumeShader) {
    return tf;
  }

  // Look for colormap input connection
  pxr::UsdShadeInput colormapInput =
      volumeShader.GetInput(pxr::TfToken("colormap"));
  if (!colormapInput || !colormapInput.HasConnectedSource()) {
    return tf;
  }

  // Get the Colormap shader
  pxr::UsdShadeConnectableAPI colormapSource;
  pxr::TfToken colormapSourceName;
  pxr::UsdShadeAttributeType colormapSourceType;
  bool hasConnection = colormapInput.GetConnectedSource(
      &colormapSource, &colormapSourceName, &colormapSourceType);

  if (!hasConnection) {
    return tf;
  }

  pxr::UsdPrim colormapPrim = colormapSource.GetPrim();
  if (!colormapPrim) {
    return tf;
  }

  pxr::UsdShadeShader colormapShader(colormapPrim);

  if (!colormapShader) {
    // Try to extract data directly from the prim even if it's not a valid
    // UsdShadeShader
    pxr::UsdAttribute rgbaPointsAttr =
        colormapPrim.GetAttribute(pxr::TfToken("rgbaPoints"));
    pxr::UsdAttribute xPointsAttr =
        colormapPrim.GetAttribute(pxr::TfToken("xPoints"));
    pxr::UsdAttribute domainAttr =
        colormapPrim.GetAttribute(pxr::TfToken("domain"));

    if (rgbaPointsAttr && xPointsAttr) {
      // Extract the data using the same logic as below
      pxr::VtArray<pxr::GfVec4f> rgbaPoints;
      pxr::VtArray<float> xPoints;

      if (rgbaPointsAttr.Get(&rgbaPoints) && xPointsAttr.Get(&xPoints)) {
        // Convert to TSD format
        tf.colors.resize(rgbaPoints.size());
        tf.xPoints.resize(xPoints.size());

        for (size_t i = 0; i < rgbaPoints.size(); ++i) {
          const auto &rgba = rgbaPoints[i];
          tf.colors[i] = math::float4(rgba[0], rgba[1], rgba[2], rgba[3]);
        }

        for (size_t i = 0; i < xPoints.size(); ++i) {
          tf.xPoints[i] = xPoints[i];
        }

        // Get domain if present
        if (domainAttr) {
          pxr::GfVec2f domain;
          if (domainAttr.Get(&domain)) {
            tf.domain = math::float2(domain[0], domain[1]);
          }
        }

        tf.hasTransferFunction = true;
      }
    }

    return tf;
  }

  // Extract transfer function data from colormap shader
  pxr::UsdAttribute rgbaPointsAttr =
      colormapShader.GetPrim().GetAttribute(pxr::TfToken("rgbaPoints"));
  pxr::UsdAttribute xPointsAttr =
      colormapShader.GetPrim().GetAttribute(pxr::TfToken("xPoints"));
  pxr::UsdAttribute domainAttr =
      colormapShader.GetPrim().GetAttribute(pxr::TfToken("domain"));

  if (rgbaPointsAttr && xPointsAttr) {
    pxr::VtArray<pxr::GfVec4f> rgbaPoints;
    pxr::VtArray<float> xPoints;

    if (rgbaPointsAttr.Get(&rgbaPoints) && xPointsAttr.Get(&xPoints)) {
      // Convert to TSD format
      tf.colors.resize(rgbaPoints.size());
      tf.xPoints.resize(xPoints.size());

      for (size_t i = 0; i < rgbaPoints.size(); ++i) {
        const auto &rgba = rgbaPoints[i];
        tf.colors[i] = math::float4(rgba[0], rgba[1], rgba[2], rgba[3]);
      }

      for (size_t i = 0; i < xPoints.size(); ++i) {
        tf.xPoints[i] = xPoints[i];
      }

      // Get domain if present
      if (domainAttr) {
        pxr::GfVec2f domain;
        if (domainAttr.Get(&domain)) {
          tf.domain = math::float2(domain[0], domain[1]);
        }
      }

      tf.hasTransferFunction = true;

      logStatus(
          "[import_USD] Found volume transfer function with %zu colors and %zu x-points, domain: [%f, %f]\n",
          tf.colors.size(),
          tf.xPoints.size(),
          tf.domain.x,
          tf.domain.y);
    }
  }

  return tf;
}

// -----------------------------------------------------------------------------
// Geometry import helpers
// -----------------------------------------------------------------------------

// Helper: Convert pxr::GfMatrix4d to tsd::math::mat4 (float4x4)
inline tsd::math::mat4 toTsdMat4(const pxr::GfMatrix4d &m)
{
  tsd::math::mat4 out;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      out[i][j] = static_cast<float>(m[i][j]);
  return out;
}

inline float3 min(const float3 &a, const float3 &b)
{
  return float3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

inline float3 max(const float3 &a, const float3 &b)
{
  return float3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

// Helper: Generate triangle indices from polygon face data
// Tessellates polygons to triangles using triangle fan (assumes convex
// polygons) Returns indices into the original vertex array
static std::vector<uint32_t> generateTriangleIndices(
    const pxr::VtArray<int> &faceVertexIndices,
    const pxr::VtArray<int> &faceVertexCounts)
{
  std::vector<uint32_t> triangleIndices;
  size_t faceVertexOffset = 0;

  for (size_t face = 0; face < faceVertexCounts.size(); ++face) {
    int vertsInFace = faceVertexCounts[face];

    // Tessellate polygon as triangle fan: (0,1,2), (0,2,3), (0,3,4), ...
    for (int v = 2; v < vertsInFace; ++v) {
      triangleIndices.push_back(faceVertexIndices[faceVertexOffset + 0]);
      triangleIndices.push_back(faceVertexIndices[faceVertexOffset + v - 1]);
      triangleIndices.push_back(faceVertexIndices[faceVertexOffset + v]);
    }

    faceVertexOffset += vertsInFace;
  }

  return triangleIndices;
}

// Helper: Tessellate faceVarying data from polygons to triangles
// FaceVarying data has one value per face-vertex (corner)
// Returns tessellated data matching the triangle fan pattern
template <typename T>
static std::vector<T> tessellateFacevaryingData(
    const pxr::VtArray<T> &faceVaryingData,
    const pxr::VtArray<int> &faceVertexCounts)
{
  std::vector<T> triangleData;
  size_t faceVertexOffset = 0;

  for (size_t face = 0; face < faceVertexCounts.size(); ++face) {
    int vertsInFace = faceVertexCounts[face];

    // Tessellate as triangle fan: (0,1,2), (0,2,3), (0,3,4), ...
    for (int v = 2; v < vertsInFace; ++v) {
      triangleData.push_back(faceVaryingData[faceVertexOffset + 0]);
      triangleData.push_back(faceVaryingData[faceVertexOffset + v - 1]);
      triangleData.push_back(faceVaryingData[faceVertexOffset + v]);
    }

    faceVertexOffset += vertsInFace;
  }

  return triangleData;
}

// Helper: Tessellate uniform (per-face) data to per-triangle
// Uniform data has one value per face
// Returns replicated data with one value per generated triangle
template <typename T>
static std::vector<T> tessellateUniformData(
    const pxr::VtArray<T> &uniformData,
    const pxr::VtArray<int> &faceVertexCounts)
{
  std::vector<T> triangleData;

  for (size_t face = 0; face < faceVertexCounts.size(); ++face) {
    int vertsInFace = faceVertexCounts[face];
    int numTriangles = vertsInFace - 2;

    // Each triangle from this face gets the same uniform value
    for (int t = 0; t < numTriangles; ++t) {
      triangleData.push_back(uniformData[face]);
    }
  }

  return triangleData;
}

// Helper: Import a UsdGeomMesh prim as a TSD mesh under the given parent node
static void importUsdMesh(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    const std::string &basePath,
    MaterialCache &matCache,
    TextureCache &texCache)
{
  pxr::UsdGeomMesh mesh(prim);

  // Get vertex positions
  pxr::VtArray<pxr::GfVec3f> points;
  mesh.GetPointsAttr().Get(&points, pxr::UsdTimeCode::EarliestTime());

  // Get face topology
  pxr::VtArray<int> faceVertexIndices;
  mesh.GetFaceVertexIndicesAttr().Get(
      &faceVertexIndices, pxr::UsdTimeCode::EarliestTime());
  pxr::VtArray<int> faceVertexCounts;
  mesh.GetFaceVertexCountsAttr().Get(
      &faceVertexCounts, pxr::UsdTimeCode::EarliestTime());

  // Get normals and their interpolation
  pxr::VtArray<pxr::GfVec3f> normals;
  pxr::TfToken normalsInterpolation = pxr::UsdGeomTokens->vertex; // Default
  mesh.GetNormalsAttr().Get(&normals, pxr::UsdTimeCode::EarliestTime());
  if (!normals.empty()) {
    normalsInterpolation = mesh.GetNormalsInterpolation();
  }

  // Get UVs and their interpolation
  pxr::VtArray<pxr::GfVec2f> uvs;
  pxr::TfToken uvsInterpolation = pxr::UsdGeomTokens->vertex; // Default
  // USD stores UVs as primvars, typically "st" or "UVMap"
  pxr::UsdGeomPrimvarsAPI primvarsAPI(mesh);
  pxr::UsdGeomPrimvar stPrimvar = primvarsAPI.GetPrimvar(pxr::TfToken("st"));
  if (!stPrimvar) {
    stPrimvar = primvarsAPI.GetPrimvar(pxr::TfToken("UVMap"));
  }
  if (stPrimvar) {
    // USD primvars can be indexed - need to use ComputeFlattened to expand
    // indices
    stPrimvar.ComputeFlattened(&uvs);
    if (!uvs.empty()) {
      uvsInterpolation = stPrimvar.GetInterpolation();
    }
  }

  std::string primName = prim.GetPath().GetString();
  if (primName.empty())
    primName = "<unnamed_mesh>";

  logStatus(
      "[import_USD] Mesh '%s': %zu points, %zu faces, %zu normals (interpolation: %s), %zu UVs (interpolation: %s)\n",
      prim.GetName().GetString().c_str(),
      points.size(),
      faceVertexCounts.size(),
      normals.size(),
      normalsInterpolation.GetText(),
      uvs.size(),
      uvsInterpolation.GetText());

  // Convert vertex positions to float3
  std::vector<float3> positions;
  positions.reserve(points.size());
  for (const auto &p : points) {
    positions.push_back(float3(p[0], p[1], p[2]));
  }

  // Generate triangle indices from polygon faces
  std::vector<uint32_t> indices =
      generateTriangleIndices(faceVertexIndices, faceVertexCounts);

  logStatus(
      "[import_USD] Mesh '%s': Generated %zu triangle indices (%zu triangles)\n",
      prim.GetName().GetString().c_str(),
      indices.size(),
      indices.size() / 3);

  // Create ANARI indexed triangle geometry
  auto meshObj = scene.createObject<Geometry>(tokens::geometry::triangle);

  // Set vertex positions
  auto vertexPositionArray =
      scene.createArray(ANARI_FLOAT32_VEC3, positions.size());
  vertexPositionArray->setData(positions.data(), positions.size());
  meshObj->setParameterObject("vertex.position", *vertexPositionArray);

  // Set triangle indices
  auto indexArray = scene.createArray(ANARI_UINT32_VEC3, indices.size() / 3);
  indexArray->setData((uint3 *)indices.data(), indices.size() / 3);
  meshObj->setParameterObject("primitive.index", *indexArray);

  // Handle normals based on USD interpolation
  if (!normals.empty()) {
    if (normalsInterpolation == pxr::UsdGeomTokens->vertex) {
      // Vertex interpolation: normals are per-vertex, shared by all triangles
      // No tessellation needed - just convert to float3
      std::vector<float3> normalData;
      normalData.reserve(normals.size());
      for (const auto &n : normals) {
        normalData.push_back(float3(n[0], n[1], n[2]));
      }

      auto normalsArray =
          scene.createArray(ANARI_FLOAT32_VEC3, normalData.size());
      normalsArray->setData(normalData.data(), normalData.size());
      meshObj->setParameterObject("vertex.normal", *normalsArray);

      logStatus("[import_USD] Mesh '%s': Set %zu normals on vertex.normal\n",
          prim.GetName().GetString().c_str(),
          normalData.size());

    } else if (normalsInterpolation == pxr::UsdGeomTokens->faceVarying) {
      // FaceVarying interpolation: normals are per face-vertex (corner)
      // Need to tessellate from polygon corners to triangle corners
      auto tessellatedNormals =
          tessellateFacevaryingData(normals, faceVertexCounts);

      std::vector<float3> normalData;
      normalData.reserve(tessellatedNormals.size());
      for (const auto &n : tessellatedNormals) {
        normalData.push_back(float3(n[0], n[1], n[2]));
      }

      auto normalsArray =
          scene.createArray(ANARI_FLOAT32_VEC3, normalData.size());
      normalsArray->setData(normalData.data(), normalData.size());
      meshObj->setParameterObject("faceVarying.normal", *normalsArray);

      logStatus(
          "[import_USD] Mesh '%s': Set %zu normals on faceVarying.normal\n",
          prim.GetName().GetString().c_str(),
          normalData.size());

    } else if (normalsInterpolation == pxr::UsdGeomTokens->uniform) {
      // Uniform interpolation: one normal per face
      // Need to replicate for each triangle generated from that face
      auto tessellatedNormals =
          tessellateUniformData(normals, faceVertexCounts);

      std::vector<float3> normalData;
      normalData.reserve(tessellatedNormals.size());
      for (const auto &n : tessellatedNormals) {
        normalData.push_back(float3(n[0], n[1], n[2]));
      }

      auto normalsArray =
          scene.createArray(ANARI_FLOAT32_VEC3, normalData.size());
      normalsArray->setData(normalData.data(), normalData.size());
      meshObj->setParameterObject("primitive.normal", *normalsArray);

      logStatus("[import_USD] Mesh '%s': Set %zu normals on primitive.normal\n",
          prim.GetName().GetString().c_str(),
          normalData.size());
    }
  }

  // Handle UVs based on USD interpolation
  if (!uvs.empty()) {
    if (uvsInterpolation == pxr::UsdGeomTokens->vertex) {
      // Vertex interpolation: UVs are per-vertex, shared by all triangles
      // No tessellation needed - just convert to float2
      std::vector<float2> uvData;
      uvData.reserve(uvs.size());
      for (const auto &uv : uvs) {
        // USD is bottom-up, ANARI is top-down
        uvData.push_back(float2(uv[0], 1.0f - uv[1]));
      }

      auto uvsArray = scene.createArray(ANARI_FLOAT32_VEC2, uvData.size());
      uvsArray->setData(uvData.data(), uvData.size());
      meshObj->setParameterObject("vertex.attribute0", *uvsArray);

      logStatus("[import_USD] Mesh '%s': Set %zu UVs on vertex.attribute0\n",
          prim.GetName().GetString().c_str(),
          uvData.size());

    } else if (uvsInterpolation == pxr::UsdGeomTokens->faceVarying) {
      // FaceVarying interpolation: UVs are per face-vertex (corner)
      // Need to tessellate from polygon corners to triangle corners
      auto tessellatedUVs = tessellateFacevaryingData(uvs, faceVertexCounts);

      std::vector<float2> uvData;
      uvData.reserve(tessellatedUVs.size());
      for (const auto &uv : tessellatedUVs) {
        // USD is bottom-up, ANARI is top-down
        uvData.push_back(float2(uv[0], 1.0f - uv[1]));
      }

      auto uvsArray = scene.createArray(ANARI_FLOAT32_VEC2, uvData.size());
      uvsArray->setData(uvData.data(), uvData.size());
      meshObj->setParameterObject("faceVarying.attribute0", *uvsArray);

      logStatus(
          "[import_USD] Mesh '%s': Set %zu UVs on faceVarying.attribute0\n",
          prim.GetName().GetString().c_str(),
          uvData.size());

    } else if (uvsInterpolation == pxr::UsdGeomTokens->uniform) {
      // Uniform interpolation: one UV per face
      // Need to replicate for each triangle generated from that face
      auto tessellatedUVs = tessellateUniformData(uvs, faceVertexCounts);

      std::vector<float2> uvData;
      uvData.reserve(tessellatedUVs.size());
      for (const auto &uv : tessellatedUVs) {
        // USD is bottom-up, ANARI is top-down
        uvData.push_back(float2(uv[0], 1.0f - uv[1]));
      }

      auto uvsArray = scene.createArray(ANARI_FLOAT32_VEC2, uvData.size());
      uvsArray->setData(uvData.data(), uvData.size());
      meshObj->setParameterObject("primitive.attribute0", *uvsArray);

      logStatus("[import_USD] Mesh '%s': Set %zu UVs on primitive.attribute0\n",
          prim.GetName().GetString().c_str(),
          uvData.size());
    }
  }

  meshObj->setName(prim.GetPath().GetText());

  // Material binding
  MaterialRef mat =
      getBoundMaterial(scene, prim, basePath, matCache, texCache);

  auto surface = scene.createSurface(primName.c_str(), meshObj, mat);
  logStatus("[import_USD] Assigned material to mesh '%s': %s\n",
      primName.c_str(),
      mat->name().c_str());
  scene.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomPoints prim as a TSD sphere geometry (point cloud),
// with animation if the positions/widths are time-sampled.
static void importUsdPoints(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    tsd::animation::AnimationManager &animMgr,
    MaterialCache &matCache,
    TextureCache &texCache)
{
  pxr::UsdGeomPoints pointsPrim(prim);
  std::string primName = prim.GetPath().GetString();
  if (primName.empty())
    primName = "<unnamed_points>";

  std::vector<double> timeSamples;
  pointsPrim.GetPointsAttr().GetTimeSamples(&timeSamples);

  // Build position+radius arrays for one time step
  auto buildFrame = [&](pxr::UsdTimeCode tc)
      -> std::pair<ObjectUsePtr<Array>, ObjectUsePtr<Array>> {
    pxr::VtArray<pxr::GfVec3f> pts;
    pxr::VtArray<float> wids;
    pointsPrim.GetPointsAttr().Get(&pts, tc);
    pointsPrim.GetWidthsAttr().Get(&wids, tc);
    std::vector<float3> outPos;
    std::vector<float> outRad;
    outPos.reserve(pts.size());
    outRad.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
      const auto &p = pts[i];
      pxr::GfVec4d wp4 = usdXform * pxr::GfVec4d(p[0], p[1], p[2], 1.0);
      outPos.push_back(float3(float(wp4[0]), float(wp4[1]), float(wp4[2])));
      outRad.push_back((wids.size() == pts.size()) ? wids[i] * 0.5f : 0.01f);
    }
    auto pa = scene.createArray(ANARI_FLOAT32_VEC3, outPos.size());
    pa->setData(outPos.data(), outPos.size());
    auto ra = scene.createArray(ANARI_FLOAT32, outRad.size());
    ra->setData(outRad.data(), outRad.size());
    return {pa, ra};
  };

  pxr::UsdTimeCode firstTC = timeSamples.empty()
      ? pxr::UsdTimeCode::EarliestTime()
      : pxr::UsdTimeCode(timeSamples[0]);

  auto [firstPosArray, firstRadArray] = buildFrame(firstTC);
  if (!firstPosArray || firstPosArray->size() == 0) {
    logStatus("[import_USD] Skipping Points prim with no point data: %s\n",
        primName.c_str());
    return;
  }

  auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);
  geom->setName(primName.c_str());
  geom->setParameterObject("vertex.position", *firstPosArray);
  geom->setParameterObject("vertex.radius", *firstRadArray);

  MaterialRef mat = getBoundMaterial(scene, prim, "", matCache, texCache);
  auto surface = scene.createSurface(primName.c_str(), geom, mat);
  scene.insertChildObjectNode(parent, surface);

  logStatus("[import_USD] Imported Points '%s' (%zu pts, %zu frames)\n",
      primName.c_str(),
      firstPosArray->size(),
      timeSamples.empty() ? size_t(1) : timeSamples.size());

  // Build animation if there are multiple time samples
  if (timeSamples.size() > 1) {
    std::vector<ObjectUsePtr<Array>> posArrays, radArrays;
    posArrays.push_back(firstPosArray);
    radArrays.push_back(firstRadArray);
    for (size_t ti = 1; ti < timeSamples.size(); ++ti) {
      auto [pa, ra] = buildFrame(pxr::UsdTimeCode(timeSamples[ti]));
      if (pa && pa->size() > 0) {
        posArrays.push_back(pa);
        radArrays.push_back(ra);
      }
    }
    if (posArrays.size() > 1) {
      auto tb = makeLinearTimeBase(posArrays.size());
      auto &anim = animMgr.addAnimation(primName.c_str());
      addArrayTimeStepBindings(anim,
          geom.data(),
          {Token("vertex.position"), Token("vertex.radius")},
          {posArrays, radArrays},
          tb);
    }
  }
}

// Helper: Import a UsdGeomBasisCurves prim as TSD curve geometry,
// with animation if the positions are time-sampled.
static void importUsdCurves(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    tsd::animation::AnimationManager &animMgr,
    MaterialCache &matCache,
    TextureCache &texCache)
{
  pxr::UsdGeomBasisCurves curvesPrim(prim);
  std::string primName = prim.GetPath().GetString();
  if (primName.empty())
    primName = "<unnamed_curves>";

  // Curve topology is static (curveVertexCounts doesn't animate).
  // Use EarliestTime() to handle attributes stored at time samples but not
  // at default time.
  pxr::VtArray<int> curveVertexCounts;
  curvesPrim.GetCurveVertexCountsAttr().Get(
      &curveVertexCounts, pxr::UsdTimeCode::EarliestTime());
  if (curveVertexCounts.empty()) {
    logStatus("[import_USD] Skipping BasisCurves with no vertex counts: %s\n",
        primName.c_str());
    return;
  }

  // Build primitive.index: for each curve of N vertices, emit N-1 segment
  // start indices, leaving a gap between curves so they stay separate.
  std::vector<uint32_t> segIndices;
  {
    uint32_t base = 0;
    for (int count : curveVertexCounts) {
      for (int i = 0; i < count - 1; ++i)
        segIndices.push_back(base + uint32_t(i));
      base += uint32_t(count);
    }
  }
  if (segIndices.empty()) {
    logStatus("[import_USD] Skipping BasisCurves with no segments: %s\n",
        primName.c_str());
    return;
  }

  // Build a position array for one time step
  auto buildFrame = [&](pxr::UsdTimeCode tc) -> ObjectUsePtr<Array> {
    pxr::VtArray<pxr::GfVec3f> pts;
    curvesPrim.GetPointsAttr().Get(&pts, tc);
    if (pts.empty())
      return {};
    std::vector<float3> outPts;
    outPts.reserve(pts.size());
    for (const auto &p : pts) {
      pxr::GfVec4d wp4 = usdXform * pxr::GfVec4d(p[0], p[1], p[2], 1.0);
      outPts.push_back(float3(float(wp4[0]), float(wp4[1]), float(wp4[2])));
    }
    auto arr = scene.createArray(ANARI_FLOAT32_VEC3, outPts.size());
    arr->setData(outPts.data(), outPts.size());
    return arr;
  };

  std::vector<double> timeSamples;
  curvesPrim.GetPointsAttr().GetTimeSamples(&timeSamples);

  pxr::UsdTimeCode firstTC = timeSamples.empty()
      ? pxr::UsdTimeCode::EarliestTime()
      : pxr::UsdTimeCode(timeSamples[0]);

  auto firstPosArray = buildFrame(firstTC);
  if (!firstPosArray) {
    logStatus("[import_USD] Skipping BasisCurves with no point data: %s\n",
        primName.c_str());
    return;
  }

  // Build a per-vertex radius array (barney's Curve::bounds() unconditionally
  // dereferences vertex.radius, so we must always supply it even when using a
  // uniform radius).
  const float kCurveRadius = 0.3f;
  std::vector<float> radii(firstPosArray->size(), kCurveRadius);
  auto radArray = scene.createArray(ANARI_FLOAT32, radii.size());
  radArray->setData(radii.data(), radii.size());

  auto geom = scene.createObject<Geometry>(tokens::geometry::curve);
  geom->setName(primName.c_str());
  geom->setParameterObject("vertex.position", *firstPosArray);
  geom->setParameterObject("vertex.radius", *radArray);

  // barney's Curve::setBarneyParameters() also unconditionally dereferences
  // primitive.index, so we must always supply it.
  auto idxArray = scene.createArray(ANARI_UINT32, segIndices.size());
  idxArray->setData(segIndices.data(), segIndices.size());
  geom->setParameterObject("primitive.index", *idxArray);

  MaterialRef mat = getBoundMaterial(scene, prim, "", matCache, texCache);
  auto surface = scene.createSurface(primName.c_str(), geom, mat);
  scene.insertChildObjectNode(parent, surface);

  logStatus("[import_USD] Imported BasisCurves '%s' (%zu segs, %zu frames)\n",
      primName.c_str(),
      segIndices.size(),
      timeSamples.empty() ? size_t(1) : timeSamples.size());

  // Build animation if there are multiple time samples
  if (timeSamples.size() > 1) {
    std::vector<ObjectUsePtr<Array>> posArrays;
    posArrays.push_back(firstPosArray);
    for (size_t ti = 1; ti < timeSamples.size(); ++ti) {
      auto arr = buildFrame(pxr::UsdTimeCode(timeSamples[ti]));
      if (arr)
        posArrays.push_back(arr);
    }
    if (posArrays.size() > 1) {
      auto tb = makeLinearTimeBase(posArrays.size());
      auto &anim = animMgr.addAnimation(primName.c_str());
      addArrayTimeStepBindings(anim,
          geom.data(),
          {Token("vertex.position")},
          {posArrays},
          tb);
    }
  }
}

// Helper: Import a UsdGeomSphere prim as a TSD sphere geometry
static void importUsdSphere(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    MaterialCache &matCache,
    TextureCache &texCache)
{
  pxr::UsdGeomSphere spherePrim(prim);
  // UsdGeomSphere is always centered at the origin in local space
  pxr::GfVec3f center(0.f, 0.f, 0.f);
  double radius = 1.0;
  spherePrim.GetRadiusAttr().Get(&radius);
  pxr::GfVec4d c4(center[0], center[1], center[2], 1.0);
  pxr::GfVec4d wc4 = usdXform * c4;
  float3 wp{float(wc4[0]), float(wc4[1]), float(wc4[2])};
  auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);
  auto posArray = scene.createArray(ANARI_FLOAT32_VEC3, 1);
  posArray->setData(&wp, 1);
  auto radArray = scene.createArray(ANARI_FLOAT32, 1);
  float r = float(radius);
  radArray->setData(&r, 1);
  geom->setParameterObject("vertex.position", *posArray);
  geom->setParameterObject("vertex.radius", *radArray);
  std::string primName = prim.GetPath().GetString();
  if (primName.empty())
    primName = "<unnamed_sphere>";
  geom->setName(primName.c_str());

  // Material binding
  MaterialRef mat = getBoundMaterial(scene, prim, "", matCache, texCache);

  auto surface = scene.createSurface(primName.c_str(), geom, mat);
  logStatus("[import_USD] Assigned material to sphere '%s': %s\n",
      primName.c_str(),
      mat->name().c_str());
  scene.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomCone prim as a TSD cone geometry
static void importUsdCone(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    MaterialCache &matCache,
    TextureCache &texCache)
{
  pxr::UsdGeomCone conePrim(prim);
  // UsdGeomCone is always centered at the origin in local space
  pxr::GfVec3f center(0.f, 0.f, 0.f);
  double height = 2.0;
  conePrim.GetHeightAttr().Get(&height);
  double radius = 1.0;
  conePrim.GetRadiusAttr().Get(&radius);
  pxr::TfToken axis;
  conePrim.GetAxisAttr().Get(&axis);
  // TODO: Handle axis != Z
  pxr::GfVec4d c4(center[0], center[1], center[2], 1.0);
  pxr::GfVec4d wc4 = usdXform * c4;
  float3 wp{float(wc4[0]), float(wc4[1]), float(wc4[2])};
  // Represent as a 2-point cone (base and apex)
  std::vector<float3> positions = {wp, wp + float3(0, 0, float(height))};
  std::vector<float> radii = {float(radius), 0.f};
  auto geom = scene.createObject<Geometry>(tokens::geometry::cone);
  auto posArray = scene.createArray(ANARI_FLOAT32_VEC3, 2);
  posArray->setData(positions.data(), 2);
  auto radArray = scene.createArray(ANARI_FLOAT32, 2);
  radArray->setData(radii.data(), 2);
  geom->setParameterObject("vertex.position", *posArray);
  geom->setParameterObject("vertex.radius", *radArray);
  std::string primName = prim.GetPath().GetString();
  if (primName.empty())
    primName = "<unnamed_cone>";
  geom->setName(primName.c_str());

  // Material binding
  MaterialRef mat = getBoundMaterial(scene, prim, "", matCache, texCache);

  auto surface = scene.createSurface(primName.c_str(), geom, mat);
  logStatus("[import_USD] Assigned material to cone '%s': %s\n",
      primName.c_str(),
      mat->name().c_str());
  scene.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomCylinder prim as a TSD cylinder geometry
static void importUsdCylinder(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    MaterialCache &matCache,
    TextureCache &texCache)
{
  pxr::UsdGeomCylinder cylPrim(prim);
  // UsdGeomCylinder is always centered at the origin in local space
  pxr::GfVec3f center(0.f, 0.f, 0.f);
  double height = 2.0;
  cylPrim.GetHeightAttr().Get(&height);
  double radius = 1.0;
  cylPrim.GetRadiusAttr().Get(&radius);
  pxr::TfToken axis;
  cylPrim.GetAxisAttr().Get(&axis);
  // TODO: Handle axis != Z
  pxr::GfVec4d c4(center[0], center[1], center[2], 1.0);
  pxr::GfVec4d wc4 = usdXform * c4;
  float3 wp{float(wc4[0]), float(wc4[1]), float(wc4[2])};
  // Represent as a 2-point cylinder (bottom and top)
  std::vector<float3> positions = {wp - float3(0, 0, float(height) * 0.5f),
      wp + float3(0, 0, float(height) * 0.5f)};
  std::vector<float> radii = {float(radius), float(radius)};
  auto geom = scene.createObject<Geometry>(tokens::geometry::cylinder);
  auto posArray = scene.createArray(ANARI_FLOAT32_VEC3, 2);
  posArray->setData(positions.data(), 2);
  auto radArray = scene.createArray(ANARI_FLOAT32, 2);
  radArray->setData(radii.data(), 2);
  geom->setParameterObject("vertex.position", *posArray);
  geom->setParameterObject("vertex.radius", *radArray);
  std::string primName = prim.GetPath().GetString();
  if (primName.empty())
    primName = "<unnamed_cylinder>";
  geom->setName(primName.c_str());

  // Material binding
  MaterialRef mat = getBoundMaterial(scene, prim, "", matCache, texCache);

  auto surface = scene.createSurface(primName.c_str(), geom, mat);
  tsd::core::logStatus("[import_USD] Assigned material to cylinder '%s': %s\n",
      primName.c_str(),
      mat->name().c_str());
  scene.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomCube prim as a triangulated TSD triangle mesh,
// with animation if xformOps on the prim or its ancestors are time-sampled.
static void importUsdCube(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    const std::string &basePath,
    tsd::animation::AnimationManager &animMgr,
    MaterialCache &matCache,
    TextureCache &texCache)
{
  pxr::UsdGeomCube cubePrim(prim);
  double size = 2.0;
  cubePrim.GetSizeAttr().Get(&size, pxr::UsdTimeCode::EarliestTime());
  float h = float(size) * 0.5f;

  // 6 faces x 4 verts = 24 vertices with per-face normals, 12 triangles
  // Face order: +Z, -Z, +Y, -Y, +X, -X
  static const float cx[6][4][3] = {
      {{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}},    // +Z
      {{1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}, {1, 1, -1}}, // -Z
      {{-1, 1, -1}, {1, 1, -1}, {1, 1, 1}, {-1, 1, 1}},     // +Y
      {{-1, -1, 1}, {1, -1, 1}, {1, -1, -1}, {-1, -1, -1}}, // -Y
      {{1, -1, 1}, {1, -1, -1}, {1, 1, -1}, {1, 1, 1}},     // +X
      {{-1, -1, -1}, {-1, -1, 1}, {-1, 1, 1}, {-1, 1, -1}}, // -X
  };
  static const float lnx[6][3] = {
      {0, 0, 1}, {0, 0, -1}, {0, 1, 0}, {0, -1, 0}, {1, 0, 0}, {-1, 0, 0}};

  // Build index buffer (static, never changes)
  std::vector<uint32_t> indices;
  indices.reserve(36);
  for (int f = 0; f < 6; ++f) {
    uint32_t base = uint32_t(f * 4);
    indices.push_back(base + 0);
    indices.push_back(base + 1);
    indices.push_back(base + 2);
    indices.push_back(base + 0);
    indices.push_back(base + 2);
    indices.push_back(base + 3);
  }

  // Build position+normal arrays for one world transform.
  // Positions and normals are baked into world space so that animated xform
  // time steps can be represented by animating vertex.position/vertex.normal
  // (mirrors the approach used by importUsdCurves).
  auto buildFrame =
      [&](const pxr::GfMatrix4d &xfm)
      -> std::pair<ObjectUsePtr<Array>, ObjectUsePtr<Array>> {
    std::vector<float3> positions;
    std::vector<float3> normals;
    positions.reserve(24);
    normals.reserve(24);
    for (int f = 0; f < 6; ++f) {
      for (int v = 0; v < 4; ++v) {
        pxr::GfVec4d lp(cx[f][v][0] * h, cx[f][v][1] * h, cx[f][v][2] * h, 1.0);
        pxr::GfVec4d wp = xfm * lp;
        positions.push_back(float3(float(wp[0]), float(wp[1]), float(wp[2])));
      }
      // TransformDir applies rotation+scale only (no translation) — correct for normals
      // when there is no non-uniform scale. Normalize to handle uniform scale.
      pxr::GfVec3d wn = xfm.TransformDir(
          pxr::GfVec3d(lnx[f][0], lnx[f][1], lnx[f][2]));
      wn.Normalize();
      float3 wn3{float(wn[0]), float(wn[1]), float(wn[2])};
      for (int v = 0; v < 4; ++v)
        normals.push_back(wn3);
    }
    auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, positions.size());
    posArr->setData(positions.data(), positions.size());
    auto normArr = scene.createArray(ANARI_FLOAT32_VEC3, normals.size());
    normArr->setData(normals.data(), normals.size());
    return {posArr, normArr};
  };

  // Collect xform time samples from this prim only — do NOT walk ancestors.
  // Parent Xform animation is already handled by the animated transform node
  // created in importUsdPrimRecursive (setAsTransformSteps).  Walking up
  // the hierarchy would bake the parent rotation into world-space vertex
  // positions while the transform node applies it a second time, producing a
  // double-transform (e.g. 720° apparent rotation for a 360° animated parent).
  std::vector<double> timeSamples;
  {
    pxr::UsdGeomXformable xformable(prim);
    if (xformable)
      xformable.GetTimeSamples(&timeSamples);
    std::sort(timeSamples.begin(), timeSamples.end());
    timeSamples.erase(
        std::unique(timeSamples.begin(), timeSamples.end()), timeSamples.end());
  }

  std::string primName = prim.GetPath().GetString();
  if (primName.empty())
    primName = "<unnamed_cube>";

  auto geom = scene.createObject<Geometry>(tokens::geometry::triangle);
  geom->setName(primName.c_str());

  auto [firstPos, firstNorm] = buildFrame(usdXform);
  geom->setParameterObject("vertex.position", *firstPos);
  geom->setParameterObject("vertex.normal", *firstNorm);

  auto idxArr = scene.createArray(ANARI_UINT32_VEC3, indices.size() / 3);
  idxArr->setData((uint3 *)indices.data(), indices.size() / 3);
  geom->setParameterObject("primitive.index", *idxArr);

  MaterialRef mat =
      getBoundMaterial(scene, prim, basePath, matCache, texCache);
  auto surface = scene.createSurface(primName.c_str(), geom, mat);
  logStatus("[import_USD] Assigned material to cube '%s': %s\n",
      primName.c_str(),
      mat->name().c_str());
  scene.insertChildObjectNode(parent, surface);

  // Build per-frame animation if xform ops have time samples
  if (timeSamples.size() > 1) {
    std::vector<ObjectUsePtr<Array>> posArrays;
    std::vector<ObjectUsePtr<Array>> normArrays;
    posArrays.push_back(firstPos);
    normArrays.push_back(firstNorm);

    pxr::UsdGeomXformCache cache;
    for (size_t ti = 1; ti < timeSamples.size(); ++ti) {
      cache.SetTime(pxr::UsdTimeCode(timeSamples[ti]));
      bool resets = false;
      auto xfm = cache.GetLocalTransformation(prim, &resets);
      auto [posArr, normArr] = buildFrame(xfm);
      posArrays.push_back(posArr);
      normArrays.push_back(normArr);
    }

    auto tb = makeLinearTimeBase(posArrays.size());
    auto &anim = animMgr.addAnimation(primName.c_str());
    addArrayTimeStepBindings(anim,
        geom.data(),
        {Token("vertex.position"), Token("vertex.normal")},
        {posArrays, normArrays},
        tb);

    logStatus(
        "[import_USD] Cube '%s': animated xform over %zu frames\n",
        primName.c_str(),
        timeSamples.size());
  }
}

// Helper: Import a UsdVolVolume prim as a TSD volume geometry
static void importUsdVolume(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform)
{
  pxr::UsdVolVolume volumePrim(prim);

  std::string primName = prim.GetPath().GetString();
  if (primName.empty())
    primName = "<unnamed_volume>";

  // Find the field data by following field relationships
  std::string filePath;

  // Try field:volume relationship first (for VDB volumes and OpenVDBAsset)
  pxr::UsdRelationship fieldRel =
      prim.GetRelationship(pxr::TfToken("field:volume"));
  if (!fieldRel) {
    // Fall back to field:density relationship for other volume types
    fieldRel = prim.GetRelationship(pxr::TfToken("field:density"));
  }

  if (fieldRel) {
    pxr::SdfPathVector targets;
    fieldRel.GetTargets(&targets);
    if (!targets.empty()) {
      // Get the field prim (could be OpenVDBAsset, FieldBase, etc.)
      pxr::UsdPrim fieldPrim = prim.GetStage()->GetPrimAtPath(targets[0]);
      if (fieldPrim) {
        pxr::UsdAttribute filePathAttr =
            fieldPrim.GetAttribute(pxr::TfToken("filePath"));
        if (filePathAttr) {
          pxr::SdfAssetPath assetPath;
          if (filePathAttr.Get(&assetPath)) {
            filePath = assetPath.GetResolvedPath();
            if (filePath.empty())
              filePath = assetPath.GetAssetPath();
          }
        }
      }
    }
  }

  if (filePath.empty()) {
    tsd::core::logStatus(
        "[import_USD] No field data file found for volume '%s'\n",
        primName.c_str());
    return;
  }

  SpatialFieldRef field;
  const auto ext = extensionOf(filePath);
  if (ext == ".raw")
    field = import_RAW(scene, filePath.c_str());
  else if (ext == ".flash")
    field = import_FLASH(scene, filePath.c_str());
  else if (ext == ".nvdb" || ext == ".vdb")
    field = import_NVDB(scene, filePath.c_str());
  else if (ext == ".mhd")
    field = import_MHD(scene, filePath.c_str());
  else {
    throw std::runtime_error(
        "[import_USD] no loader for file type '" + ext + "'");
  }

  if (!field) {
    tsd::core::logStatus(
        "[import_USD] No field data found for volume '%s'\n", primName.c_str());
    return;
  }

  // Get volume bounds from the field itself (MHD files contain spatial
  // information) We'll let the field define its own spatial extents

  // Check for transfer function from USD material
  VolumeTransferFunction tf = getVolumeTransferFunction(prim);

  ArrayRef colorArray;
  // Default to the field's value range to avoid undefined ranges.
  math::float2 valueRange = field->computeValueRange();

  // Create a volume node and assign the field, color map, and value range
  auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
      parent, tokens::volume::transferFunction1D);
  volume->setName(primName.c_str());
  volume->setParameterObject("value", *field);

  if (tf.hasTransferFunction && !tf.colors.empty() && !tf.xPoints.empty()) {
    // Use transfer function from USD material
    colorArray = scene.createArray(ANARI_FLOAT32_VEC4, tf.colors.size());
    colorArray->setData(tf.colors.data(), tf.colors.size());
    if (tf.domain.x < tf.domain.y)
      valueRange = tf.domain;

    // Create opacity control points from USD transfer function
    std::vector<math::float2> opacityControlPoints;
    opacityControlPoints.reserve(tf.colors.size());

    for (size_t i = 0; i < tf.colors.size(); ++i) {
      // x = position in transfer function, y = opacity value
      opacityControlPoints.emplace_back(tf.xPoints[i], tf.colors[i].w);
    }

    // Set the opacity control points as metadata
    volume->setMetadataArray("opacityControlPoints",
        ANARI_FLOAT32_VEC2,
        opacityControlPoints.data(),
        opacityControlPoints.size());
  } else {
    // Use default transfer function if available, otherwise create default
    colorArray = scene.createArray(ANARI_FLOAT32_VEC4, 256);
    colorArray->setData(makeDefaultColorMap(colorArray->size()));
  }

  volume->setParameterObject("color", *colorArray);
  volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
}

// -----------------------------------------------------------------------------
// Light import helpers
// -----------------------------------------------------------------------------

static void importUsdDistantLight(
    Scene &scene, const pxr::UsdPrim &prim, LayerNodeRef parent)
{
  pxr::UsdLuxDistantLight usdLight(prim);
  auto light = scene.createObject<Light>(tokens::light::directional);
  float intensity = 1.0f;
  usdLight.GetIntensityAttr().Get(&intensity);
  pxr::GfVec3f color(1.0f);
  usdLight.GetColorAttr().Get(&color);
  light->setParameter("color", float3(color[0], color[1], color[2]));
  light->setParameter("irradiance", intensity);
  // TODO: set direction from transform
  scene.insertChildObjectNode(parent, light);
}

static void importUsdRectLight(
    Scene &scene, const pxr::UsdPrim &prim, LayerNodeRef parent)
{
  pxr::UsdLuxRectLight usdLight(prim);
  auto light = scene.createObject<Light>(tokens::light::quad);
  float intensity = 1.0f;
  usdLight.GetIntensityAttr().Get(&intensity);
  pxr::GfVec3f color(1.0f);
  usdLight.GetColorAttr().Get(&color);
  double width = 1.0, height = 1.0;
  usdLight.GetWidthAttr().Get(&width);
  usdLight.GetHeightAttr().Get(&height);
  light->setParameter("color", float3(color[0], color[1], color[2]));
  light->setParameter("intensity", intensity);
  light->setParameter("edge1", float3(width, 0.f, 0.f));
  light->setParameter("edge2", float3(0.f, height, 0.f));
  // TODO: set position from transform
  scene.insertChildObjectNode(parent, light);
}

static void importUsdSphereLight(
    Scene &scene, const pxr::UsdPrim &prim, LayerNodeRef parent)
{
  pxr::UsdLuxSphereLight usdLight(prim);
  auto light = scene.createObject<Light>(tokens::light::point);
  float intensity = 1.0f;
  usdLight.GetIntensityAttr().Get(&intensity);
  pxr::GfVec3f color(1.0f);
  usdLight.GetColorAttr().Get(&color);
  double radius = 1.0;
  usdLight.GetRadiusAttr().Get(&radius);
  light->setParameter("color", float3(color[0], color[1], color[2]));
  light->setParameter("intensity", intensity);
  // TODO: set position from transform
  // Optionally, set radius as metadata or custom param
  scene.insertChildObjectNode(parent, light);
}

static void importUsdDiskLight(
    Scene &scene, const pxr::UsdPrim &prim, LayerNodeRef parent)
{
  pxr::UsdLuxDiskLight usdLight(prim);
  auto light = scene.createObject<Light>(tokens::light::ring);
  float intensity = 1.0f;
  usdLight.GetIntensityAttr().Get(&intensity);
  pxr::GfVec3f color(1.0f);
  usdLight.GetColorAttr().Get(&color);
  double radius = 1.0;
  usdLight.GetRadiusAttr().Get(&radius);
  light->setParameter("color", float3(color[0], color[1], color[2]));
  light->setParameter("intensity", intensity);
  // TODO: set position from transform
  // Optionally, set radius as metadata or custom param
  scene.insertChildObjectNode(parent, light);
}

static void importUsdDomeLight(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const std::string &basePath,
    const pxr::GfMatrix4d &usdXform)
{
  pxr::UsdLuxDomeLight usdLight(prim);
  auto light = scene.createObject<Light>(tokens::light::hdri);
  light->setName(prim.GetName().GetText());
  float intensity = 1.0f;
  usdLight.GetIntensityAttr().Get(&intensity);
  pxr::GfVec3f color(1.0f);
  usdLight.GetColorAttr().Get(&color);
  light->setParameter("color", float3(color[0], color[1], color[2]));
  light->setParameter("scale", intensity);

  // Extract direction and up vectors from transformation matrix
  // ANARI defaults: direction=(1,0,0), up=(0,0,1)
  // USD dome lights use Z-up by default, matching ANARI
  auto xfm = pxr::GfMatrix4d(
      // clang-format off
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0
      // clang-format on
  );
  xfm *= usdXform;
  pxr::GfVec3d dirVec = xfm.TransformDir(pxr::GfVec3d(0, 0, -1));
  pxr::GfVec3d upVec = xfm.TransformDir(pxr::GfVec3d(0, 1, 0));

  float3 direction(dirVec[0], dirVec[1], dirVec[2]);
  float3 up(upVec[0], upVec[1], upVec[2]);

  light->setParameter("direction", direction);
  light->setParameter("up", up);

  // Synthesize a 1x1 fallback radiance from solid color * intensity if no
  // texture is provided (barney requires radiance to be set on HDRI lights).
  {
    float3 solidColor(color[0], color[1], color[2]);
    solidColor *= intensity;
    auto fallbackRadiance = scene.createArray(ANARI_FLOAT32_VEC3, 1, 1);
    fallbackRadiance->setData(&solidColor);
    light->setParameterObject("radiance", *fallbackRadiance);
  }

  // Load and set environment texture from usdLight.GetTextureFileAttr()
  pxr::SdfAssetPath textureAsset;
  if (usdLight.GetTextureFileAttr().Get(&textureAsset)) {
    std::string texFile = textureAsset.GetResolvedPath();
    if (texFile.empty())
      texFile = textureAsset.GetAssetPath();
    if (!texFile.empty()) {
      // Use basePath to resolve relative paths if needed
      std::string resolvedPath = texFile;
      if (!resolvedPath.empty() && resolvedPath[0] != '/') {
        // Try to resolve relative to basePath
        resolvedPath = basePath + texFile;
      }

      ArrayRef radiance = {};
      if (resolvedPath.find(".exr") != std::string::npos
          || resolvedPath.find(".hdr") != std::string::npos) {
        HDRImage img;
        if (img.import(resolvedPath)) {
          std::vector<float3> rgb(img.width * img.height);

          if (img.numComponents == 3) {
            memcpy(rgb.data(), img.pixel.data(), sizeof(rgb[0]) * rgb.size());
          } else if (img.numComponents == 4) {
            for (size_t i = 0; i < img.pixel.size(); i += 4) {
              rgb[i / 4] =
                  float3(img.pixel[i], img.pixel[i + 1], img.pixel[i + 2]);
            }
          }

          // Handle color temperature if present
          float colorTemp = 0.0f;
          if (usdLight.GetColorTemperatureAttr().Get(&colorTemp)
              && colorTemp > 0.0f) {
            // Convert color temperature to RGB multiplier
            // Using approximation from Planckian locus
            auto kelvinToRGB = [](float kelvin) -> float3 {
              // https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html
              float temp = kelvin / 100.0f;
              float red, green, blue;

              // Calculate red
              if (temp <= 66.0f) {
                red = 1.0f;
              } else {
                red = temp - 60.0f;
                red = 329.698727446f * std::pow(red, -0.1332047592f);
                red = std::clamp(red / 255.0f, 0.0f, 1.0f);
              }

              // Calculate green
              if (temp <= 66.0f) {
                green = temp;
                green = 99.4708025861f * std::log(green) - 161.1195681661f;
                green = std::clamp(green / 255.0f, 0.0f, 1.0f);
              } else {
                green = temp - 60.0f;
                green = 288.1221695283f * std::pow(green, -0.0755148492f);
                green = std::clamp(green / 255.0f, 0.0f, 1.0f);
              }

              // Calculate blue
              if (temp >= 66.0f) {
                blue = 1.0f;
              } else if (temp <= 19.0f) {
                blue = 0.0f;
              } else {
                blue = temp - 10.0f;
                blue = 138.5177312231f * std::log(blue) - 305.0447927307f;
                blue = std::clamp(blue / 255.0f, 0.0f, 1.0f);
              }

              return float3(red, green, blue);
            };

            float3 tempColor = kelvinToRGB(colorTemp);
            for (auto &color : rgb) {
              color *= float3(tempColor.x, tempColor.y, tempColor.z);
            }
            tsd::core::logStatus(
                "[import_USD] Applied dome light color temperature: %f K (%f %f %f)\n",
                colorTemp,
                tempColor.x,
                tempColor.y,
                tempColor.z);
          }

          // Apply exposure adjustment if present
          float exposure = 0.0f;
          if (usdLight.GetExposureAttr().Get(&exposure)) {
            // Convert exposure to linear scale: multiplier = 2^exposure
            float exposureScale = std::pow(2.0f, exposure);
            for (auto &color : rgb) {
              color *= exposureScale;
            }
            tsd::core::logStatus(
                "[import_USD] Applied dome light exposure: %f (scale: %f)\n",
                exposure,
                exposureScale);
          }

          radiance =
              scene.createArray(ANARI_FLOAT32_VEC3, img.width, img.height);
          radiance->setData(rgb.data());
        }
      }
      if (radiance)
        light->setParameterObject("radiance", *radiance);
      else
        tsd::core::logStatus(
            "[import_USD] Warning: Failed to load dome light texture: %s\n",
            resolvedPath.c_str());
    }
  }
  scene.insertChildObjectNode(parent, light);
}

// Helper: Import a UsdGeomCamera prim as an animated TSD camera.
// Collects xform time samples from the prim and parent hierarchy so that
// orbit/crane rigs animate correctly even when the camera prim itself is static.
static void importUsdCamera(Scene &scene,
    const pxr::UsdPrim &prim,
    tsd::animation::AnimationManager &animMgr)
{
  std::string primName = prim.GetName().GetString();
  if (primName.empty())
    primName = "<unnamed_camera>";

  pxr::UsdGeomCamera usdCamera(prim);

  // Read intrinsics at default time (usually static)
  pxr::GfCamera gfCamDef = usdCamera.GetCamera(pxr::UsdTimeCode::Default());
  bool isPerspective =
      gfCamDef.GetProjection() == pxr::GfCamera::Perspective;
  const char *cameraType = isPerspective ? "perspective" : "orthographic";
  float focalLength = gfCamDef.GetFocalLength();
  float horizAp = gfCamDef.GetHorizontalAperture();
  float vertAp = gfCamDef.GetVerticalAperture();
  float fovV = 2.f * std::atan(vertAp / (2.f * focalLength));

  auto camera = scene.createObject<Camera>(cameraType);
  camera->setName(primName.c_str());

  if (isPerspective) {
    camera->setParameter("fovy", fovV);
    camera->setParameter("aspect", horizAp / vertAp);
  } else {
    camera->setParameter("height", vertAp);
    camera->setParameter("aspect", horizAp / vertAp);
  }

  // Collect xform time samples from prim and its parent hierarchy
  std::vector<double> timeSamples;
  for (auto cur = prim; cur && !cur.IsPseudoRoot(); cur = cur.GetParent()) {
    pxr::UsdGeomXformable xformable(cur);
    if (xformable) {
      std::vector<double> ts;
      xformable.GetTimeSamples(&ts);
      for (double t : ts)
        timeSamples.push_back(t);
    }
  }

  // Also collect time samples from all animatable intrinsic attributes
  std::vector<double> intrinsicTs;
  {
    auto collect = [&](pxr::UsdAttribute attr) {
      std::vector<double> tmp;
      attr.GetTimeSamples(&tmp);
      for (double t : tmp)
        intrinsicTs.push_back(t);
    };
    collect(usdCamera.GetFocalLengthAttr());
    collect(usdCamera.GetHorizontalApertureAttr());
    collect(usdCamera.GetVerticalApertureAttr());
    collect(usdCamera.GetFStopAttr());
    collect(usdCamera.GetFocusDistanceAttr());
    collect(usdCamera.GetClippingRangeAttr());
  }
  bool hasIntrinsicAnimation = !intrinsicTs.empty();
  for (double t : intrinsicTs)
    timeSamples.push_back(t);

  std::sort(timeSamples.begin(), timeSamples.end());
  timeSamples.erase(
      std::unique(timeSamples.begin(), timeSamples.end()), timeSamples.end());

  // Compute world-space pose from a transform cache at a given time
  auto buildPose = [&](pxr::UsdGeomXformCache &cache)
      -> std::tuple<float3, float3, float3> {
    auto xfm = cache.GetLocalToWorldTransform(prim);
    auto gfPos = xfm.Transform(pxr::GfVec3d(0, 0, 0));
    auto gfDir = xfm.TransformDir(pxr::GfVec3d(0, 0, -1));
    gfDir.Normalize();
    auto gfUp = xfm.TransformDir(pxr::GfVec3d(0, 1, 0));
    gfUp.Normalize();
    return {float3{float(gfPos[0]), float(gfPos[1]), float(gfPos[2])},
        float3{float(gfDir[0]), float(gfDir[1]), float(gfDir[2])},
        float3{float(gfUp[0]), float(gfUp[1]), float(gfUp[2])}};
  };

  // Always seed the camera with a valid pose at Default time
  {
    pxr::UsdGeomXformCache initCache(pxr::UsdTimeCode::Default());
    auto [pos, dir, up] = buildPose(initCache);
    camera->setParameter("position", pos);
    camera->setParameter("direction", dir);
    camera->setParameter("up", up);
  }

  if (timeSamples.size() <= 1) {
    logStatus("[import_USD] Created static camera '%s'\n", primName.c_str());
    return;
  }

  // Build flat per-param arrays (TimeStepValues: one big array per parameter,
  // element-indexed by frame — same pattern as Context.cpp camera path animation)
  size_t numFrames = timeSamples.size();
  auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, numFrames);
  auto dirArr = scene.createArray(ANARI_FLOAT32_VEC3, numFrames);
  auto upArr = scene.createArray(ANARI_FLOAT32_VEC3, numFrames);
  posArr->setName((primName + "_anim_position").c_str());
  dirArr->setName((primName + "_anim_direction").c_str());
  upArr->setName((primName + "_anim_up").c_str());

  // Intrinsic animation arrays (only allocated when needed)
  ObjectUsePtr<Array> fovArr, aspectArr, focusDistArr, apertureRadiusArr;
  float *fovs = nullptr, *aspects = nullptr;
  float *focusDists = nullptr, *apertureRadii = nullptr;
  if (hasIntrinsicAnimation) {
    fovArr = scene.createArray(ANARI_FLOAT32, numFrames);
    fovArr->setName((primName + "_anim_fovy").c_str());
    fovs = fovArr->mapAs<float>();

    aspectArr = scene.createArray(ANARI_FLOAT32, numFrames);
    aspectArr->setName((primName + "_anim_aspect").c_str());
    aspects = aspectArr->mapAs<float>();

    if (isPerspective) {
      focusDistArr = scene.createArray(ANARI_FLOAT32, numFrames);
      focusDistArr->setName((primName + "_anim_focusDistance").c_str());
      focusDists = focusDistArr->mapAs<float>();

      apertureRadiusArr = scene.createArray(ANARI_FLOAT32, numFrames);
      apertureRadiusArr->setName((primName + "_anim_apertureRadius").c_str());
      apertureRadii = apertureRadiusArr->mapAs<float>();
    }
  }

  auto *positions = posArr->mapAs<math::float3>();
  auto *directions = dirArr->mapAs<math::float3>();
  auto *ups = upArr->mapAs<math::float3>();

  pxr::UsdGeomXformCache cache;
  for (size_t i = 0; i < numFrames; ++i) {
    pxr::UsdTimeCode tc(timeSamples[i]);
    cache.SetTime(tc);
    auto [pos, dir, up] = buildPose(cache);
    positions[i] = pos;
    directions[i] = dir;
    ups[i] = up;
    if (fovs) {
      pxr::GfCamera gfc = usdCamera.GetCamera(tc);
      float fl = gfc.GetFocalLength();
      float va = gfc.GetVerticalAperture();
      float ha = gfc.GetHorizontalAperture();
      fovs[i] = 2.f * std::atan(va / (2.f * fl));
      aspects[i] = ha / va;
      if (focusDists) {
        focusDists[i] = gfc.GetFocusDistance();
        // apertureRadius from fStop: fl is in tenths of scene units
        float fStop = gfc.GetFStop();
        apertureRadii[i] =
            fStop > 0.f ? (fl / 10.f) / (2.f * fStop) : 0.f;
      }
    }
  }

  posArr->unmap();
  dirArr->unmap();
  upArr->unmap();
  if (fovArr) {
    fovArr->unmap();
    aspectArr->unmap();
    if (focusDistArr) {
      focusDistArr->unmap();
      apertureRadiusArr->unmap();
    }
  }

  std::vector<Token> animParams{"position", "direction", "up"};
  std::vector<ObjectUsePtr<Array>> animArrays{posArr, dirArr, upArr};
  if (hasIntrinsicAnimation) {
    animParams.push_back("fovy");
    animArrays.push_back(fovArr);
    animParams.push_back("aspect");
    animArrays.push_back(aspectArr);
    if (isPerspective) {
      animParams.push_back("focusDistance");
      animArrays.push_back(focusDistArr);
      animParams.push_back("apertureRadius");
      animArrays.push_back(apertureRadiusArr);
    }
  }

  auto tb = makeLinearTimeBase(numFrames);
  auto &anim = animMgr.addAnimation(primName.c_str());
  addValueTimeStepBindings(anim,
      camera.data(),
      animParams,
      animArrays,
      tb,
      tsd::animation::InterpolationRule::LINEAR);

  logStatus("[import_USD] Created animated camera '%s' (%zu frames)\n",
      primName.c_str(),
      numFrames);
}

// Helper to check if a GfMatrix4d is identity
static bool isIdentity(const pxr::GfMatrix4d &m)
{
  static const pxr::GfMatrix4d IDENTITY(1.0);
  return m == IDENTITY;
}

// -----------------------------------------------------------------------------
// Recursive import function for prims and their children
// -----------------------------------------------------------------------------

static void importUsdPrimRecursive(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    pxr::UsdGeomXformCache &xformCache,
    const std::string &basePath,
    const pxr::GfMatrix4d &parentWorldXform,
    tsd::animation::AnimationManager &animMgr,
    MaterialCache &matCache,
    TextureCache &texCache)
{
  // if (prim.IsPrototype()) return;
  if (prim.IsInstance()) {
    pxr::UsdPrim prototype = prim.GetPrototype();
    if (prototype) {
      bool resetsXformStack = false;
      pxr::GfMatrix4d usdLocalXform =
          xformCache.GetLocalTransformation(prim, &resetsXformStack);
      pxr::GfMatrix4d thisWorldXform =
          resetsXformStack ? usdLocalXform : parentWorldXform * usdLocalXform;
      tsd::math::mat4 tsdXform = toTsdMat4(usdLocalXform);
      std::string primName = prim.GetName().GetString();
      if (primName.empty())
        primName = "<unnamed_instance>";
      auto xformNode =
          scene.insertChildTransformNode(parent, tsdXform, primName.c_str());
      importUsdPrimRecursive(scene,
          prototype,
          xformNode,
          xformCache,
          basePath,
          thisWorldXform,
          animMgr,
          matCache,
          texCache);
    } else {
      tsd::core::logStatus("[import_USD] Instance has no prototype: %s\n",
          prim.GetName().GetString().c_str());
    }
    return;
  }

  // Cameras are imported as standalone TSD Camera objects (not scene nodes).
  // importUsdCamera walks the hierarchy itself for animated rigs.
  if (prim.IsA<pxr::UsdGeomCamera>()) {
    importUsdCamera(scene, prim, animMgr);
    return;
  }

  // Only declare these in the main body (non-instance case)
  bool resetsXformStack = false;
  pxr::GfMatrix4d usdLocalXform =
      xformCache.GetLocalTransformation(prim, &resetsXformStack);
  pxr::GfMatrix4d thisWorldXform =
      resetsXformStack ? usdLocalXform : parentWorldXform * usdLocalXform;

  // Determine if this prim is a geometry or light
  bool isGeometry = prim.IsA<pxr::UsdGeomMesh>()
      || prim.IsA<pxr::UsdGeomPoints>() || prim.IsA<pxr::UsdGeomSphere>()
      || prim.IsA<pxr::UsdGeomCone>() || prim.IsA<pxr::UsdGeomCylinder>()
      || prim.IsA<pxr::UsdGeomCube>() || prim.IsA<pxr::UsdGeomBasisCurves>();
  bool isVolume = prim.IsA<pxr::UsdVolVolume>();
  bool isLight = prim.IsA<pxr::UsdLuxDistantLight>()
      || prim.IsA<pxr::UsdLuxRectLight>() || prim.IsA<pxr::UsdLuxSphereLight>()
      || prim.IsA<pxr::UsdLuxDiskLight>() || prim.IsA<pxr::UsdLuxDomeLight>();
  bool isDomeLight = prim.IsA<pxr::UsdLuxDomeLight>();
  bool isXform = prim.IsA<pxr::UsdGeomXform>() || prim.IsA<pxr::UsdGeomScope>();

  // Count children
  size_t numChildren = 0;
  for (const auto &child : prim.GetChildren())
    ++numChildren;

  // For pure xform/scope prims, check for time-sampled animation *before*
  // deciding whether to create a node — an animated xform that happens to be
  // identity at the default time still needs a node.
  std::vector<double> xformTimeSamples;
  if (isXform) {
    pxr::UsdGeomXformable xformable(prim);
    if (xformable)
      xformable.GetTimeSamples(&xformTimeSamples);
  }
  bool hasXformAnimation = xformTimeSamples.size() > 1;

  // Only create a transform node if:
  // - The local transform is not identity
  // - The prim is geometry, light (not dome), or volume
  //   For the domelight, the rationale is the domelight can encode the
  //   transformation in
  //     its orientation axes and at least VisRTX and Barney do not correctly
  //     support transforming the HDRI lights.
  // - The prim resets the xform stack
  // - The prim is an animated pure-xform node
  bool createNode = !isIdentity(usdLocalXform) || isGeometry || isLight
      || isVolume || resetsXformStack || hasXformAnimation;
  createNode = createNode && !isDomeLight;

  tsd::math::mat4 tsdXform = toTsdMat4(usdLocalXform);
  std::string primName = prim.GetName().GetString();
  if (primName.empty())
    primName = "<unnamed_xform>";

  LayerNodeRef thisNode = parent;
  if (createNode) {
    thisNode =
        scene.insertChildTransformNode(parent, tsdXform, primName.c_str());
  }

  // Attach xform animation for pure xform/scope prims with time samples.
  // Geometry prims are excluded — proc shapes bake world-space positions, and
  // mesh vertices are already in local space but we don't yet handle the
  // animated-xform-on-mesh case here.
  if (hasXformAnimation) {
    pxr::UsdGeomXformCache tc;
    std::vector<math::mat4> frames;
    frames.reserve(xformTimeSamples.size());
    for (double t : xformTimeSamples) {
      pxr::UsdTimeCode timeCode(t);
      tc.SetTime(timeCode);
      bool resets = false;
      frames.push_back(toTsdMat4(tc.GetLocalTransformation(prim, &resets)));
    }
    size_t numFrames = frames.size();
    auto tb = makeLinearTimeBase(numFrames);
    auto &anim = animMgr.addAnimation(primName.c_str());
    addTransformStepBinding(anim, thisNode, frames, tb);
    logStatus("[import_USD] Xform '%s': animated transform (%zu frames)\n",
        primName.c_str(),
        numFrames);
  }

  // Import geometry for this prim (if any).
  // Pass identity as the vertex-space transform: geometry data (USD mesh
  // vertices, curve points, implicit shape origins) is always expressed in the
  // prim's own local space.  The transform node created above (tsdXform =
  // usdLocalXform) and its animated parent chain handle all world positioning,
  // so baking world-space positions here would double-apply every ancestor
  // transform.  Lights and volumes are excluded — they have different
  // world-space semantics and are handled separately.
  const pxr::GfMatrix4d identity(1.0);
  if (prim.IsA<pxr::UsdGeomMesh>()) {
    importUsdMesh(
        scene, prim, thisNode, identity, basePath, matCache, texCache);
  } else if (prim.IsA<pxr::UsdGeomPoints>()) {
    importUsdPoints(
        scene, prim, thisNode, identity, animMgr, matCache, texCache);
  } else if (prim.IsA<pxr::UsdGeomSphere>()) {
    importUsdSphere(scene, prim, thisNode, identity, matCache, texCache);
  } else if (prim.IsA<pxr::UsdGeomCone>()) {
    importUsdCone(scene, prim, thisNode, identity, matCache, texCache);
  } else if (prim.IsA<pxr::UsdGeomCylinder>()) {
    importUsdCylinder(scene, prim, thisNode, identity, matCache, texCache);
  } else if (prim.IsA<pxr::UsdGeomCube>()) {
    importUsdCube(
        scene, prim, thisNode, identity, basePath, animMgr, matCache, texCache);
  } else if (prim.IsA<pxr::UsdGeomBasisCurves>()) {
    importUsdCurves(
        scene, prim, thisNode, identity, animMgr, matCache, texCache);
  } else if (prim.IsA<pxr::UsdLuxDistantLight>()) {
    importUsdDistantLight(scene, prim, thisNode);
  } else if (prim.IsA<pxr::UsdLuxRectLight>()) {
    importUsdRectLight(scene, prim, thisNode);
  } else if (prim.IsA<pxr::UsdLuxSphereLight>()) {
    importUsdSphereLight(scene, prim, thisNode);
  } else if (prim.IsA<pxr::UsdLuxDiskLight>()) {
    importUsdDiskLight(scene, prim, thisNode);
  } else if (prim.IsA<pxr::UsdLuxDomeLight>()) {
    importUsdDomeLight(scene, prim, thisNode, basePath, thisWorldXform);
  } else if (prim.IsA<pxr::UsdVolVolume>()) {
    importUsdVolume(scene, prim, thisNode, thisWorldXform);
  }
  // Recurse into children
  for (const auto &child : prim.GetChildren()) {
    importUsdPrimRecursive(scene,
        child,
        thisNode,
        xformCache,
        basePath,
        thisWorldXform,
        animMgr,
        matCache,
        texCache);
  }
}

void import_USD(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filepath,
    LayerNodeRef location)
{
  pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filepath);
  if (!stage) {
    tsd::core::logStatus("[import_USD] failed to open stage '%s'", filepath);
    return;
  }
  tsd::core::logStatus("[import_USD] Opened USD stage: %s\n", filepath);
  auto defaultPrim = stage->GetDefaultPrim();
  if (defaultPrim) {
    tsd::core::logStatus("[import_USD] Default prim: %s\n",
        defaultPrim.GetPath().GetString().c_str());
  } else {
    tsd::core::logStatus("[import_USD] No default prim set.\n");
  }
  size_t primCount = 0;
  for (auto _ : stage->Traverse())
    ++primCount;
  tsd::core::logStatus(
      "[import_USD] Number of prims in stage: %zu\n", primCount);
  auto usd_root = scene.insertChildNode(
      location ? location : scene.defaultLayer()->root(), filepath);

  pxr::UsdGeomXformCache xformCache(pxr::UsdTimeCode::Default());

  std::string basePath = pathOf(filepath);
  MaterialCache matCache;
  TextureCache texCache;

  // Traverse all prims in the USD file, but only import top-level prims
  for (pxr::UsdPrim const &prim : stage->Traverse()) {
    // if (prim.IsPrototype()) continue;
    if (prim.GetParent() && prim.GetParent().IsPseudoRoot()) {
      importUsdPrimRecursive(scene,
          prim,
          usd_root,
          xformCache,
          basePath,
          pxr::GfMatrix4d(1.0),
          animMgr,
          matCache,
          texCache);
    }
  }

  if (!matCache.empty())
    logStatus("[import_USD] Imported %zu unique materials\n", matCache.size());
  if (!texCache.empty())
    logStatus("[import_USD] Loaded %zu unique textures\n", texCache.size());
}
#else
void import_USD(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filepath,
    LayerNodeRef location)
{
  tsd::core::logError("[import_USD] USD not enabled in TSD build.");
}
#endif

} // namespace tsd::io

// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_USD
#define TSD_USE_USD 1
#endif

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#if TSD_USE_USD
// usd
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/cone.h>
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
static MaterialRef import_usd_preview_surface_material(Scene &scene,
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
  std::string matName = usdMat.GetPrim().GetName().GetString();
  if (matName.empty())
    matName = "USDPreviewSurface";
  mat->setName(matName.c_str());
  logStatus("[import_USD] Created material: %s\n", matName.c_str());

  return mat;
}

// Helper to get the bound material for a prim (USD or default)
static MaterialRef get_bound_material(
    Scene &scene, const pxr::UsdPrim &prim, const std::string &basePath)
{
  MaterialRef mat = scene.defaultMaterial();
  pxr::UsdShadeMaterialBindingAPI binding(prim);
  pxr::UsdShadeMaterial usdMat = binding.ComputeBoundMaterial();
  if (usdMat)
    mat = import_usd_preview_surface_material(scene, usdMat, basePath);
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

static VolumeTransferFunction get_volume_transfer_function(
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
inline tsd::math::mat4 to_tsd_mat4(const pxr::GfMatrix4d &m)
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

// Helper: Import a UsdGeomMesh prim as a TSD mesh under the given parent node
static void import_usd_mesh(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    float3 &sceneMin,
    float3 &sceneMax,
    const std::string &basePath)
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
    sceneMin = std::min(sceneMin, wp);
    sceneMax = std::max(sceneMax, wp);
  }

  pxr::VtArray<int> faceVertexIndices;
  mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);
  pxr::VtArray<int> faceVertexCounts;
  mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts);
  pxr::VtArray<pxr::GfVec3f> normals;
  mesh.GetNormalsAttr().Get(&normals);

  // Try to get UV coordinates from primvars
  pxr::UsdGeomPrimvarsAPI primvarsAPI(prim);
  pxr::VtArray<pxr::GfVec2f> uvs;
  pxr::TfToken uvInterpolation;
  bool hasUVs = false;
  
  // Try common UV primvar names
  const char* uvPrimvarNames[] = {"st", "uv", "UVMap"};
  for (const char* uvName : uvPrimvarNames) {
    pxr::UsdGeomPrimvar uvPrimvar = primvarsAPI.GetPrimvar(pxr::TfToken(uvName));
    if (uvPrimvar && uvPrimvar.HasValue()) {
      if (uvPrimvar.Get(&uvs)) {
        uvInterpolation = uvPrimvar.GetInterpolation();
        hasUVs = true;
        logStatus("[import_USD] Mesh '%s': Found UV primvar '%s' with %zu values, interpolation: %s\n",
            prim.GetName().GetString().c_str(),
            uvName,
            uvs.size(),
            uvInterpolation.GetText());
        break;
      }
    }
  }

  logStatus("[import_USD] Mesh '%s': %zu points, %zu faces, %zu normals, %zu UVs\n",
      prim.GetName().GetString().c_str(),
      points.size(),
      faceVertexCounts.size(),
      normals.size(),
      uvs.size());

  std::vector<float3> outVertices;
  std::vector<float3> outNormals;
  std::vector<float2> outUVs;
  size_t index = 0;
  size_t uvIndex = 0;
  
  for (size_t face = 0; face < faceVertexCounts.size(); ++face) {
    int vertsInFace = faceVertexCounts[face];
    for (int v = 2; v < vertsInFace; ++v) {
      int idx0 = faceVertexIndices[index];
      int idx1 = faceVertexIndices[index + v - 1];
      int idx2 = faceVertexIndices[index + v];
      
      // Vertices
      outVertices.push_back(
          float3(points[idx0][0], points[idx0][1], points[idx0][2]));
      outVertices.push_back(
          float3(points[idx1][0], points[idx1][1], points[idx1][2]));
      outVertices.push_back(
          float3(points[idx2][0], points[idx2][1], points[idx2][2]));
      
      // Normals
      if (normals.size() == points.size()) {
        outNormals.push_back(
            float3(normals[idx0][0], normals[idx0][1], normals[idx0][2]));
        outNormals.push_back(
            float3(normals[idx1][0], normals[idx1][1], normals[idx1][2]));
        outNormals.push_back(
            float3(normals[idx2][0], normals[idx2][1], normals[idx2][2]));
      }
      
      // UVs - handle different interpolation modes
      if (hasUVs && !uvs.empty()) {
        if (uvInterpolation == pxr::UsdGeomTokens->faceVarying) {
          // FaceVarying: one UV per face-vertex (most common)
          if (uvIndex + v < uvs.size()) {
            outUVs.push_back(math::float2(uvs[uvIndex][0], uvs[uvIndex][1]));
            outUVs.push_back(math::float2(uvs[uvIndex + v - 1][0], uvs[uvIndex + v - 1][1]));
            outUVs.push_back(math::float2(uvs[uvIndex + v][0], uvs[uvIndex + v][1]));
          }
        } else if (uvInterpolation == pxr::UsdGeomTokens->vertex) {
          // Vertex: one UV per vertex (indexed like positions)
          if (idx0 < (int)uvs.size() && idx1 < (int)uvs.size() && idx2 < (int)uvs.size()) {
            outUVs.push_back(math::float2(uvs[idx0][0], uvs[idx0][1]));
            outUVs.push_back(math::float2(uvs[idx1][0], uvs[idx1][1]));
            outUVs.push_back(math::float2(uvs[idx2][0], uvs[idx2][1]));
          }
        } else if (uvInterpolation == pxr::UsdGeomTokens->uniform) {
          // Uniform: one UV per face
          if (face < uvs.size()) {
            math::float2 faceUV(uvs[face][0], uvs[face][1]);
            outUVs.push_back(faceUV);
            outUVs.push_back(faceUV);
            outUVs.push_back(faceUV);
          }
        }
      }
    }
    
    // Advance indices
    index += vertsInFace;
    if (hasUVs && uvInterpolation == pxr::UsdGeomTokens->faceVarying) {
      uvIndex += vertsInFace;
    }
  }

  auto meshObj = scene.createObject<Geometry>(tokens::geometry::triangle);
  auto vertexPositionArray =
      scene.createArray(ANARI_FLOAT32_VEC3, outVertices.size());
  vertexPositionArray->setData(outVertices.data(), outVertices.size());
  meshObj->setParameterObject("vertex.position", *vertexPositionArray);
  
  if (!outNormals.empty()) {
    auto normalsArray = scene.createArray(ANARI_FLOAT32_VEC3, outNormals.size());
    normalsArray->setData(outNormals.data(), outNormals.size());
    meshObj->setParameterObject("vertex.normal", *normalsArray);
  }
  
  if (!outUVs.empty()) {
    auto uvArray = scene.createArray(ANARI_FLOAT32_VEC2, outUVs.size());
    uvArray->setData(outUVs.data(), outUVs.size());
    meshObj->setParameterObject("vertex.attribute0", *uvArray);
    logStatus("[import_USD] Mesh '%s': Set %zu UV coordinates on vertex.attribute0\n",
        prim.GetName().GetString().c_str(),
        outUVs.size());
  }
  
  std::string primName = prim.GetName().GetString();
  if (primName.empty())
    primName = "<unnamed_mesh>";
  meshObj->setName(primName.c_str());

  // Material binding
  MaterialRef mat = get_bound_material(scene, prim, basePath);

  auto surface = scene.createSurface(primName.c_str(), meshObj, mat);
  logStatus("[import_USD] Assigned material to mesh '%s': %s\n",
      primName.c_str(),
      mat->name().c_str());
  scene.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomPoints prim as a TSD sphere geometry (point cloud)
static void import_usd_points(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    float3 &sceneMin,
    float3 &sceneMax)
{
  pxr::UsdGeomPoints pointsPrim(prim);
  pxr::VtArray<pxr::GfVec3f> points;
  pointsPrim.GetPointsAttr().Get(&points);
  pxr::VtArray<float> widths;
  pointsPrim.GetWidthsAttr().Get(&widths);

  std::vector<float3> outPositions;
  std::vector<float> outRadii;
  for (size_t i = 0; i < points.size(); ++i) {
    pxr::GfVec3f p = points[i];
    pxr::GfVec4d p4(p[0], p[1], p[2], 1.0);
    pxr::GfVec4d wp4 = usdXform * p4;
    float3 wp{float(wp4[0]), float(wp4[1]), float(wp4[2])};
    sceneMin = std::min(sceneMin, wp);
    sceneMax = std::max(sceneMax, wp);
    outPositions.push_back(wp);
    float r = (widths.size() == points.size()) ? widths[i] * 0.5f : 0.01f;
    outRadii.push_back(r);
  }
  auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);
  auto posArray = scene.createArray(ANARI_FLOAT32_VEC3, outPositions.size());
  posArray->setData(outPositions.data(), outPositions.size());
  auto radArray = scene.createArray(ANARI_FLOAT32, outRadii.size());
  radArray->setData(outRadii.data(), outRadii.size());
  geom->setParameterObject("vertex.position", *posArray);
  geom->setParameterObject("vertex.radius", *radArray);
  std::string primName = prim.GetName().GetString();
  if (primName.empty())
    primName = "<unnamed_points>";
  geom->setName(primName.c_str());

  // Material binding
  MaterialRef mat = get_bound_material(scene, prim, "");

  auto surface = scene.createSurface(primName.c_str(), geom, mat);
  logStatus("[import_USD] Assigned material to sphere '%s': %s\n",
      primName.c_str(),
      mat->name().c_str());
  scene.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomSphere prim as a TSD sphere geometry
static void import_usd_sphere(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    float3 &sceneMin,
    float3 &sceneMax)
{
  pxr::UsdGeomSphere spherePrim(prim);
  // UsdGeomSphere is always centered at the origin in local space
  pxr::GfVec3f center(0.f, 0.f, 0.f);
  double radius = 1.0;
  spherePrim.GetRadiusAttr().Get(&radius);
  pxr::GfVec4d c4(center[0], center[1], center[2], 1.0);
  pxr::GfVec4d wc4 = usdXform * c4;
  float3 wp{float(wc4[0]), float(wc4[1]), float(wc4[2])};
  sceneMin = std::min(sceneMin, wp - float3(radius));
  sceneMax = std::max(sceneMax, wp + float3(radius));
  auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);
  auto posArray = scene.createArray(ANARI_FLOAT32_VEC3, 1);
  posArray->setData(&wp, 1);
  auto radArray = scene.createArray(ANARI_FLOAT32, 1);
  float r = float(radius);
  radArray->setData(&r, 1);
  geom->setParameterObject("vertex.position", *posArray);
  geom->setParameterObject("vertex.radius", *radArray);
  std::string primName = prim.GetName().GetString();
  if (primName.empty())
    primName = "<unnamed_sphere>";
  geom->setName(primName.c_str());

  // Material binding
  MaterialRef mat = get_bound_material(scene, prim, "");

  auto surface = scene.createSurface(primName.c_str(), geom, mat);
  logStatus("[import_USD] Assigned material to sphere '%s': %s\n",
      primName.c_str(),
      mat->name().c_str());
  scene.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomCone prim as a TSD cone geometry
static void import_usd_cone(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    float3 &sceneMin,
    float3 &sceneMax)
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
  sceneMin = std::min(sceneMin, wp - float3(radius, radius, height * 0.5f));
  sceneMax = std::max(sceneMax, wp + float3(radius, radius, height * 0.5f));
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
  std::string primName = prim.GetName().GetString();
  if (primName.empty())
    primName = "<unnamed_cone>";
  geom->setName(primName.c_str());

  // Material binding
  MaterialRef mat = get_bound_material(scene, prim, "");

  auto surface = scene.createSurface(primName.c_str(), geom, mat);
  logStatus("[import_USD] Assigned material to cone '%s': %s\n",
      primName.c_str(),
      mat->name().c_str());
  scene.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomCylinder prim as a TSD cylinder geometry
static void import_usd_cylinder(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    float3 &sceneMin,
    float3 &sceneMax)
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
  sceneMin = std::min(sceneMin, wp - float3(radius, radius, height * 0.5f));
  sceneMax = std::max(sceneMax, wp + float3(radius, radius, height * 0.5f));
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
  std::string primName = prim.GetName().GetString();
  if (primName.empty())
    primName = "<unnamed_cylinder>";
  geom->setName(primName.c_str());

  // Material binding
  MaterialRef mat = get_bound_material(scene, prim, "");

  auto surface = scene.createSurface(primName.c_str(), geom, mat);
  tsd::core::logStatus("[import_USD] Assigned material to cylinder '%s': %s\n",
      primName.c_str(),
      mat->name().c_str());
  scene.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdVolVolume prim as a TSD volume geometry
static void import_usd_volume(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const pxr::GfMatrix4d &usdXform,
    float3 &sceneMin,
    float3 &sceneMax)
{
  pxr::UsdVolVolume volumePrim(prim);

  std::string primName = prim.GetName().GetString();
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
  VolumeTransferFunction tf = get_volume_transfer_function(prim);

  ArrayRef colorArray;
  math::float2 valueRange;

  // Create a volume node and assign the field, color map, and value range
  auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
      parent, tokens::volume::transferFunction1D);
  volume->setName(primName.c_str());
  volume->setParameterObject("value", *field);

  if (tf.hasTransferFunction && !tf.colors.empty() && !tf.xPoints.empty()) {
    // Use transfer function from USD material
    colorArray = scene.createArray(ANARI_FLOAT32_VEC4, tf.colors.size());
    colorArray->setData(tf.colors.data(), tf.colors.size());
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
    // Create a default color map
    colorArray = scene.createArray(ANARI_FLOAT32_VEC4, 256);
    colorArray->setData(makeDefaultColorMap(colorArray->size()).data());
  }

  volume->setParameterObject("color", *colorArray);
  volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
}

// -----------------------------------------------------------------------------
// Light import helpers
// -----------------------------------------------------------------------------

static void import_usd_distant_light(
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

static void import_usd_rect_light(
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

static void import_usd_sphere_light(
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

static void import_usd_disk_light(
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

static void import_usd_dome_light(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    const std::string &basePath)
{
  pxr::UsdLuxDomeLight usdLight(prim);
  auto light = scene.createObject<Light>(tokens::light::hdri);
  float intensity = 1.0f;
  usdLight.GetIntensityAttr().Get(&intensity);
  pxr::GfVec3f color(1.0f);
  usdLight.GetColorAttr().Get(&color);
  light->setParameter("color", float3(color[0], color[1], color[2]));
  light->setParameter("scale", intensity);
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
      // Try to import the texture as a sampler
      static TextureCache domeCache;
      auto sampler = importTexture(scene, resolvedPath, domeCache);
      if (sampler)
        light->setParameterObject("image", *sampler);
      else
        tsd::core::logStatus(
            "[import_USD] Warning: Failed to load dome light texture: %s\n",
            resolvedPath.c_str());
    }
  }
  scene.insertChildObjectNode(parent, light);
}

// Helper to check if a GfMatrix4d is identity
static bool is_identity(const pxr::GfMatrix4d &m)
{
  static const pxr::GfMatrix4d IDENTITY(1.0);
  return m == IDENTITY;
}

// -----------------------------------------------------------------------------
// Recursive import function for prims and their children
// -----------------------------------------------------------------------------

static void import_usd_prim_recursive(Scene &scene,
    const pxr::UsdPrim &prim,
    LayerNodeRef parent,
    pxr::UsdGeomXformCache &xformCache,
    float3 &sceneMin,
    float3 &sceneMax,
    const std::string &basePath,
    const pxr::GfMatrix4d &parentWorldXform = pxr::GfMatrix4d(1.0))
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
      tsd::math::mat4 tsdXform = to_tsd_mat4(usdLocalXform);
      std::string primName = prim.GetName().GetString();
      if (primName.empty())
        primName = "<unnamed_instance>";
      auto xformNode =
          scene.insertChildTransformNode(parent, tsdXform, primName.c_str());
      // Recursively import the prototype under this transform node
      import_usd_prim_recursive(scene,
          prototype,
          xformNode,
          xformCache,
          sceneMin,
          sceneMax,
          basePath,
          thisWorldXform);
    } else {
      tsd::core::logStatus("[import_USD] Instance has no prototype: %s\n",
          prim.GetName().GetString().c_str());
    }
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
      || prim.IsA<pxr::UsdGeomCone>() || prim.IsA<pxr::UsdGeomCylinder>();
  bool isVolume = prim.IsA<pxr::UsdVolVolume>();
  bool isLight = prim.IsA<pxr::UsdLuxDistantLight>()
      || prim.IsA<pxr::UsdLuxRectLight>() || prim.IsA<pxr::UsdLuxSphereLight>()
      || prim.IsA<pxr::UsdLuxDiskLight>() || prim.IsA<pxr::UsdLuxDomeLight>();
  bool isXform = prim.IsA<pxr::UsdGeomXform>() || prim.IsA<pxr::UsdGeomScope>();

  // Count children
  size_t numChildren = 0;
  for (const auto &child : prim.GetChildren())
    ++numChildren;

  // Only create a transform node if:
  // - The local transform is not identity
  // - The prim is geometry, light, or volume
  // - The prim resets the xform stack
  bool createNode = !is_identity(usdLocalXform) || isGeometry || isLight
      || isVolume || resetsXformStack;

  tsd::math::mat4 tsdXform = to_tsd_mat4(usdLocalXform);
  std::string primName = prim.GetName().GetString();
  if (primName.empty())
    primName = "<unnamed_xform>";

  LayerNodeRef thisNode = parent;
  if (createNode) {
    thisNode = scene.insertChildTransformNode(parent, tsdXform, primName.c_str());
  }

  // Import geometry for this prim (if any)
  if (prim.IsA<pxr::UsdGeomMesh>()) {
    import_usd_mesh(
        scene, prim, thisNode, thisWorldXform, sceneMin, sceneMax, basePath);
  } else if (prim.IsA<pxr::UsdGeomPoints>()) {
    import_usd_points(scene, prim, thisNode, thisWorldXform, sceneMin, sceneMax);
  } else if (prim.IsA<pxr::UsdGeomSphere>()) {
    import_usd_sphere(scene, prim, thisNode, thisWorldXform, sceneMin, sceneMax);
  } else if (prim.IsA<pxr::UsdGeomCone>()) {
    import_usd_cone(scene, prim, thisNode, thisWorldXform, sceneMin, sceneMax);
  } else if (prim.IsA<pxr::UsdGeomCylinder>()) {
    import_usd_cylinder(
        scene, prim, thisNode, thisWorldXform, sceneMin, sceneMax);
  } else if (prim.IsA<pxr::UsdLuxDistantLight>()) {
    import_usd_distant_light(scene, prim, thisNode);
  } else if (prim.IsA<pxr::UsdLuxRectLight>()) {
    import_usd_rect_light(scene, prim, thisNode);
  } else if (prim.IsA<pxr::UsdLuxSphereLight>()) {
    import_usd_sphere_light(scene, prim, thisNode);
  } else if (prim.IsA<pxr::UsdLuxDiskLight>()) {
    import_usd_disk_light(scene, prim, thisNode);
  } else if (prim.IsA<pxr::UsdLuxDomeLight>()) {
    import_usd_dome_light(scene, prim, thisNode, basePath);
  } else if (prim.IsA<pxr::UsdVolVolume>()) {
    import_usd_volume(scene, prim, thisNode, thisWorldXform, sceneMin, sceneMax);
  }
  // Recurse into children
  for (const auto &child : prim.GetChildren()) {
    import_usd_prim_recursive(scene,
        child,
        thisNode,
        xformCache,
        sceneMin,
        sceneMax,
        basePath,
        thisWorldXform);
  }
}

void import_USD(Scene &scene,
    const char *filepath,
    LayerNodeRef location,
    bool useDefaultMaterial)
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
  for (auto _ : stage->Traverse()) ++primCount;
  tsd::core::logStatus(
      "[import_USD] Number of prims in stage: %zu\n", primCount);
  float3 sceneMin(std::numeric_limits<float>::max());
  float3 sceneMax(std::numeric_limits<float>::lowest());
  auto usd_root = scene.insertChildNode(
      location ? location : scene.defaultLayer()->root(), filepath);

  pxr::UsdGeomXformCache xformCache(pxr::UsdTimeCode::Default());

  std::string basePath = pathOf(filepath);

  // Traverse all prims in the USD file, but only import top-level prims
  for (pxr::UsdPrim const &prim : stage->Traverse()) {
    // if (prim.IsPrototype()) continue;
    if (prim.GetParent() && prim.GetParent().IsPseudoRoot()) {
      import_usd_prim_recursive(
          scene, prim, usd_root, xformCache, sceneMin, sceneMax, basePath);
    }
  }
  tsd::core::logStatus(
      "[import_USD] Scene bounds: min=(%f, %f, %f) max=(%f, %f, %f)\n",
      sceneMin.x,
      sceneMin.y,
      sceneMin.z,
      sceneMax.x,
      sceneMax.y,
      sceneMax.z);
}
#else
void import_USD(Scene &scene,
    const char *filepath,
    LayerNodeRef location,
    bool useDefaultMaterial)
{
  tsd::core::logStatus("[import_USD] USD not enabled in TSD build.");
}
#endif

} // namespace tsd::io

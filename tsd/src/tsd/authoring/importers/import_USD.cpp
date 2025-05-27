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
#include <pxr/usd/usdGeom/points.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/input.h>
#include <pxr/usd/usdShade/output.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#endif
// std
#include <string>
#include <vector>
#include <limits>

namespace tsd {

#if TSD_USE_USD

// -----------------------------------------------------------------------------
// Material-related helpers
// -----------------------------------------------------------------------------

// Template helpers for setting material parameters from USD shader inputs
static void setShaderInputIfPresent(tsd::MaterialRef &mat, pxr::UsdShadeShader &shader, const char *inputName, const char *paramName) {
  pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(inputName));
  pxr::GfVec3f colorVal;
  if (input && input.Get(&colorVal)) {
    printf("[import_USD] Setting %s: %f %f %f\n", paramName, colorVal[0], colorVal[1], colorVal[2]);
    mat->setParameter(tsd::Token(paramName), tsd::float3(colorVal[0], colorVal[1], colorVal[2]));
  }
}

static void setShaderInputIfPresent(tsd::MaterialRef &mat, pxr::UsdShadeShader &shader, const char *inputName, const char *paramName, float) {
  pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(inputName));
  float floatVal;
  if (input && input.Get(&floatVal)) {
    printf("[import_USD] Setting %s: %f\n", paramName, floatVal);
    mat->setParameter(tsd::Token(paramName), floatVal);
  }
}

// Helper: Import a UsdPreviewSurface material as a physicallyBased TSD material
static MaterialRef import_usd_preview_surface_material(
    Context &ctx,
    const pxr::UsdShadeMaterial &usdMat,
    const std::string &basePath)
{
  // Find the UsdPreviewSurface shader
  pxr::UsdShadeShader surfaceShader;
  pxr::TfToken outputName("surface");
  pxr::UsdShadeOutput surfaceOutput = usdMat.GetOutput(outputName);
  
  if (surfaceOutput && surfaceOutput.HasConnectedSource()) {
    printf("[import_USD] Surface output has connected source\n");
    pxr::UsdShadeConnectableAPI source;
    pxr::TfToken sourceName;
    pxr::UsdShadeAttributeType sourceType;
    surfaceOutput.GetConnectedSource(&source, &sourceName, &sourceType);
    surfaceShader = pxr::UsdShadeShader(source.GetPrim());
  } 
  
  if (!surfaceShader) return ctx.defaultMaterial();

  auto mat = ctx.createObject<Material>(tokens::material::physicallyBased);

  setShaderInputIfPresent(mat, surfaceShader, "diffuseColor", "baseColor");
  setShaderInputIfPresent(mat, surfaceShader, "emissiveColor", "emissive");
  setShaderInputIfPresent(mat, surfaceShader, "metallic", "metallic", 0.0f);
  setShaderInputIfPresent(mat, surfaceShader, "roughness", "roughness", 0.0f);
  setShaderInputIfPresent(mat, surfaceShader, "clearcoat", "clearcoat", 0.0f);
  setShaderInputIfPresent(mat, surfaceShader, "clearcoatRoughness", "clearcoatRoughness", 0.0f);
  setShaderInputIfPresent(mat, surfaceShader, "opacity", "opacity", 0.0f);
  setShaderInputIfPresent(mat, surfaceShader, "ior", "ior", 0.0f);

  // Set name
  std::string matName = usdMat.GetPrim().GetName().GetString();
  if (matName.empty()) matName = "USDPreviewSurface";
  mat->setName(matName.c_str());
  printf("[import_USD] Created material: %s\n", matName.c_str());

  return mat;
}

// Helper to get the bound material for a prim (USD or default)
static MaterialRef get_bound_material(Context &ctx, const pxr::UsdPrim &prim, const std::string &basePath) {
  MaterialRef mat = ctx.defaultMaterial();
#if TSD_USE_USD
  pxr::UsdShadeMaterialBindingAPI binding(prim);
  pxr::UsdShadeMaterial usdMat = binding.ComputeBoundMaterial();
  if (usdMat)
    mat = import_usd_preview_surface_material(ctx, usdMat, basePath);
#endif
  return mat;
}



// -----------------------------------------------------------------------------
// Geometry import helpers
// -----------------------------------------------------------------------------

// Helper: Convert pxr::GfMatrix4d to tsd::mat4 (float4x4)
inline tsd::mat4 to_tsd_mat4(const pxr::GfMatrix4d &m)
{
  tsd::mat4 out;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      out[i][j] = static_cast<float>(m[i][j]);
  return out;
}

inline float3 min(const float3 &a, const float3 &b) {
  return float3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

inline float3 max(const float3 &a, const float3 &b) {
  return float3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

// Helper: Import a UsdGeomMesh prim as a TSD mesh under the given parent node
static void import_usd_mesh(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent, const pxr::GfMatrix4d &usdXform, float3 &sceneMin, float3 &sceneMax, const std::string &basePath)
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

  // Material binding
  MaterialRef mat = get_bound_material(ctx, prim, basePath);

  auto surface = ctx.createSurface(primName.c_str(), meshObj, mat);
  printf("[import_USD] Assigned material to mesh '%s': %s\n", primName.c_str(), mat->name().c_str());
  ctx.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomPoints prim as a TSD sphere geometry (point cloud)
static void import_usd_points(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent, const pxr::GfMatrix4d &usdXform, float3 &sceneMin, float3 &sceneMax)
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
    sceneMin = tsd::min(sceneMin, wp);
    sceneMax = tsd::max(sceneMax, wp);
    outPositions.push_back(wp);
    float r = (widths.size() == points.size()) ? widths[i] * 0.5f : 0.01f;
    outRadii.push_back(r);
  }
  auto geom = ctx.createObject<Geometry>(tokens::geometry::sphere);
  auto posArray = ctx.createArray(ANARI_FLOAT32_VEC3, outPositions.size());
  posArray->setData(outPositions.data(), outPositions.size());
  auto radArray = ctx.createArray(ANARI_FLOAT32, outRadii.size());
  radArray->setData(outRadii.data(), outRadii.size());
  geom->setParameterObject("vertex.position", *posArray);
  geom->setParameterObject("vertex.radius", *radArray);
  std::string primName = prim.GetName().GetString();
  if (primName.empty()) primName = "<unnamed_points>";
  geom->setName(primName.c_str());

  // Material binding
  MaterialRef mat = get_bound_material(ctx, prim, "");

  auto surface = ctx.createSurface(primName.c_str(), geom, mat);
  printf("[import_USD] Assigned material to sphere '%s': %s\n", primName.c_str(), mat->name().c_str());
  ctx.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomSphere prim as a TSD sphere geometry
static void import_usd_sphere(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent, const pxr::GfMatrix4d &usdXform, float3 &sceneMin, float3 &sceneMax)
{
  pxr::UsdGeomSphere spherePrim(prim);
  // UsdGeomSphere is always centered at the origin in local space
  pxr::GfVec3f center(0.f, 0.f, 0.f);
  double radius = 1.0;
  spherePrim.GetRadiusAttr().Get(&radius);
  pxr::GfVec4d c4(center[0], center[1], center[2], 1.0);
  pxr::GfVec4d wc4 = usdXform * c4;
  float3 wp{float(wc4[0]), float(wc4[1]), float(wc4[2])};
  sceneMin = tsd::min(sceneMin, wp - float3(radius));
  sceneMax = tsd::max(sceneMax, wp + float3(radius));
  auto geom = ctx.createObject<Geometry>(tokens::geometry::sphere);
  auto posArray = ctx.createArray(ANARI_FLOAT32_VEC3, 1);
  posArray->setData(&wp, 1);
  auto radArray = ctx.createArray(ANARI_FLOAT32, 1);
  float r = float(radius);
  radArray->setData(&r, 1);
  geom->setParameterObject("vertex.position", *posArray);
  geom->setParameterObject("vertex.radius", *radArray);
  std::string primName = prim.GetName().GetString();
  if (primName.empty()) primName = "<unnamed_sphere>";
  geom->setName(primName.c_str());

  // Material binding
  MaterialRef mat = get_bound_material(ctx, prim, "");

  auto surface = ctx.createSurface(primName.c_str(), geom, mat);
  printf("[import_USD] Assigned material to sphere '%s': %s\n", primName.c_str(), mat->name().c_str());
  ctx.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomCone prim as a TSD cone geometry
static void import_usd_cone(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent, const pxr::GfMatrix4d &usdXform, float3 &sceneMin, float3 &sceneMax)
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
  sceneMin = tsd::min(sceneMin, wp - float3(radius, radius, height * 0.5f));
  sceneMax = tsd::max(sceneMax, wp + float3(radius, radius, height * 0.5f));
  // Represent as a 2-point cone (base and apex)
  std::vector<float3> positions = {wp, wp + float3(0, 0, float(height))};
  std::vector<float> radii = {float(radius), 0.f};
  auto geom = ctx.createObject<Geometry>(tokens::geometry::cone);
  auto posArray = ctx.createArray(ANARI_FLOAT32_VEC3, 2);
  posArray->setData(positions.data(), 2);
  auto radArray = ctx.createArray(ANARI_FLOAT32, 2);
  radArray->setData(radii.data(), 2);
  geom->setParameterObject("vertex.position", *posArray);
  geom->setParameterObject("vertex.radius", *radArray);
  std::string primName = prim.GetName().GetString();
  if (primName.empty()) primName = "<unnamed_cone>";
  geom->setName(primName.c_str());

  // Material binding
  MaterialRef mat = get_bound_material(ctx, prim, "");

  auto surface = ctx.createSurface(primName.c_str(), geom, mat);
  printf("[import_USD] Assigned material to cone '%s': %s\n", primName.c_str(), mat->name().c_str());
  ctx.insertChildObjectNode(parent, surface);
}

// Helper: Import a UsdGeomCylinder prim as a TSD cylinder geometry
static void import_usd_cylinder(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent, const pxr::GfMatrix4d &usdXform, float3 &sceneMin, float3 &sceneMax)
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
  sceneMin = tsd::min(sceneMin, wp - float3(radius, radius, height * 0.5f));
  sceneMax = tsd::max(sceneMax, wp + float3(radius, radius, height * 0.5f));
  // Represent as a 2-point cylinder (bottom and top)
  std::vector<float3> positions = {wp - float3(0, 0, float(height) * 0.5f), wp + float3(0, 0, float(height) * 0.5f)};
  std::vector<float> radii = {float(radius), float(radius)};
  auto geom = ctx.createObject<Geometry>(tokens::geometry::cylinder);
  auto posArray = ctx.createArray(ANARI_FLOAT32_VEC3, 2);
  posArray->setData(positions.data(), 2);
  auto radArray = ctx.createArray(ANARI_FLOAT32, 2);
  radArray->setData(radii.data(), 2);
  geom->setParameterObject("vertex.position", *posArray);
  geom->setParameterObject("vertex.radius", *radArray);
  std::string primName = prim.GetName().GetString();
  if (primName.empty()) primName = "<unnamed_cylinder>";
  geom->setName(primName.c_str());

  // Material binding
  MaterialRef mat = get_bound_material(ctx, prim, "");

  auto surface = ctx.createSurface(primName.c_str(), geom, mat);
  printf("[import_USD] Assigned material to cylinder '%s': %s\n", primName.c_str(), mat->name().c_str());
  ctx.insertChildObjectNode(parent, surface);
}

// -----------------------------------------------------------------------------
// Light import helpers
// -----------------------------------------------------------------------------

static void import_usd_distant_light(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent) {
  pxr::UsdLuxDistantLight usdLight(prim);
  auto light = ctx.createObject<Light>(tokens::light::directional);
  float intensity = 1.0f;
  usdLight.GetIntensityAttr().Get(&intensity);
  pxr::GfVec3f color(1.0f);
  usdLight.GetColorAttr().Get(&color);
  light->setParameter("color", float3(color[0], color[1], color[2]));
  light->setParameter("irradiance", intensity);
  // TODO: set direction from transform
  ctx.insertChildObjectNode(parent, light);
}

static void import_usd_rect_light(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent) {
  pxr::UsdLuxRectLight usdLight(prim);
  auto light = ctx.createObject<Light>(tokens::light::quad);
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
  ctx.insertChildObjectNode(parent, light);
}

static void import_usd_sphere_light(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent) {
  pxr::UsdLuxSphereLight usdLight(prim);
  auto light = ctx.createObject<Light>(tokens::light::point);
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
  ctx.insertChildObjectNode(parent, light);
}

static void import_usd_disk_light(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent) {
  pxr::UsdLuxDiskLight usdLight(prim);
  auto light = ctx.createObject<Light>(tokens::light::ring);
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
  ctx.insertChildObjectNode(parent, light);
}

static void import_usd_dome_light(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent, const std::string &basePath) {
  pxr::UsdLuxDomeLight usdLight(prim);
  auto light = ctx.createObject<Light>(tokens::light::hdri);
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
      auto sampler = importTexture(ctx, resolvedPath, domeCache);
      if (sampler)
        light->setParameterObject("image", *sampler);
      else
        printf("[import_USD] Warning: Failed to load dome light texture: %s\n", resolvedPath.c_str());
    }
  }
  ctx.insertChildObjectNode(parent, light);
}

// -----------------------------------------------------------------------------
// Recursive import function for prims and their children
// -----------------------------------------------------------------------------

static void import_usd_prim_recursive(Context &ctx, const pxr::UsdPrim &prim, LayerNodeRef parent, pxr::UsdGeomXformCache &xformCache, float3 &sceneMin, float3 &sceneMax, const std::string &basePath)
{
  if (prim.IsInstance()) {
    pxr::UsdPrim prototype = prim.GetPrototype();
    if (prototype) {
      pxr::GfMatrix4d usdXform = xformCache.GetLocalToWorldTransform(prim);
      tsd::mat4 tsdXform = to_tsd_mat4(usdXform);
      std::string primName = prim.GetName().GetString();
      if (primName.empty()) primName = "<unnamed_instance>";
      auto xformNode = ctx.insertChildTransformNode(parent, tsdXform, primName.c_str());
      // Recursively import the prototype under this transform node
      import_usd_prim_recursive(ctx, prototype, xformNode, xformCache, sceneMin, sceneMax, basePath);
    } else {
      printf("[import_USD] Instance has no prototype: %s\n", prim.GetName().GetString().c_str());
    }
    return;
  }

  // Usual transform node logic
  pxr::GfMatrix4d usdXform = xformCache.GetLocalToWorldTransform(prim);
  tsd::mat4 tsdXform = to_tsd_mat4(usdXform);
  std::string primName = prim.GetName().GetString();
  if (primName.empty()) primName = "<unnamed_xform>";
  auto xformNode = ctx.insertChildTransformNode(parent, tsdXform, primName.c_str());

  // Import geometry for this prim (if any)
  if (prim.IsA<pxr::UsdGeomMesh>()) {
    import_usd_mesh(ctx, prim, xformNode, usdXform, sceneMin, sceneMax, basePath);
  } else if (prim.IsA<pxr::UsdGeomPoints>()) {
    import_usd_points(ctx, prim, xformNode, usdXform, sceneMin, sceneMax);
  } else if (prim.IsA<pxr::UsdGeomSphere>()) {
    import_usd_sphere(ctx, prim, xformNode, usdXform, sceneMin, sceneMax);
  } else if (prim.IsA<pxr::UsdGeomCone>()) {
    import_usd_cone(ctx, prim, xformNode, usdXform, sceneMin, sceneMax);
  } else if (prim.IsA<pxr::UsdGeomCylinder>()) {
    import_usd_cylinder(ctx, prim, xformNode, usdXform, sceneMin, sceneMax);
  } else if (prim.IsA<pxr::UsdLuxDistantLight>()){
    import_usd_distant_light(ctx, prim, xformNode);
  } else if (prim.IsA<pxr::UsdLuxRectLight>()){
    import_usd_rect_light(ctx, prim, xformNode);
  } else if (prim.IsA<pxr::UsdLuxSphereLight>()){
    import_usd_sphere_light(ctx, prim, xformNode);
  } else if (prim.IsA<pxr::UsdLuxDiskLight>()){
    import_usd_disk_light(ctx, prim, xformNode);
  } else if (prim.IsA<pxr::UsdLuxDomeLight>()){
    import_usd_dome_light(ctx, prim, xformNode, basePath);
  }
  // Recurse into children
  for (const auto &child : prim.GetChildren()) {
    import_usd_prim_recursive(ctx, child, xformNode, xformCache, sceneMin, sceneMax, basePath);
  }
}

void import_USD(Context &ctx,
    const char *filepath,
    LayerNodeRef location,
    bool useDefaultMaterial)
{
  pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filepath);
  if (!stage) {
    printf("[import_USD] failed to open stage '%s'", filepath);
    return;
  }
  printf("[import_USD] Opened USD stage: %s\n", filepath);
  auto defaultPrim = stage->GetDefaultPrim();
  if (defaultPrim) {
    printf("[import_USD] Default prim: %s\n", defaultPrim.GetPath().GetString().c_str());
  } else {
    printf("[import_USD] No default prim set.\n");
  }
  size_t primCount = 0;
  for (auto it = stage->Traverse().begin(); it != stage->Traverse().end(); ++it)
    ++primCount;
  printf("[import_USD] Number of prims in stage: %zu\n", primCount);
  float3 sceneMin(std::numeric_limits<float>::max());
  float3 sceneMax(std::numeric_limits<float>::lowest());
  auto usd_root = ctx.insertChildNode(
      location ? location : ctx.defaultLayer()->root(), filepath);

  pxr::UsdGeomXformCache xformCache(pxr::UsdTimeCode::Default());

  std::string basePath = pathOf(filepath);

  // Traverse all prims in the USD file, but only import top-level prims
  for (pxr::UsdPrim const &prim : stage->Traverse()) {
    if (prim.GetParent() && prim.GetParent().IsPseudoRoot()) {
      import_usd_prim_recursive(ctx, prim, usd_root, xformCache, sceneMin, sceneMax, basePath);
    }
  }
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
  printf("[import_USD] USD not enabled in TSD build.");
}
#endif

} // namespace tsd

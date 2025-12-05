// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_USD
#define TSD_USE_USD 1
#endif

// USD helpers
#include "tsd/io/importers/detail/usd/OmniPbrMaterial.h"

// tsd_core
#include <tsd/core/Logging.hpp>
// tsd_io
#include <tsd/io/importers/detail/importer_common.hpp>
#if TSD_USE_USD
// pxr
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdShade/input.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/output.h>
#include <pxr/usd/usdShade/shader.h>
#endif
// std
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace std::string_view_literals;

namespace tsd::io {

using namespace tsd::core;

#if TSD_USE_USD

using namespace pxr;

bool isTimeStepValuesAnimated(const std::vector<Any> &values)
{
  if (values.size() < 2) {
    return false;
  }

  const auto &firstValue = values.front();
  for (const auto &val : values) {
    if (val != firstValue) {
      return true;
    }
  }
  return false;
}

static void importUsdGeomCamera(Scene &scene, const pxr::UsdPrim &prim)
{
  // Import default camera parameters
  std::string primName = prim.GetName().GetString();
  if (primName.empty())
    primName = "<unnamed_camera>";

  // Read camera parameters at default time code
  pxr::UsdGeomCamera usdCamera(prim);
  pxr::GfCamera gfCamera = usdCamera.GetCamera(pxr::UsdTimeCode::Default());

  auto cameraType = gfCamera.GetProjection() == pxr::GfCamera::Perspective
      ? "perspective"
      : "orthographic";

  auto camera = scene.createObject<Camera>(cameraType);
  camera->setName(primName.c_str());

  // Get focal length and sensor dimensions
  float focalLength = gfCamera.GetFocalLength();
  float horizontalAperture = gfCamera.GetHorizontalAperture();
  float verticalAperture = gfCamera.GetVerticalAperture();

  // Calculate field of view from focal length and aperture
  // FOV = 2 * atan(aperture / (2 * focal_length))
  float fovHorizontal =
      2.0f * std::atan(horizontalAperture / (2.0f * focalLength));
  float fovVertical = 2.0f * std::atan(verticalAperture / (2.0f * focalLength));

  // Get f-stop for depth of field
  float fStop = gfCamera.GetFStop();

  // Set ANARI camera parameters
  if (cameraType == "perspective"sv) {
    camera->setParameter("fovy", fovVertical);
    camera->setParameter("aspect", horizontalAperture / verticalAperture);
  }
  // For orthographic camera, set height based on vertical aperture
  else if (cameraType == "orthographic"sv) {
    camera->setParameter("height", verticalAperture);
    camera->setParameter("aspect", horizontalAperture / verticalAperture);
  }

  pxr::UsdGeomXformCache xformCache;
  auto stage = prim.GetStage();
  auto startTc = stage->GetStartTimeCode();
  auto endTc = stage->GetEndTimeCode();

  if (startTc == endTc) {
    // No animation, just set static transform
    xformCache.SetTime(pxr::UsdTimeCode::Default());
    auto xfm = xformCache.GetLocalToWorldTransform(prim);
    auto gfPos = xfm.Transform(pxr::GfVec3d(0, 0, 0));
    camera->setParameter(
        "position", tsd::math::float3(gfPos[0], gfPos[1], gfPos[2]));

    auto gfDir = xfm.TransformDir(pxr::GfVec3d(0, 0, -1));
    camera->setParameter(
        "direction", tsd::math::float3(gfDir[0], gfDir[1], gfDir[2]));

    auto gfUp = xfm.TransformDir(pxr::GfVec3d(0, 1, 0));
    camera->setParameter("up", tsd::math::float3(gfUp[0], gfUp[1], gfUp[2]));

    logStatus("[import_USD] Created static camera '%s'\n", primName.c_str());
    return;
  }

  // Assume that the prim is time sampled
  std::vector<Any> positions;
  std::vector<Any> directions;
  std::vector<Any> ups;
  std::vector<Any> fovs;

  for (auto tc = startTc; tc <= endTc; ++tc) {
    xformCache.SetTime(tc);

    auto xfm = xformCache.GetLocalToWorldTransform(prim);
    auto gfPos = xfm.Transform(pxr::GfVec3d(0, 0, 0));
    positions.push_back(tsd::math::float3(gfPos[0], gfPos[1], gfPos[2]));

    auto gfDir = xfm.TransformDir(pxr::GfVec3d(0, 0, -1));
    directions.push_back(tsd::math::float3(gfDir[0], gfDir[1], gfDir[2]));

    auto gfUp = xfm.TransformDir(pxr::GfVec3d(0, 1, 0));
    ups.push_back(tsd::math::float3(gfUp[0], gfUp[1], gfUp[2]));

    pxr::GfCamera gfCamera = usdCamera.GetCamera(tc);

    // Get focal length and sensor dimensions
    float focalLength = gfCamera.GetFocalLength();
    float verticalAperture = gfCamera.GetVerticalAperture();

    // Calculate field of view from focal length and aperture
    // FOV = 2 * atan(aperture / (2 * focal_length))
    float fovVertical =
        2.0f * std::atan(verticalAperture / (2.0f * focalLength));

    fovs.push_back(fovVertical);
  }

  // Before actually creating the animation object, check if we have varying
  // time samples
  auto isPositionsAnimated = isTimeStepValuesAnimated(positions);
  auto isDirectionsAnimated = isTimeStepValuesAnimated(directions);
  auto isUpsAnimated = isTimeStepValuesAnimated(ups);
  auto isFovsAnimated = isTimeStepValuesAnimated(fovs);

  if (!isPositionsAnimated) {
    camera->setParameter("position", positions.front());
  }
  if (!isDirectionsAnimated) {
    camera->setParameter("direction", directions.front());
  }
  if (!isUpsAnimated) {
    camera->setParameter("up", ups.front());
  }
  if (!isFovsAnimated) {
    camera->setParameter("fovy", fovs.front());
  }

  if (!isPositionsAnimated && !isDirectionsAnimated && !isUpsAnimated
      && !isFovsAnimated) {
    logStatus("[import_USD] Created static camera '%s'\n", primName.c_str());
    return;
  }

  auto cameraAnimation = scene.addAnimation(primName.c_str());
  std::vector<Token> animatedParams;
  std::vector<std::vector<Any>> animatedValues;

  if (isPositionsAnimated) {
    animatedParams.push_back("position");
    animatedValues.push_back(positions);
  }
  if (isDirectionsAnimated) {
    animatedParams.push_back("direction");
    animatedValues.push_back(directions);
  }
  if (isUpsAnimated) {
    animatedParams.push_back("up");
    animatedValues.push_back(ups);
  }
  if (isFovsAnimated) {
    animatedParams.push_back("fovy");
    animatedValues.push_back(fovs);
  }

  std::vector<TimeStepValues> finalValueArrays;
  for (size_t i = 0; i < animatedValues.size(); i++) {
    auto &values = animatedValues[i];
    auto &paramName = animatedParams[i];

    auto type = values[0].type();
    auto arr = scene.createArray(values[0].type(), values.size());
    arr->setName((std::string("animated_") + paramName.c_str()).c_str());

    auto *ptr = (uint8_t *)arr->map();
    for (size_t j = 0; j < values.size(); j++) {
      auto &v = values[j];
      auto *dst = ptr + (j * anari::sizeOf(type));
      std::memcpy(dst, v.data(), anari::sizeOf(type));
    }
    arr->unmap();

    finalValueArrays.push_back(arr);
  }

  cameraAnimation->setAsTimeSteps(*camera, animatedParams, finalValueArrays);

  logStatus("[import_USD] Created camera '%s'\n", primName.c_str());
}

void import_USD2(Scene &scene,
    const char *filename,
    LayerNodeRef location)
{
  // Open the USD stage
  pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filename);
  if (!stage) {
    logStatus("[import_USD2] Failed to open USD stage: %s\n", filename);
    return;
  }

  logStatus("[import_USD2] Opened USD stage: %s\n", filename);

  // Count total prims, cameras, and materials for reporting
  size_t totalPrims = 0;
  size_t cameraCount = 0;
  size_t materialCount = 0;

  // Material cache: USD material path -> TSD MaterialRef
  std::unordered_map<std::string, MaterialRef> materialCache;

  // Extract base path for resolving relative texture paths
  std::string basePath = pathOf(filename);

  // Texture cache for reusing loaded textures
  TextureCache textureCache;

  // Traverse all prims in the stage
  for (pxr::UsdPrim const &prim : stage->Traverse()) {
    totalPrims++;

    // Check if this prim is a camera
    if (prim.IsA<pxr::UsdGeomCamera>()) {
      logStatus("[import_USD2] Found camera: %s\n",
          prim.GetPath().GetString().c_str());

      // Import the camera with time-sampled animation
      importUsdGeomCamera(scene, prim);
      cameraCount++;
    }
    // Check if this prim is a material
    else if (prim.IsA<pxr::UsdShadeMaterial>()) {
      std::string materialPath = prim.GetPath().GetString();
      logStatus("[import_USD2] Found material: %s\n", materialPath.c_str());

      // Detect material type by checking the MDL source asset
      auto usdMaterial = pxr::UsdShadeMaterial(prim);
      auto surfaceOutputs = usdMaterial.GetSurfaceOutputs();
      auto surfaceOutput = usdMaterial.GetSurfaceOutput(pxr::TfToken("mdl"));

      MaterialRef mat;

      for (auto &connectionSourceInfo : surfaceOutput.GetConnectedSources()) {
        UsdShadeShader shader(connectionSourceInfo.source);
        TfToken subIdentifier;
        shader.GetSourceAssetSubIdentifier(&subIdentifier, TfToken("mdl"));

        if (subIdentifier == TfToken("OmniPBR")) {
          mat = materials::importOmniPBRMaterial(
              scene, usdMaterial, shader, basePath, textureCache);
          break;
        } else {
          logStatus("Don't know how to process %s\n", subIdentifier.GetText());
        }
      }

      // Fallback to default if no valid material was created
      if (!mat) {
        mat = scene.defaultMaterial();
      }

      // Cache the material for later use
      materialCache[materialPath] = mat;
      materialCount++;
    }
  }

  logStatus(
      "[import_USD2] Traversed %zu prims, imported %zu cameras and %zu materials\n",
      totalPrims,
      cameraCount,
      materialCount);
  logStatus("[import_USD2] Loaded %zu unique textures\n", textureCache.size());

  // Report imported materials
  if (!materialCache.empty()) {
    logStatus("[import_USD2] Material summary:\n");
    for (const auto &entry : materialCache) {
      logStatus("[import_USD2]   %s -> '%s'\n",
          entry.first.c_str(),
          entry.second->name().c_str());
    }
  }

  // =========================================================================
  // Wire imported materials to existing surfaces
  // =========================================================================

  logStatus("[import_USD2] Wiring materials to surfaces...\n");
  size_t surfacesUpdated = 0;
  size_t surfacesProcessed = 0;

  // Access all surfaces in the scene
  const auto &surfaceDB = scene.objectDB().surface;

  // Iterate through all surfaces
  for (size_t i = 0; i < surfaceDB.size(); ++i) {
    surfacesProcessed++;
    Surface *surface = &surfaceDB[i];

    if (!surface || !surface->geometry()) {
      continue;
    }

    // Get the surface name, which is the matching USD prim path
    auto primPath = pxr::SdfPath(surface->name());

    // Get the USD prim directly by path
    auto prim = stage->GetPrimAtPath(primPath);

    if (!prim.IsValid()) {
      logStatus("[import_USD2]   Surface '%s': Could not find USD prim\n",
          primPath.GetText());
      continue;
    }

    // Check if this prim has a material binding
    if (!prim.HasAPI<pxr::UsdShadeMaterialBindingAPI>()) {
      continue;
    }

    pxr::UsdShadeMaterialBindingAPI binding(prim);
    pxr::UsdShadeMaterial usdMat = binding.ComputeBoundMaterial();

    if (!usdMat) {
      continue;
    }

    // Get the material path
    std::string materialPath = usdMat.GetPath().GetString();

    // Check if we imported this material
    auto matIt = materialCache.find(materialPath);
    if (matIt != materialCache.end()) {
      // Wire the imported material to this surface
      surface->setMaterial(matIt->second);
      surfacesUpdated++;

      logStatus("[import_USD2]   Surface '%s': Assigned material '%s'\n",
          primPath.GetText(),
          matIt->second->name().c_str());
    }
  }

  logStatus(
      "[import_USD2] Material wiring complete: %zu/%zu surfaces updated\n",
      surfacesUpdated,
      surfacesProcessed);
}
#else
void import_USD2(Scene &scene,
    const char *filename,
    LayerNodeRef location,
    bool useDefaultMaterial)
{
  tsd::core::logError("[import_USD2] USD not enabled in TSD build.");
}
#endif

} // namespace tsd::io

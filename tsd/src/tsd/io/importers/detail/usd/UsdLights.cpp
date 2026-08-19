// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers/detail/usd/UsdLights.h"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/io/importers/detail/HDRImage.h"
#include "tsd/io/importers/detail/usd/UsdAnimation.h"
// usd
#include <pxr/base/gf/camera.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/shapingAPI.h>
#include <pxr/usd/usdLux/sphereLight.h>
// std
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace tsd::io::usd {

using namespace tsd::core;

namespace {

// Planckian-locus approximation, used to tint a light by its colour
// temperature the way a reference viewer does.
// https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html
float3 kelvinToRGB(float kelvin)
{
  const float temp = kelvin / 100.0f;
  float red = 1.f;
  float green = 1.f;
  float blue = 1.f;

  if (temp > 66.0f) {
    red = std::clamp(
        329.698727446f * std::pow(temp - 60.0f, -0.1332047592f) / 255.0f,
        0.0f,
        1.0f);
    green = std::clamp(
        288.1221695283f * std::pow(temp - 60.0f, -0.0755148492f) / 255.0f,
        0.0f,
        1.0f);
  } else {
    green =
        std::clamp((99.4708025861f * std::log(temp) - 161.1195681661f) / 255.0f,
            0.0f,
            1.0f);
    blue = temp <= 19.0f
        ? 0.0f
        : std::clamp(
              (138.5177312231f * std::log(temp - 10.0f) - 305.0447927307f)
                  / 255.0f,
              0.0f,
              1.0f);
  }

  return float3(red, green, blue);
}

// An imported Stage is systematically the wrong brightness unless exposure,
// normalization, and colour temperature all reach the emitted light.
struct LightRadiometry
{
  float3 color{1.f, 1.f, 1.f};
  float intensity{1.f};
};

LightRadiometry readRadiometry(const pxr::UsdPrim &prim, float area)
{
  LightRadiometry retval;
  pxr::UsdLuxLightAPI light(prim);
  if (!light)
    return retval;

  float intensity = 1.f;
  light.GetIntensityAttr().Get(&intensity);

  float exposure = 0.f;
  light.GetExposureAttr().Get(&exposure);
  intensity *= std::pow(2.0f, exposure);

  bool normalize = false;
  light.GetNormalizeAttr().Get(&normalize);
  if (normalize && area > 0.f)
    intensity /= area;

  pxr::GfVec3f color(1.f);
  light.GetColorAttr().Get(&color);
  retval.color = float3(color[0], color[1], color[2]);

  bool enableColorTemperature = false;
  light.GetEnableColorTemperatureAttr().Get(&enableColorTemperature);
  if (enableColorTemperature) {
    float colorTemperature = 6500.f;
    light.GetColorTemperatureAttr().Get(&colorTemperature);
    if (colorTemperature > 0.f)
      retval.color *= kelvinToRGB(colorTemperature);
  }

  retval.intensity = intensity;
  return retval;
}

// Light types differ only in subtype and in which parameter their brightness
// lands on, so those are what a caller passes. The dome light is the exception
// and stays hand-built: it has no `color`, its colour being baked into the
// radiance it maps over the sphere.
LightRef makeLight(ImportContext &ctx,
    const Token &subtype,
    const pxr::SdfPath &primPath,
    const LightRadiometry &radiometry,
    const char *brightnessParameter)
{
  auto retval = ctx.scene->createObject<Light>(subtype);
  retval->setName(primPath.GetName().c_str());
  retval->setParameter("color", radiometry.color);
  retval->setParameter(brightnessParameter, radiometry.intensity);
  return retval;
}

// A sphere or disk light carrying shaping attributes is a spot light; USD
// expresses the cone as a half-angle plus a softness fraction.
bool readShaping(
    const pxr::UsdPrim &prim, float *openingAngle, float *falloffAngle)
{
  pxr::UsdLuxShapingAPI shaping(prim);
  if (!shaping || !prim.HasAPI<pxr::UsdLuxShapingAPI>())
    return false;

  auto coneAngleAttr = shaping.GetShapingConeAngleAttr();
  if (!coneAngleAttr || !coneAngleAttr.HasAuthoredValue())
    return false;

  float coneAngle = 90.f;
  coneAngleAttr.Get(&coneAngle);
  float softness = 0.f;
  if (auto softnessAttr = shaping.GetShapingConeSoftnessAttr())
    softnessAttr.Get(&softness);

  *openingAngle = 2.f * coneAngle * float(M_PI) / 180.f;
  *falloffAngle = std::clamp(softness, 0.f, 1.f) * 0.5f * *openingAngle;
  return true;
}

ArrayRef readDomeRadiance(ImportContext &ctx,
    const pxr::UsdLuxDomeLight &domeLight,
    const LightRadiometry &radiometry)
{
  pxr::SdfAssetPath textureAsset;
  if (!domeLight.GetTextureFileAttr().Get(&textureAsset))
    return {};

  auto file = textureAsset.GetResolvedPath();
  if (file.empty())
    file = textureAsset.GetAssetPath();
  if (file.empty())
    return {};
  if (!isAbsolute(file))
    file = ctx.basePath + file;

  HDRImage image;
  if (!image.import(file)) {
    ctx.reportSkip(domeLight.GetPrim().GetPath(),
        "domeLight",
        UsdSkipReason::TEXTURE_LOAD_FAILED,
        file);
    return {};
  }

  std::vector<float3> rgb(size_t(image.width) * image.height);
  if (image.numComponents == 3) {
    std::memcpy(rgb.data(), image.pixel.data(), sizeof(rgb[0]) * rgb.size());
  } else if (image.numComponents == 4) {
    for (size_t i = 0; i < image.pixel.size(); i += 4)
      rgb[i / 4] =
          float3(image.pixel[i], image.pixel[i + 1], image.pixel[i + 2]);
  } else {
    return {};
  }

  for (auto &texel : rgb)
    texel *= radiometry.color;

  // Keyed on the radiometry as well as the file: the scale above is baked
  // into the texels, so two dome lights sharing a file but not a colour are
  // genuinely different images.
  const auto id = "usd:domelight:" + file + ":"
      + std::to_string(radiometry.color.x) + ","
      + std::to_string(radiometry.color.y) + ","
      + std::to_string(radiometry.color.z);
  // Stored bottom-up: an hdri light's radiance is mapped over the sphere by
  // the light rather than addressed by an image sampler, so the top-left
  // origin samplers are stored for does not apply to it.
  auto acquired = ctx.textureCache.acquireDecoded(
      {id, ColorSpace::LINEAR, RowOrder::BOTTOM_UP},
      ANARI_FLOAT32_VEC3,
      image.width,
      image.height,
      image.rowOrder,
      rgb.data());
  return acquired.texels;
}

} // namespace

bool isLightPrimType(const pxr::TfToken &primType)
{
  return pxr::HdPrimTypeIsLight(primType);
}

LightRef convertLight(ImportContext &ctx,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    std::string *skipDetail)
{
  // Light radiometry is modelled by UsdLux itself, so it is read from the
  // retained Stage rather than re-derived from the resolved prim.
  auto usdPrim = ctx.stage->GetPrimAtPath(primPath);
  if (!usdPrim) {
    *skipDetail =
        "light has no Stage prim (instanced lights are not supported)";
    return {};
  }

  const auto &type = prim.primType;

  if (type == pxr::HdPrimTypeTokens->distantLight) {
    const auto radiometry = readRadiometry(usdPrim, 0.f);
    return makeLight(
        ctx, tokens::light::directional, primPath, radiometry, "irradiance");
  }

  if (type == pxr::HdPrimTypeTokens->rectLight) {
    pxr::UsdLuxRectLight rectLight(usdPrim);
    float width = 1.f;
    float height = 1.f;
    rectLight.GetWidthAttr().Get(&width);
    rectLight.GetHeightAttr().Get(&height);
    const auto radiometry = readRadiometry(usdPrim, width * height);

    auto light =
        makeLight(ctx, tokens::light::quad, primPath, radiometry, "intensity");
    light->setParameter("position", float3(-0.5f * width, -0.5f * height, 0.f));
    light->setParameter("edge1", float3(width, 0.f, 0.f));
    light->setParameter("edge2", float3(0.f, height, 0.f));
    return light;
  }

  if (type == pxr::HdPrimTypeTokens->sphereLight
      || type == pxr::HdPrimTypeTokens->diskLight) {
    const bool isDisk = type == pxr::HdPrimTypeTokens->diskLight;
    float radius = 0.5f;
    if (isDisk)
      pxr::UsdLuxDiskLight(usdPrim).GetRadiusAttr().Get(&radius);
    else
      pxr::UsdLuxSphereLight(usdPrim).GetRadiusAttr().Get(&radius);

    const float area = isDisk ? float(M_PI) * radius * radius
                              : 4.f * float(M_PI) * radius * radius;
    const auto radiometry = readRadiometry(usdPrim, area);

    float openingAngle = 0.f;
    float falloffAngle = 0.f;
    if (readShaping(usdPrim, &openingAngle, &falloffAngle)) {
      auto light = makeLight(
          ctx, tokens::light::spot, primPath, radiometry, "intensity");
      light->setParameter("openingAngle", openingAngle);
      light->setParameter("falloffAngle", falloffAngle);
      return light;
    }

    auto light = makeLight(ctx,
        isDisk ? tokens::light::ring : tokens::light::point,
        primPath,
        radiometry,
        "intensity");
    light->setParameter("radius", radius);
    return light;
  }

  if (type == pxr::HdPrimTypeTokens->domeLight) {
    pxr::UsdLuxDomeLight domeLight(usdPrim);
    const auto radiometry = readRadiometry(usdPrim, 0.f);

    auto light = ctx.scene->createObject<Light>(tokens::light::hdri);
    light->setName(primPath.GetName().c_str());
    light->setParameter("scale", radiometry.intensity);

    // Dome orientation is baked into the light's own direction and up rather
    // than left to a transform, because devices mishandle transformed dome
    // lights (and no corrective root transform is inserted for up-axis).
    pxr::UsdGeomXformCache xformCache(ctx.importTime);
    auto worldXform = xformCache.GetLocalToWorldTransform(usdPrim);
    auto orientation = pxr::GfMatrix4d(
        // clang-format off
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
        // clang-format on
    );
    orientation *= worldXform;
    const auto direction = orientation.TransformDir(pxr::GfVec3d(0, 0, -1));
    const auto up = orientation.TransformDir(pxr::GfVec3d(0, 1, 0));
    light->setParameter(
        "direction", float3(direction[0], direction[1], direction[2]));
    light->setParameter("up", float3(up[0], up[1], up[2]));

    auto radiance = readDomeRadiance(ctx, domeLight, radiometry);
    if (!radiance) {
      // Devices require radiance to be set; synthesize a constant environment
      // from the light's own colour so an untextured dome still lights.
      const float3 solid = radiometry.color * radiometry.intensity;
      radiance = ctx.scene->createArray(ANARI_FLOAT32_VEC3, 1, 1);
      radiance->setData(&solid, 1);
    }
    light->setParameterObject("radiance", *radiance);
    return light;
  }

  return {};
}

///////////////////////////////////////////////////////////////////////////////
// Cameras ////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void convertCamera(ImportContext &ctx, const pxr::SdfPath &primPath)
{
  auto usdPrim = ctx.stage->GetPrimAtPath(primPath);
  if (!usdPrim)
    return;

  pxr::UsdGeomCamera usdCamera(usdPrim);
  if (!usdCamera)
    return;

  const auto name = primPath.GetName();
  const auto defaultCamera = usdCamera.GetCamera(ctx.importTime);
  const bool isPerspective =
      defaultCamera.GetProjection() == pxr::GfCamera::Perspective;

  auto camera = ctx.scene->createObject<Camera>(isPerspective
          ? tokens::camera::perspective
          : tokens::camera::orthographic);
  camera->setName(name.c_str());

  auto setIntrinsics = [&](const pxr::GfCamera &gfCamera) {
    const float verticalAperture = gfCamera.GetVerticalAperture();
    const float horizontalAperture = gfCamera.GetHorizontalAperture();
    const float aspect =
        verticalAperture > 0.f ? horizontalAperture / verticalAperture : 1.f;
    if (isPerspective) {
      const float focalLength = gfCamera.GetFocalLength();
      camera->setParameter("fovy",
          focalLength > 0.f
              ? 2.f * std::atan(verticalAperture / (2.f * focalLength))
              : 1.f);
    } else {
      camera->setParameter("height", verticalAperture);
    }
    camera->setParameter("aspect", aspect);
  };
  setIntrinsics(defaultCamera);

  auto poseAt = [&](pxr::UsdGeomXformCache &cache) {
    const auto xform = cache.GetLocalToWorldTransform(usdPrim);
    auto position = xform.Transform(pxr::GfVec3d(0, 0, 0));
    auto direction = xform.TransformDir(pxr::GfVec3d(0, 0, -1)).GetNormalized();
    auto up = xform.TransformDir(pxr::GfVec3d(0, 1, 0)).GetNormalized();
    return std::make_tuple(float3(position[0], position[1], position[2]),
        float3(direction[0], direction[1], direction[2]),
        float3(up[0], up[1], up[2]));
  };

  {
    pxr::UsdGeomXformCache cache(ctx.importTime);
    auto [position, direction, up] = poseAt(cache);
    camera->setParameter("position", position);
    camera->setParameter("direction", direction);
    camera->setParameter("up", up);
  }

  // Animation is captured at the times actually authored -- anywhere in the
  // rig above the camera, so orbit and crane rigs animate even when the camera
  // prim itself is static.
  std::vector<double> sampleTimes;
  for (auto current = usdPrim; current && !current.IsPseudoRoot();
       current = current.GetParent()) {
    pxr::UsdGeomXformable xformable(current);
    if (!xformable)
      continue;
    std::vector<double> times;
    xformable.GetTimeSamples(&times);
    sampleTimes.insert(sampleTimes.end(), times.begin(), times.end());
  }

  std::vector<double> intrinsicTimes;
  for (auto attribute : {usdCamera.GetFocalLengthAttr(),
           usdCamera.GetHorizontalApertureAttr(),
           usdCamera.GetVerticalApertureAttr(),
           usdCamera.GetFStopAttr(),
           usdCamera.GetFocusDistanceAttr()}) {
    std::vector<double> times;
    if (attribute)
      attribute.GetTimeSamples(&times);
    intrinsicTimes.insert(intrinsicTimes.end(), times.begin(), times.end());
  }
  const bool hasIntrinsicAnimation = !intrinsicTimes.empty();
  sampleTimes.insert(
      sampleTimes.end(), intrinsicTimes.begin(), intrinsicTimes.end());

  std::sort(sampleTimes.begin(), sampleTimes.end());
  sampleTimes.erase(
      std::unique(sampleTimes.begin(), sampleTimes.end()), sampleTimes.end());
  if (sampleTimes.size() < 2)
    return;

  const size_t frameCount = sampleTimes.size();
  auto positions = ctx.scene->createArray(ANARI_FLOAT32_VEC3, frameCount);
  auto directions = ctx.scene->createArray(ANARI_FLOAT32_VEC3, frameCount);
  auto ups = ctx.scene->createArray(ANARI_FLOAT32_VEC3, frameCount);

  ArrayRef fovs, aspects;
  if (hasIntrinsicAnimation) {
    fovs = ctx.scene->createArray(ANARI_FLOAT32, frameCount);
    aspects = ctx.scene->createArray(ANARI_FLOAT32, frameCount);
  }

  auto *positionData = positions->mapAs<float3>();
  auto *directionData = directions->mapAs<float3>();
  auto *upData = ups->mapAs<float3>();
  float *fovData = fovs ? fovs->mapAs<float>() : nullptr;
  float *aspectData = aspects ? aspects->mapAs<float>() : nullptr;

  pxr::UsdGeomXformCache cache;
  for (size_t i = 0; i < frameCount; ++i) {
    const pxr::UsdTimeCode time(sampleTimes[i]);
    cache.SetTime(time);
    auto [position, direction, up] = poseAt(cache);
    positionData[i] = position;
    directionData[i] = direction;
    upData[i] = up;
    if (fovData) {
      const auto gfCamera = usdCamera.GetCamera(time);
      const float verticalAperture = gfCamera.GetVerticalAperture();
      const float focalLength = gfCamera.GetFocalLength();
      fovData[i] = focalLength > 0.f
          ? 2.f * std::atan(verticalAperture / (2.f * focalLength))
          : 1.f;
      aspectData[i] = verticalAperture > 0.f
          ? gfCamera.GetHorizontalAperture() / verticalAperture
          : 1.f;
    }
  }

  positions->unmap();
  directions->unmap();
  ups->unmap();
  if (fovs) {
    fovs->unmap();
    aspects->unmap();
  }

  std::vector<Token> parameterNames{"position", "direction", "up"};
  std::vector<ObjectUsePtr<Array>> parameterArrays{positions, directions, ups};
  if (hasIntrinsicAnimation) {
    parameterNames.push_back("fovy");
    parameterArrays.push_back(fovs);
    parameterNames.push_back("aspect");
    parameterArrays.push_back(aspects);
  }

  // The authored times are the binding's own time base, rescaled onto the
  // Stage's clock so this camera shares one clock with every other binding
  // from the same import. Nothing is resampled.
  const auto timeBase = normalizeSampleTimes(ctx.stage, sampleTimes);
  auto &animation = ctx.animMgr->addAnimation(name);
  addValueTimeStepBindings(animation,
      camera.data(),
      parameterNames,
      parameterArrays,
      timeBase,
      tsd::animation::InterpolationRule::LINEAR);
}

} // namespace tsd::io::usd

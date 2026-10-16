// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Reuse the existing public-object ownership and checked mapping support. Fog
// scenes and receiver scenes otherwise keep their own independent fixtures.
#include "RendererFogSupport.h"

#include <anari/anari_cpp/ext/std.h>

namespace visrtx::test {

using fogtest::exitOnErrorStatus;
using fogtest::FrameMapping;
using fogtest::ObjectOwner;
using Vec4 = std::array<float, 4>;
using Size = std::array<unsigned, 2>;

// Fixed diffuse XZ receiver at y=0. The caller owns the returned Surface.
inline anari::Surface receiverFloor(
    anari::Device device, float reflectance = 0.6f)
{
  const std::array<std::array<float, 3>, 4> pos = {
      std::array<float, 3>{-6.f, 0.f, -6.f},
      std::array<float, 3>{6.f, 0.f, -6.f},
      std::array<float, 3>{6.f, 0.f, 6.f},
      std::array<float, 3>{-6.f, 0.f, 6.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};

  auto geom = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(device, geom, "vertex.position", pos.data(), 4);
  anari::setParameterArray1D(device, geom, "primitive.index", idx.data(), 2);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device,
      mat,
      "baseColor",
      std::array<float, 3>{reflectance, reflectance, reflectance});
  anari::setParameter(device, mat, "metallic", 0.f);
  anari::setParameter(device, mat, "roughness", 1.f);
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);
  return surface;
}

/* Fixed receiver view shared by the geometry-light radiometry tests. The
 * caller keeps its World reference; the returned Frame owns camera/Renderer.
 * This deliberately does not construct emitters, references or a scene graph.
 */
inline anari::Frame receiverFrame(anari::Device device,
    anari::World world,
    Size size,
    const char *rendererSubtype,
    int pixelSamples)
{
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  const ObjectOwner cameraOwner(device, camera);
  anari::setParameter(
      device, camera, "position", std::array<float, 3>{0.f, 0.5f, -3.f});
  anari::setParameter(
      device, camera, "direction", std::array<float, 3>{0.f, -0.15f, 1.f});
  anari::setParameter(
      device, camera, "up", std::array<float, 3>{0.f, 1.f, 0.f});
  anari::setParameter(device, camera, "aspect", size[0] / float(size[1]));
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, rendererSubtype);
  const ObjectOwner rendererOwner(device, renderer);
  anari::setParameter(device, renderer, "background", Vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", pixelSamples);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  ObjectOwner frameOwner(device, frame);
  anari::setParameter(device, frame, "size", size);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);
  frameOwner.release();
  return frame;
}

/* Half-open pixel bounds stay at the assertion site: they define the measured
 * physical receiver, not a configurable scene or an implicit tolerance.
 */
struct PixelRegion
{
  unsigned xBegin, xEnd, yBegin, yEnd;
};

inline double meanLuminance(const Vec4 *pixels, Size size, PixelRegion region)
{
  if (region.xBegin >= region.xEnd || region.xEnd > size[0]
      || region.yBegin >= region.yEnd || region.yEnd > size[1])
    throw std::runtime_error("Invalid receiver measurement region");
  double sum = 0.;
  for (unsigned y = region.yBegin; y < region.yEnd; ++y)
    for (unsigned x = region.xBegin; x < region.xEnd; ++x) {
      const auto &p = pixels[std::size_t(y) * size[0] + x];
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    }
  return sum
      / (std::size_t(region.xEnd - region.xBegin)
          * (region.yEnd - region.yBegin));
}

inline double receiverLuminance(
    anari::Device device, anari::Frame frame, Size size, PixelRegion region)
{
  anari::render(device, frame);
  anari::wait(device, frame);
  const FrameMapping<Vec4> mapping(device, frame, "channel.color");
  return meanLuminance(
      mapping.checkedData(size[0], size[1], ANARI_FLOAT32_VEC4), size, region);
}

} // namespace visrtx::test

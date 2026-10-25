// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

#include "../MdlModuleReuseFixture.h"

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;
static unsigned reuseCount = 0;

static void check(bool condition, const char *message)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR
      || severity == ANARI_SEVERITY_ERROR) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
  if (std::strstr(message, "Reusing MDL OptiX module for slot"))
    ++reuseCount;
}

static anari::Surface makeSurface(ANARIDevice device, int variant)
{
  const float x = variant == 0 ? -0.75f : 0.75f;
  const std::array<Vec3, 4> vertices = {Vec3{x - 0.6f, -0.6f, 0.f},
      Vec3{x + 0.6f, -0.6f, 0.f},
      Vec3{x + 0.6f, 0.6f, 0.f},
      Vec3{x - 0.6f, 0.6f, 0.f}};
  const std::array<std::array<unsigned, 3>, 2> indices = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};
  auto geometry = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setParameterArray1D(
      device, geometry, "vertex.position", vertices.data(), vertices.size());
  anari::setParameterArray1D(
      device, geometry, "primitive.index", indices.data(), indices.size());
  anari::commitParameters(device, geometry);
  auto material = anari::newObject<anari::Material>(device, "mdl");
  anari::setParameter(device, material, "sourceType", "code");
  anari::setParameter(device, material, "source", moduleReuseSource(variant));
  anari::setParameter(device, material, "materialName", "test");
  const auto name = std::string("tint") + std::to_string(variant);
  anari::setParameter(device,
      material,
      name.c_str(),
      variant == 0 ? Vec3{4.f, 0.f, 0.f} : Vec3{0.f, 4.f, 0.f});
  anari::commitParameters(device, material);
  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);
  return surface;
}

int main()
{
  auto device = makeVisRTXDevice(status);
  auto camera = anari::newObject<anari::Camera>(device, "orthographic");
  anari::setParameter(device, camera, "position", Vec3{0.f, 0.f, 3.f});
  anari::setParameter(device, camera, "direction", Vec3{0.f, 0.f, -1.f});
  anari::setParameter(device, camera, "up", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(device, camera, "height", 2.f);
  anari::setParameter(device, camera, "aspect", 2.f);
  anari::commitParameters(device, camera);
  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "background", Vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(device, renderer, "pixelSamples", 1);
  anari::setParameter(device, renderer, "fireflyFilterMode", "none");
  anari::commitParameters(device, renderer);
  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", std::array<unsigned, 2>{64, 32});
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);

  auto render = [&](const std::array<anari::Surface, 2> &surfaces,
                    uint32_t expectedCount) {
    std::vector<anari::Surface> live;
    for (auto surface : surfaces)
      if (surface)
        live.push_back(surface);
    auto world = anari::newObject<anari::World>(device);
    if (!live.empty())
      anari::setParameterArray1D(
          device, world, "surface", live.data(), live.size());
    anari::commitParameters(device, world);
    anari::setAndReleaseParameter(device, frame, "world", world);
    anari::commitParameters(device, frame);
    anari::render(device, frame);
    anari::wait(device, frame);
    uint32_t count = ~0u;
    check(anari::getProperty(
              device, device, "numRegisteredMdlMaterials", count, ANARI_WAIT),
        "missing registry count");
    check(count == expectedCount,
        "unexpected UUID count (fallback or leaked material)");
    auto pixels = anari::map<Vec4>(device, frame, "channel.color");
    check(pixels.data != nullptr, "missing framebuffer");
    for (int i = 0; i < 2; ++i) {
      const auto &pixel = pixels.data[16 * 64 + (i == 0 ? 20 : 44)];
      for (int c = 0; c < 3; ++c) {
        check(std::isfinite(pixel[c]), "nonfinite pixel");
        if (surfaces[i] && c == i)
          check(pixel[c] > 0.1f, "shared module lost material-specific tint");
        else
          check(std::abs(pixel[c]) < 0.001f,
              "wrong tint or removed surface still visible");
      }
    }
    anari::unmap(device, frame, "channel.color");
  };

  std::array<anari::Surface, 2> surfaces{};
  render(surfaces, 0);
  // Exercise both representative/alias release orders. Empty slots are reused
  // on re-acquisition, forcing the same renderer to rebuild shared modules.
  for (int first : {0, 1}) {
    surfaces = {makeSurface(device, 0), makeSurface(device, 1)};
    render(surfaces, 2);
    for (int removed : {first, 1 - first}) {
      auto dead = surfaces[removed];
      surfaces[removed] = nullptr;
      // Detach the old world before releasing our final surface reference.
      render(surfaces, removed == first ? 2 : 1);
      anari::release(device, dead);
      render(surfaces, removed == first ? 1 : 0);
    }
  }
  check(reuseCount >= 2, "test did not exercise OptiX module sharing");
  // Leave shared modules alive for device/frame teardown too.
  surfaces = {makeSurface(device, 0), makeSurface(device, 1)};
  render(surfaces, 2);
  for (auto surface : surfaces)
    anari::release(device, surface);
  anari::release(device, frame);
  anari::release(device, device);
  std::puts("MDL module reuse, independent tints, release and rebuild passed");
}

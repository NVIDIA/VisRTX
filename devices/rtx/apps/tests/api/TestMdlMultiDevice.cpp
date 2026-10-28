// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

// `mi_factory` hands out a single live INeuray per loaded library image, so
// devices must share one refcounted SDK instance rather than each acquiring
// their own. This covers overlapping lifetimes, sequential create/destroy
// cycles, and concurrent creation.

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;

namespace {

void check(bool condition, const char *message)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR
      || severity == ANARI_SEVERITY_ERROR) {
    std::fprintf(stderr, "ERROR: %s\n", message);
    std::exit(1);
  }
}

// A module whose single material emits `channel` at full intensity, so the
// rendered pixel identifies which module and which device produced it.
std::string moduleSource(int channel)
{
  const Vec3 intensity = {channel == 0 ? 4.f : 0.f,
      channel == 1 ? 4.f : 0.f,
      channel == 2 ? 4.f : 0.f};
  return "mdl 1.6; import ::df::*; export material emitter() = material("
         "surface: material_surface(emission: material_emission("
         "emission: df::diffuse_edf(), intensity: color("
      + std::to_string(intensity[0]) + ", " + std::to_string(intensity[1])
      + ", " + std::to_string(intensity[2]) + "))));";
}

struct Setup
{
  anari::Device device{};
  anari::Frame frame{};
};

// Builds a device rendering a full-screen quad with the given MDL material.
Setup makeSetup(int channel)
{
  Setup setup;
  setup.device = makeVisRTXDevice(status);

  const std::array<Vec3, 4> vertices = {Vec3{-1.f, -1.f, 0.f},
      Vec3{1.f, -1.f, 0.f},
      Vec3{1.f, 1.f, 0.f},
      Vec3{-1.f, 1.f, 0.f}};
  const std::array<std::array<unsigned, 3>, 2> indices = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};
  auto geometry = anari::newObject<anari::Geometry>(setup.device, "triangle");
  anari::setParameterArray1D(setup.device,
      geometry,
      "vertex.position",
      vertices.data(),
      vertices.size());
  anari::setParameterArray1D(setup.device,
      geometry,
      "primitive.index",
      indices.data(),
      indices.size());
  anari::commitParameters(setup.device, geometry);

  auto material = anari::newObject<anari::Material>(setup.device, "mdl");
  anari::setParameter(setup.device, material, "sourceType", "code");
  anari::setParameter(setup.device, material, "source", moduleSource(channel));
  anari::setParameter(setup.device, material, "materialName", "emitter");
  anari::commitParameters(setup.device, material);

  auto surface = anari::newObject<anari::Surface>(setup.device);
  anari::setAndReleaseParameter(setup.device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(setup.device, surface, "material", material);
  anari::commitParameters(setup.device, surface);

  auto world = anari::newObject<anari::World>(setup.device);
  anari::setParameterArray1D(setup.device, world, "surface", &surface, 1);
  anari::commitParameters(setup.device, world);
  anari::release(setup.device, surface);

  auto camera = anari::newObject<anari::Camera>(setup.device, "orthographic");
  anari::setParameter(setup.device, camera, "position", Vec3{0.f, 0.f, 3.f});
  anari::setParameter(setup.device, camera, "direction", Vec3{0.f, 0.f, -1.f});
  anari::setParameter(setup.device, camera, "up", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(setup.device, camera, "height", 2.f);
  anari::setParameter(setup.device, camera, "aspect", 1.f);
  anari::commitParameters(setup.device, camera);

  auto renderer = anari::newObject<anari::Renderer>(setup.device, "quality");
  anari::setParameter(
      setup.device, renderer, "background", Vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(setup.device, renderer, "ambientRadiance", 0.f);
  anari::setParameter(setup.device, renderer, "pixelSamples", 1);
  anari::setParameter(setup.device, renderer, "fireflyFilterMode", "none");
  anari::commitParameters(setup.device, renderer);

  setup.frame = anari::newObject<anari::Frame>(setup.device);
  anari::setParameter(
      setup.device, setup.frame, "size", std::array<unsigned, 2>{32, 32});
  anari::setParameter(
      setup.device, setup.frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(setup.device, setup.frame, "camera", camera);
  anari::setAndReleaseParameter(
      setup.device, setup.frame, "renderer", renderer);
  anari::setAndReleaseParameter(setup.device, setup.frame, "world", world);
  anari::commitParameters(setup.device, setup.frame);
  return setup;
}

// Renders and asserts the quad emits on `channel` only: an MDL failure would
// fall back to a different material or leave the frame black.
void renderAndCheck(const Setup &setup, int channel, const char *context)
{
  anari::render(setup.device, setup.frame);
  anari::wait(setup.device, setup.frame);
  auto pixels = anari::map<Vec4>(setup.device, setup.frame, "channel.color");
  check(pixels.data != nullptr, context);
  const auto &pixel = pixels.data[16 * 32 + 16];
  for (int c = 0; c < 3; ++c) {
    check(std::isfinite(pixel[c]), context);
    if (c == channel)
      check(pixel[c] > 0.1f, context);
    else
      check(std::abs(pixel[c]) < 0.001f, context);
  }
  anari::unmap(setup.device, setup.frame, "channel.color");
}

void releaseSetup(const Setup &setup)
{
  anari::release(setup.device, setup.frame);
  anari::release(setup.device, setup.device);
}

} // namespace

int main()
{
  // 1. Overlapping devices: the original failure. The second device used to
  // get a null INeuray from mi_factory and abort MDL initialization.
  {
    auto first = makeSetup(0);
    renderAndCheck(first, 0, "first device failed before second was created");
    auto second = makeSetup(1);
    renderAndCheck(second, 1, "second concurrent device failed");
    // The first device must be unaffected by the second one's existence.
    renderAndCheck(first, 0, "first device broke after second was created");
    releaseSetup(second);
    // ... and must survive the second one's teardown, which must not shut the
    // shared SDK instance down.
    renderAndCheck(first, 0, "first device broke after second was destroyed");
    releaseSetup(first);
  }

  // 2. Sequential cycles: the last release does shut the SDK down, so a later
  // device has to reload and restart it.
  for (int i = 0; i < 2; ++i) {
    auto setup = makeSetup(2);
    renderAndCheck(setup, 2, "device after full shutdown failed");
    releaseSetup(setup);
  }

  // 3. Concurrent creation, covering the shared-instance acquisition lock.
  {
    constexpr int kDeviceCount = 3;
    std::vector<Setup> setups(kDeviceCount);
    std::vector<std::thread> threads;
    for (int i = 0; i < kDeviceCount; ++i)
      threads.emplace_back([&setups, i] { setups[i] = makeSetup(i % 3); });
    for (auto &thread : threads)
      thread.join();
    for (int i = 0; i < kDeviceCount; ++i)
      renderAndCheck(setups[i], i % 3, "concurrently created device failed");
    for (const auto &setup : setups)
      releaseSetup(setup);
  }

  std::puts("MDL multi-device creation and teardown passed");
  return 0;
}

// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

// The MDL SDK takes a single receiving logger per INeuray instance, and every
// device in a process shares one instance. There is therefore one "master"
// logger per instance -- the first device in installs it -- and SDK-internal
// diagnostics reach that one whichever device provoked them.
//
// This pins down both halves of that contract:
//   - SDK diagnostics (module compilation and friends) go to the master,
//     even when a later device caused them;
//   - VisRTX's own messages still reach the device they belong to, because
//     they never go through the SDK's logger at all.
//
// See "Logging" in docs/mdl-instance-sharing.md for why per-device routing of
// SDK diagnostics was deliberately dropped.

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;

namespace {

int failures = 0;

void check(bool ok, const char *what)
{
  std::printf("%-58s %s\n", what, ok ? "ok" : "FAILED");
  if (!ok)
    ++failures;
}

// Counted separately per device so we can see where each message landed.
struct Counters
{
  int sdk = 0; // MDL SDK's own diagnostics ("MDLC" category)
  int visrtx = 0; // messages VisRTX logs itself
};

Counters firstDevice;
Counters secondDevice;
bool counting = false;

void record(Counters &counters, const char *message)
{
  if (!counting || std::strstr(message, "VISRTX:MDL") == nullptr)
    return;
  if (std::strstr(message, "MDLC") != nullptr)
    ++counters.sdk;
  else
    ++counters.visrtx;
}

void firstStatus(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity,
    ANARIStatusCode,
    const char *message)
{
  record(firstDevice, message);
}

void secondStatus(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity,
    ANARIStatusCode,
    const char *message)
{
  record(secondDevice, message);
}

// The unreferenced local function makes the MDL core compiler emit a C225
// warning, giving a deterministic SDK-sourced diagnostic to trace.
std::string warningSource()
{
  return "mdl 1.6; import ::df::*; float unused_fn(float x){return x;} "
         "export material emitter()=material(surface: material_surface("
         "emission: material_emission(emission: df::diffuse_edf(), "
         "intensity: color(4.0))));";
}

anari::Frame makeScene(anari::Device device)
{
  const std::array<Vec3, 4> vertices = {Vec3{-1.f, -1.f, 0.f},
      Vec3{1.f, -1.f, 0.f},
      Vec3{1.f, 1.f, 0.f},
      Vec3{-1.f, 1.f, 0.f}};
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
  anari::setParameter(device, material, "source", warningSource());
  anari::setParameter(device, material, "materialName", "emitter");
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::commitParameters(device, world);
  anari::release(device, surface);

  auto camera = anari::newObject<anari::Camera>(device, "orthographic");
  anari::setParameter(device, camera, "position", Vec3{0.f, 0.f, 3.f});
  anari::setParameter(device, camera, "direction", Vec3{0.f, 0.f, -1.f});
  anari::setParameter(device, camera, "up", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(device, camera, "height", 2.f);
  anari::setParameter(device, camera, "aspect", 1.f);
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "quality");
  anari::setParameter(device, renderer, "pixelSamples", 1);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", std::array<unsigned, 2>{32, 32});
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(device, frame, "camera", camera);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::setAndReleaseParameter(device, frame, "world", world);
  anari::commitParameters(device, frame);
  return frame;
}

} // namespace

int main()
{
  // The first device to initialize MDL installs the instance's logger.
  auto first = makeVisRTXDevice(firstStatus);
  anari::setParameter(first, first, "forceInit", true);
  anari::commitParameters(first, first);

  auto second = makeVisRTXDevice(secondStatus);
  anari::setParameter(second, second, "forceInit", true);
  anari::commitParameters(second, second);

  counting = true;
  // Compile and render on the *second* device only. Its MDL compile runs on
  // that device's worker pool, so this also covers messages raised off the
  // calling thread.
  auto frame = makeScene(second);
  anari::render(second, frame);
  anari::wait(second, frame);
  counting = false;

  std::printf(
      "rendered on the second device: first saw %d SDK / %d VisRTX, "
      "second saw %d SDK / %d VisRTX\n",
      firstDevice.sdk,
      firstDevice.visrtx,
      secondDevice.sdk,
      secondDevice.visrtx);

  anari::release(second, frame);
  anari::release(first, first);
  anari::release(second, second);

  // SDK diagnostics land on the master logger, which belongs to the first
  // device. This is the documented trade for not routing per device.
  check(firstDevice.sdk > 0,
      "SDK diagnostics reach the instance's master logger");
  check(secondDevice.sdk == 0,
      "SDK diagnostics do not also reach the provoking device");
  // VisRTX's own messages never go through the SDK's logger, so they stay
  // with the device that produced them.
  check(
      secondDevice.visrtx > 0, "VisRTX's own messages stay with their device");
  check(firstDevice.visrtx == 0,
      "an idle device sees none of the other's VisRTX messages");

  if (failures) {
    std::fprintf(stderr, "FAIL: %d logging check(s) failed\n", failures);
    return 1;
  }
  std::puts("MDL logging follows the shared-instance master-logger contract");
  return 0;
}

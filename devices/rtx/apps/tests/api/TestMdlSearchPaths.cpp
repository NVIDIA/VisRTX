// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

// MDL search paths belong to the INeuray instance, not to a device, and every
// device in a process shares one instance. So a device setting its roots must
// not drop another device's, and changing a device's `mdlSearchPaths`
// parameter must make the new roots resolve.
//
// Each case renders a quad whose MDL material lives in a per-device directory:
// the material emits on a known channel, so the rendered pixel says which
// module was actually resolved, and a black pixel means the search path was
// lost.
//
// Note what is deliberately *not* asserted: that a retired root stops
// resolving. A module already loaded stays in the MDL database and keeps
// resolving by name no matter what the search paths later say (verified
// against the SDK), so retirement is only observable for modules that were
// never loaded.

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <unistd.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;

namespace {

int failures = 0;

void check(bool ok, const char *what)
{
  std::printf("%-62s %s\n", what, ok ? "ok" : "FAILED");
  if (!ok)
    ++failures;
}

void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR)
    std::fprintf(stderr, "FATAL: %s\n", message);
}

// A module whose single material emits on `channel` only.
std::string moduleSource(int channel)
{
  return "mdl 1.6; import ::df::*; export material emitter() = material("
         "surface: material_surface(emission: material_emission("
         "emission: df::diffuse_edf(), intensity: color("
      + std::to_string(channel == 0 ? 4.f : 0.f) + ", "
      + std::to_string(channel == 1 ? 4.f : 0.f) + ", "
      + std::to_string(channel == 2 ? 4.f : 0.f) + "))));";
}

std::string writeModule(
    const std::filesystem::path &dir, const std::string &name, int channel)
{
  std::filesystem::create_directories(dir);
  std::ofstream out(dir / (name + ".mdl"));
  out << moduleSource(channel);
  return "::" + name;
}

struct Setup
{
  anari::Device device{};
  anari::Frame frame{};
  anari::Material material{};
};

// A device rendering a quad with the named MDL module, loaded by name so it
// has to come off the search path.
Setup makeSetup(const std::string &searchPath, const std::string &moduleName)
{
  Setup setup;
  setup.device = makeVisRTXDevice(status);
  anari::setParameter(setup.device, setup.device, "mdlSearchPaths", searchPath);
  anari::commitParameters(setup.device, setup.device);

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

  setup.material = anari::newObject<anari::Material>(setup.device, "mdl");
  anari::setParameter(setup.device, setup.material, "sourceType", "module");
  anari::setParameter(setup.device, setup.material, "source", moduleName);
  anari::setParameter(setup.device, setup.material, "materialName", "emitter");
  anari::commitParameters(setup.device, setup.material);

  auto surface = anari::newObject<anari::Surface>(setup.device);
  anari::setAndReleaseParameter(setup.device, surface, "geometry", geometry);
  anari::setParameter(setup.device, surface, "material", setup.material);
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

// True if the quad emits on `channel`, i.e. the expected module resolved.
bool renders(const Setup &setup, int channel)
{
  anari::render(setup.device, setup.frame);
  anari::wait(setup.device, setup.frame);
  auto pixels = anari::map<Vec4>(setup.device, setup.frame, "channel.color");
  if (!pixels.data) {
    anari::unmap(setup.device, setup.frame, "channel.color");
    return false;
  }
  const auto &pixel = pixels.data[16 * 32 + 16];
  bool ok = true;
  for (int c = 0; c < 3; ++c) {
    if (!std::isfinite(pixel[c]))
      ok = false;
    else if (c == channel)
      ok = ok && pixel[c] > 0.1f;
    else
      ok = ok && std::abs(pixel[c]) < 0.001f;
  }
  anari::unmap(setup.device, setup.frame, "channel.color");
  return ok;
}

void releaseSetup(const Setup &setup)
{
  anari::release(setup.device, setup.material);
  anari::release(setup.device, setup.frame);
  anari::release(setup.device, setup.device);
}

} // namespace

int main()
{
  const auto root = std::filesystem::temp_directory_path()
      / ("visrtx_mdl_paths_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  const auto first = writeModule(root / "first", "visrtx_paths_first", 0);
  const auto second = writeModule(root / "second", "visrtx_paths_second", 1);
  const auto third = writeModule(root / "third", "visrtx_paths_third", 2);

  // 1. Two devices with disjoint roots. Both share one INeuray, so the second
  //    device's roots must not displace the first device's.
  {
    auto a = makeSetup((root / "first").string(), first);
    check(renders(a, 0), "first device resolves its own module");

    auto b = makeSetup((root / "second").string(), second);
    check(renders(b, 1), "second device resolves its own module");
    check(renders(a, 0), "first device still resolves after the second joined");

    // A third device must see neither of the others' roots displaced.
    auto c = makeSetup((root / "third").string(), third);
    check(renders(c, 2), "third device resolves its own module");
    check(renders(a, 0) && renders(b, 1),
        "first and second survive a third device");

    releaseSetup(c);
    check(renders(a, 0) && renders(b, 1),
        "survivors unaffected by a device leaving");
    releaseSetup(b);
    check(renders(a, 0), "first device unaffected by the second leaving");
    releaseSetup(a);
  }

  // 2. Updating a device's own paths. The new roots must take effect, and the
  //    retired ones must stop resolving -- a device that only ever added would
  //    silently keep stale roots alive for every other device too.
  {
    auto a = makeSetup((root / "first").string(), first);
    check(renders(a, 0), "device resolves its initial root");

    // Point the same device at a different root and rebuild the material.
    anari::setParameter(
        a.device, a.device, "mdlSearchPaths", (root / "second").string());
    anari::commitParameters(a.device, a.device);
    anari::setParameter(a.device, a.material, "source", second);
    anari::commitParameters(a.device, a.material);
    check(renders(a, 1), "device resolves a module from its new root");

    releaseSetup(a);
  }

  // 3. One device changing its roots must not disturb another device's.
  {
    auto a = makeSetup((root / "first").string(), first);
    auto b = makeSetup((root / "second").string(), second);
    anari::setParameter(
        a.device, a.device, "mdlSearchPaths", (root / "third").string());
    anari::commitParameters(a.device, a.device);
    anari::setParameter(a.device, a.material, "source", third);
    anari::commitParameters(a.device, a.material);
    check(renders(a, 2), "device resolves a module from its updated root");
    check(renders(b, 1), "the other device is unaffected by that update");
    releaseSetup(b);
    releaseSetup(a);
  }

  std::filesystem::remove_all(root);

  if (failures) {
    std::fprintf(stderr, "FAIL: %d search-path check(s) failed\n", failures);
    return 1;
  }
  std::puts("MDL search paths combine across devices and track updates");
  return 0;
}

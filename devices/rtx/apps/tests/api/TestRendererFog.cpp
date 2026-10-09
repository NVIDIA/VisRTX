// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "RendererFogSupport.h"

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace visrtx::fogtest;

/* Public ANARI fixture: a full-coverage black plane, no lighting noise or
 * nonlinear filters. Every pixel is safely away from geometry edges.
 */
struct Scene
{
  Scene(anari::Device device, const char *subtype);
  ~Scene();
  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;
  anari::Renderer newRenderer() const;
  Vec4 render(anari::Renderer r);
  void distance(float d);
  bool expect(const char *label, const Vec4 &expected);

  anari::Device device;
  const char *subtype;
  anari::World world;
  anari::Camera camera;
  anari::Material material;
  anari::Renderer renderer;
  anari::Frame frame;
  anari::Renderer boundRenderer{};
};

Scene::Scene(anari::Device d, const char *s) : device(d), subtype(s)
{
  material = anari::newObject<anari::Material>(d, "matte");
  ObjectOwner materialOwner(d, material);
  anari::setParameter(d, material, "color", Vec3{0.f, 0.f, 0.f});
  anari::commitParameters(d, material);
  constexpr std::array<Vec3, 4> POSITIONS = {Vec3{-2.f, -2.f, 0.f},
      Vec3{2.f, -2.f, 0.f},
      Vec3{2.f, 2.f, 0.f},
      Vec3{-2.f, 2.f, 0.f}};
  auto geometry = anari::newObject<anari::Geometry>(d, "quad");
  const ObjectOwner geometryOwner(d, geometry);
  anari::setParameterArray1D(
      d, geometry, "vertex.position", POSITIONS.data(), POSITIONS.size());
  anari::commitParameters(d, geometry);
  auto surface = anari::newObject<anari::Surface>(d);
  const ObjectOwner surfaceOwner(d, surface);
  anari::setParameter(d, surface, "geometry", geometry);
  anari::setParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);
  world = anari::newObject<anari::World>(d);
  ObjectOwner worldOwner(d, world);
  anari::setParameterArray1D(d, world, "surface", &surface, 1);
  anari::commitParameters(d, world);
  camera = anari::newObject<anari::Camera>(d, "orthographic");
  ObjectOwner cameraOwner(d, camera);
  anari::setParameter(d, camera, "direction", Vec3{0.f, 0.f, 2.f});
  anari::setParameter(d, camera, "up", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, camera, "height", 1.f);
  distance(10.f);
  renderer = newRenderer();
  ObjectOwner rendererOwner(d, renderer);
  frame = anari::newObject<anari::Frame>(d);
  ObjectOwner frameOwner(d, frame);
  anari::setParameter(d, frame, "size", UVec2{4, 4});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", camera);
  // The complete Scene now owns these references, including on later throws.
  materialOwner.release();
  worldOwner.release();
  cameraOwner.release();
  rendererOwner.release();
  frameOwner.release();
}

Scene::~Scene()
{
  anari::release(device, frame);
  anari::release(device, renderer);
  anari::release(device, camera);
  anari::release(device, world);
  anari::release(device, material);
}

anari::Renderer Scene::newRenderer() const
{
  auto r = anari::newObject<anari::Renderer>(device, subtype);
  ObjectOwner owner(device, r);
  anari::setParameter(device, r, "denoise", false);
  anari::setParameter(device, r, "fireflyFilterMode", "none");
  anari::setParameter(device, r, "ambientSamples", 0);
  anari::setParameter(device, r, "ambientRadiance", 1.f);
  anari::setParameter(device, r, "background", Vec4{0.f, 0.f, 0.f, 0.f});
  anari::commitParameters(device, r);
  owner.release();
  return r;
}

void Scene::distance(float d)
{
  anari::setParameter(device, camera, "position", Vec3{0.f, 0.f, -d});
  anari::commitParameters(device, camera);
}

Vec4 Scene::render(anari::Renderer r)
{
  if (boundRenderer != r) {
    anari::setParameter(device, frame, "renderer", r);
    anari::commitParameters(device, frame);
    boundRenderer = r;
  }
  anari::render(device, frame);
  anari::wait(device, frame);
  return readChannel<Vec4>(device, frame, "channel.color", 4, 4)[5];
}

bool Scene::expect(const char *label, const Vec4 &expected)
{
  anari::commitParameters(device, renderer);
  render(renderer);
  const auto pixels = readChannel<Vec4>(device, frame, "channel.color", 4, 4);
  bool ok = true;
  for (unsigned p = 0; p < 16; ++p) {
    const auto &actual = pixels[p];
    bool pixelOk = true;
    for (int c = 0; c < 4; ++c)
      pixelOk &= std::isfinite(actual[c])
          && std::abs(double(actual[c]) - expected[c])
              <= 1e-5 + 1e-4 * std::abs(double(expected[c]));
    if (!pixelOk && ok)
      fprintf(stderr,
          "%s %s pixel %u: got (%g,%g,%g,%g), expected (%g,%g,%g,%g)\n",
          subtype,
          label,
          p,
          actual[0],
          actual[1],
          actual[2],
          actual[3],
          expected[0],
          expected[1],
          expected[2],
          expected[3]);
    ok &= pixelOk;
  }
  return ok;
}

// Reusing a Frame without committing it is intentional: renderer edits must
// invalidate its old accumulation, rather than relying on a frame recommit.
int testLifecycle(Scene &scene)
{
  struct Settings
  {
    const char *mode;
    std::optional<Vec3> color;
    std::optional<float> start;
    std::optional<float> end;
    std::optional<float> density;
    Vec4 expected;
  };
  constexpr Settings CASES[] = {
      {"linear",
          Vec3{0.2f, 0.4f, 0.8f},
          0.f,
          20.f,
          {},
          {0.1f, 0.2f, 0.4f, 1.f}},
      {"linear", {}, 0.f, 20.f, {}, {0.5f, 0.5f, 0.5f, 1.f}},
      {"linear", {}, 5.f, 20.f, {}, {1.f / 3.f, 1.f / 3.f, 1.f / 3.f, 1.f}},
      {"linear", {}, {}, 20.f, {}, {0.5f, 0.5f, 0.5f, 1.f}},
      {"linear", {}, {}, {}, {}, {1.f, 1.f, 1.f, 1.f}},
      {"exp", {}, {}, {}, 0.1f, {0.63212056f, 0.63212056f, 0.63212056f, 1.f}},
      {"exp", {}, {}, {}, {}, {0.9999546f, 0.9999546f, 0.9999546f, 1.f}},
      {"exp2", {}, {}, {}, 0.05f, {0.22119922f, 0.22119922f, 0.22119922f, 1.f}},
      {"none", {}, {}, {}, {}, {0.f, 0.f, 0.f, 1.f}},
      {"linear", {}, {}, 20.f, {}, {0.5f, 0.5f, 0.5f, 1.f}},
      {nullptr, {}, {}, {}, {}, {0.f, 0.f, 0.f, 1.f}},
  };
  const auto d = scene.device;
  auto setOptional = [&](anari::Renderer r, const char *name, const auto &v) {
    if (v)
      anari::setParameter(d, r, name, *v);
    else
      anari::unsetParameter(d, r, name);
  };
  auto configure = [&](anari::Renderer r, const Settings &s) {
    if (s.mode)
      anari::setParameter(d, r, "fogMode", s.mode);
    else
      anari::unsetParameter(d, r, "fogMode");
    setOptional(r, "fogColor", s.color);
    setOptional(r, "fogStart", s.start);
    setOptional(r, "fogEnd", s.end);
    setOptional(r, "fogDensity", s.density);
    anari::commitParameters(d, r);
  };
  int failures = 0;
  scene.distance(10.f);
  for (const auto &settings : CASES) {
    // Populate old accumulated color before changing only the Renderer.
    for (int i = 0; i < 4; ++i)
      scene.render(scene.renderer);
    configure(scene.renderer, settings);
    failures += !scene.expect("recommit/unset defaults", settings.expected);
    const auto reused = scene.render(scene.renderer);
    auto fresh = scene.newRenderer();
    const ObjectOwner freshOwner(d, fresh);
    configure(fresh, settings);
    const auto reference = scene.render(fresh);
    for (int c = 0; c < 4; ++c)
      failures += std::abs(reused[c] - reference[c])
          > 1e-5f + 1e-4f * std::abs(reference[c]);
    // The same World can be viewed by a second Renderer without leaking fog.
    anari::setParameter(d, fresh, "fogMode", "none");
    anari::commitParameters(d, fresh);
    const auto unfogged = scene.render(fresh);
    failures += unfogged != Vec4{0.f, 0.f, 0.f, 1.f};
    failures += !scene.expect("renderer-local state", settings.expected);
  }
  scene.distance(0.5f);
  configure(scene.renderer, {"linear", {}, {}, {}, {}, {}});
  failures += !scene.expect("default linear interval", {0.5f, 0.5f, 0.5f, 1.f});
  configure(scene.renderer, {"exp", {}, {}, {}, {}, {}});
  failures += !scene.expect(
      "default exp density", {0.39346934f, 0.39346934f, 0.39346934f, 1.f});
  configure(scene.renderer, {"exp2", {}, {}, {}, {}, {}});
  failures += !scene.expect(
      "default exp2 density", {0.22119922f, 0.22119922f, 0.22119922f, 1.f});
  return failures;
}

int testColor(Scene &scene)
{
  const auto d = scene.device;
  const auto r = scene.renderer;
  int failures = 0;
  scene.distance(10.f);
  anari::setParameter(d, scene.material, "color", Vec3{1.f, 1.f, 1.f});
  anari::commitParameters(d, scene.material);
  anari::setParameter(d, r, "fogMode", "exp");
  anari::setParameter(d, r, "fogDensity", 0.1f);
  anari::setParameter(d, r, "fogStart", 5.f);
  anari::setParameter(d, r, "fogColor", Vec3{0.f, 0.f, 0.f});
  failures += !scene.expect("unshifted exp visibility, not 0.606531",
      {0.36787944f, 0.36787944f, 0.36787944f, 1.f});
  anari::setParameter(d, scene.material, "color", Vec3{0.f, 0.f, 0.f});
  anari::commitParameters(d, scene.material);
  anari::setParameter(d, r, "fogMode", "linear");
  anari::setParameter(d, r, "fogStart", 0.f);
  anari::setParameter(d, r, "fogEnd", 20.f);
  anari::setParameter(d, r, "fogColor", Vec3{4.f, 2.f, 0.5f});
  failures += !scene.expect("HDR linear RGB", {2.f, 1.f, 0.25f, 1.f});
  anari::setParameter(d, scene.material, "color", Vec3{0.2f, 0.4f, 0.8f});
  anari::commitParameters(d, scene.material);
  anari::unsetParameter(d, r, "fogMode");
  failures +=
      !scene.expect("known-color unfogged surface", {0.2f, 0.4f, 0.8f, 1.f});
  const auto baseline = scene.render(r);
  anari::setParameter(d, r, "fogMode", "none");
  failures += !scene.expect(
      "explicit none known-color surface", {0.2f, 0.4f, 0.8f, 1.f});
  failures += scene.render(r) != baseline;
  anari::setParameter(d, r, "fogMode", "linear");
  failures += !scene.expect("shade then fog", {2.1f, 1.2f, 0.65f, 1.f});
  // Quality's stochastic coverage is exercised separately from these opaque
  // numeric references. Keep the existing straight-through checks unchanged.
  if (std::string(scene.subtype) != "quality") {
    anari::setParameter(d, scene.material, "opacity", 0.5f);
    anari::commitParameters(d, scene.material);
    failures +=
        !scene.expect("premultiplied coverage", {1.05f, 0.6f, 0.325f, 0.5f});
    anari::setParameter(d, scene.material, "opacity", 0.f);
    anari::commitParameters(d, scene.material);
    failures += !scene.expect(
        "transparent surface injects no fog", {0.f, 0.f, 0.f, 0.f});
  }
  anari::setParameter(d, scene.material, "opacity", 1.f);
  anari::setParameter(d, scene.material, "color", Vec3{0.f, 0.f, 0.f});
  anari::commitParameters(d, scene.material);

  // Decode the actual 8-bit channels, compare with a linear-fog-then-sRGB
  // reference. Allow one code value for quantization and encoder rounding.
  anari::setParameter(d, r, "fogColor", Vec3{0.1f, 0.5f, 1.f});
  anari::commitParameters(d, r);
  anari::setParameter(d, scene.frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::commitParameters(d, scene.frame);
  anari::render(d, scene.frame);
  anari::wait(d, scene.frame);
  using Pixel8 = std::array<unsigned char, 4>;
  const auto pixels = readChannel<Pixel8>(
      d, scene.frame, "channel.color", 4, 4, ANARI_UFIXED8_RGBA_SRGB);
  constexpr Vec3 LINEAR{0.05f, 0.25f, 0.5f};
  for (int c = 0; c < 3; ++c) {
    const double encoded = LINEAR[c] <= 0.0031308
        ? 12.92 * LINEAR[c]
        : 1.055 * std::pow(double(LINEAR[c]), 1.0 / 2.4) - 0.055;
    const int expected = std::lround(encoded * 255.0);
    failures += std::abs(int(pixels[5][c]) - expected) > 1;
  }
  failures += pixels[5][3] != 255;
  anari::setParameter(d, scene.frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::commitParameters(d, scene.frame);

  anari::setParameter(d, r, "fogMode", "exp");
  anari::setParameter(d, r, "fogDensity", 1e-38f);
  anari::setParameter(d, r, "fogColor", Vec3{1e38f, 1e38f, 1e38f});
  failures +=
      !scene.expect("tiny density with HDR color", {10.f, 10.f, 10.f, 1.f});
  anari::setParameter(d, r, "fogMode", "exp2");
  anari::setParameter(d, r, "fogDensity", 1e-20f);
  failures +=
      !scene.expect("tiny squared optical distance", {1.f, 1.f, 1.f, 1.f});
  anari::setParameter(d, r, "fogMode", "linear");
  anari::setParameter(d, r, "fogEnd", std::numeric_limits<float>::max());
  failures += !scene.expect("extreme linear end with HDR color",
      {2.9387359f, 2.9387359f, 2.9387359f, 1.f});
  const float largeStart =
      std::nextafter(std::numeric_limits<float>::max(), 0.f);
  anari::setParameter(d, r, "fogStart", largeStart);
  failures += !scene.expect("extreme linear start", {0.f, 0.f, 0.f, 1.f});
  anari::setParameter(d, r, "fogStart", 0.f);
  anari::setParameter(d, r, "fogEnd", 20.f);
  constexpr float MAX_COLOR = std::numeric_limits<float>::max();
  anari::setParameter(d, r, "fogColor", Vec3{MAX_COLOR, MAX_COLOR, MAX_COLOR});
  failures += !scene.expect("maximum accepted finite fog RGB",
      {0.5f * MAX_COLOR, 0.5f * MAX_COLOR, 0.5f * MAX_COLOR, 1.f});
  return failures;
}

int testDistanceSweep(Scene &scene)
{
  const auto d = scene.device;
  const auto r = scene.renderer;
  anari::setParameter(d, r, "fogColor", Vec3{1.f, 1.f, 1.f});
  anari::setParameter(d, r, "fogStart", 2.f);
  anari::setParameter(d, r, "fogEnd", 22.f);
  anari::setParameter(d, r, "fogDensity", 0.05f);
  int failures = 0;
  for (const char *mode : {"linear", "exp", "exp2"}) {
    anari::setParameter(d, r, "fogMode", mode);
    float previous = 0.f;
    for (float distance :
        {0.001f, 0.1f, 1.f, 5.f, 10.f, 20.f, 100.f, 10000.f}) {
      scene.distance(distance);
      const long double x = static_cast<long double>(0.05f) * distance;
      const long double fraction = std::string(mode) == "linear"
          ? std::fmin(1.L, std::fmax(0.L, (distance - 2.L) / 20.L))
          : -std::expm1(std::string(mode) == "exp" ? -x : -x * x);
      const float f = fraction;
      failures += !scene.expect("monotonic distance sweep", {f, f, f, 1.f});
      const float actual = scene.render(r)[0];
      failures += !std::isfinite(actual) || actual < previous || actual < 0.f
          || actual > 1.f;
      previous = actual;
    }
  }
  printf(
      "%s monotonic distance sweep: 24 references, bounded finite fractions\n",
      scene.subtype);
  return failures;
}

int testDiagnosticRenderers(anari::Device d)
{
  int failures = 0;
  for (const char *subtype : {"debug", "debug_Ng", "test"}) {
    Scene scene(d, subtype);
    const auto baseline = scene.render(scene.renderer);
    anari::setParameter(d, scene.renderer, "fogMode", "linear");
    anari::setParameter(d, scene.renderer, "fogEnd", 1.f);
    anari::setParameter(d, scene.renderer, "fogColor", Vec3{4.f, 2.f, 1.f});
    anari::commitParameters(d, scene.renderer);
    if (scene.render(scene.renderer) != baseline) {
      fprintf(stderr,
          "%s diagnostic output changed with fog parameters\n",
          subtype);
      ++failures;
    }
  }
  return failures;
}

} // namespace

int main()
{
  std::vector<std::string> warnings;
  auto device = makeVisRTXDevice(collectWarningsStatus, &warnings);
  const ObjectOwner deviceOwner(device, device);
  int failures = requireRendererFogSupport(device);
  for (const char *subtype : {"fast", "interactive", "default", "quality"}) {
    Scene scene(device, subtype);
    failures += !scene.expect("omitted fog", {0.f, 0.f, 0.f, 1.f});
    anari::setParameter(device, scene.renderer, "fogMode", "linear");
    anari::setParameter(device, scene.renderer, "fogEnd", 20.f);
    failures += !scene.expect("linear midpoint", {0.5f, 0.5f, 0.5f, 1.f});
    anari::setParameter(device, scene.renderer, "fogStart", 5.f);
    anari::setParameter(device, scene.renderer, "fogEnd", 15.f);
    for (float distance : {1.f, 5.f, 10.f, 15.f, 20.f}) {
      scene.distance(distance);
      const float fraction = distance <= 5.f ? 0.f
          : distance >= 15.f                 ? 1.f
                                             : 0.5f;
      failures += !scene.expect(
          "linear endpoints/clamping", {fraction, fraction, fraction, 1.f});
    }
    // A valid interval below a common 1e-6 world-unit epsilon.
    anari::setParameter(device, scene.renderer, "fogStart", 1e-8f);
    anari::setParameter(device, scene.renderer, "fogEnd", 3e-8f);
    scene.distance(2e-8f);
    failures += !scene.expect("short linear interval", {0.5f, 0.5f, 0.5f, 1.f});
    scene.distance(10.f);
    anari::setParameter(device, scene.renderer, "fogStart", 5.f);
    anari::setParameter(device, scene.renderer, "fogEnd", 15.f);
    for (const char *mode : {"exp", "exp2"}) {
      anari::setParameter(device, scene.renderer, "fogMode", mode);
      for (float density :
          {0.f, 0.01f, 0.1f, 1.f, std::numeric_limits<float>::max()}) {
        anari::setParameter(device, scene.renderer, "fogDensity", density);
        // Independent long-double reference, evaluated in optical distance;
        // the nonzero linear start must never shift it.
        const long double opticalDistance = 10.L * density;
        const long double exponent = std::string(mode) == "exp"
            ? opticalDistance
            : opticalDistance * opticalDistance;
        const float fraction = -std::expm1(-exponent);
        failures += !scene.expect(mode, {fraction, fraction, fraction, 1.f});
      }
      anari::setParameter(device, scene.renderer, "fogDensity", 0.1f);
      anari::setParameter(device, scene.renderer, "fogStart", 100.f);
      anari::setParameter(device, scene.renderer, "fogEnd", -20.f);
      failures += !scene.expect("exponential ignores changed start/end",
          {0.63212056f, 0.63212056f, 0.63212056f, 1.f});
      anari::setParameter(device, scene.renderer, "fogStart", 5.f);
      anari::setParameter(device, scene.renderer, "fogEnd", 15.f);
    }
    anari::setParameter(device, scene.renderer, "fogMode", "exp");
    anari::setParameter(device, scene.renderer, "fogDensity", 0.1f);
    failures += !scene.expect("legacy unshifted exp regression",
        {0.63212056f, 0.63212056f, 0.63212056f, 1.f});
    // A surface at the camera plane is clipped by ordinary camera rays.
    // Supply origins behind it while retaining a zero-depth camera reference.
    auto referenceCamera = anari::newObject<anari::Camera>(device, "rayBuffer");
    const ObjectOwner referenceCameraOwner(device, referenceCamera);
    std::array<Vec3, 16> origins;
    origins.fill(Vec3{0.f, 0.f, -10.f});
    anari::setParameterArray2D(
        device, referenceCamera, "ray.org", origins.data(), 4, 4);
    anari::setParameter(
        device, referenceCamera, "position", Vec3{0.f, 0.f, 0.f});
    anari::setParameter(
        device, referenceCamera, "direction", Vec3{0.f, 0.f, 2.f});
    anari::commitParameters(device, referenceCamera);
    anari::setParameter(device, scene.frame, "camera", referenceCamera);
    anari::commitParameters(device, scene.frame);
    failures += !scene.expect("zero distance", {0.f, 0.f, 0.f, 1.f});
    anari::setParameter(device, scene.renderer, "fogMode", "exp2");
    failures += !scene.expect("zero distance exp2", {0.f, 0.f, 0.f, 1.f});
    anari::setParameter(
        device, referenceCamera, "position", Vec3{0.f, 0.f, 10.f});
    anari::commitParameters(device, referenceCamera);
    failures += !scene.expect(
        "negative view depth clamps to zero", {0.f, 0.f, 0.f, 1.f});
    anari::setParameter(device, scene.frame, "camera", scene.camera);
    anari::commitParameters(device, scene.frame);

    auto validLinear = [&](anari::Renderer r = nullptr) {
      if (!r)
        r = scene.renderer;
      anari::setParameter(device, r, "fogMode", "linear");
      anari::setParameter(device, r, "fogStart", 5.f);
      anari::setParameter(device, r, "fogEnd", 15.f);
      anari::setParameter(device, r, "fogColor", Vec3{1.f, 1.f, 1.f});
      anari::setParameter(device, r, "fogDensity", 0.1f);
    };
    auto invalid = [&](const char *name, auto value, bool exponential = false) {
      validLinear();
      if (exponential)
        anari::setParameter(device, scene.renderer, "fogMode", "exp");
      failures += !scene.expect("valid before invalid",
          exponential ? Vec4{0.63212056f, 0.63212056f, 0.63212056f, 1.f}
                      : Vec4{0.5f, 0.5f, 0.5f, 1.f});
      warnings.clear();
      anari::setParameter(device, scene.renderer, name, value);
      failures += !scene.expect(name, {0.f, 0.f, 0.f, 1.f});
      bool warned = false;
      for (const auto &warning : warnings)
        warned |= warning.find(name) != std::string::npos;
      if (!warned) {
        fprintf(stderr, "%s: missing warning for %s\n", subtype, name);
        ++failures;
      }
      auto fresh = scene.newRenderer();
      const ObjectOwner freshOwner(device, fresh);
      validLinear(fresh);
      if (exponential)
        anari::setParameter(device, fresh, "fogMode", "exp");
      anari::setParameter(device, fresh, name, value);
      anari::commitParameters(device, fresh);
      const auto freshInvalid = scene.render(fresh);
      failures += scene.render(scene.renderer) != freshInvalid;
      validLinear();
      failures += !scene.expect("corrected commit", {0.5f, 0.5f, 0.5f, 1.f});
      validLinear(fresh);
      anari::commitParameters(device, fresh);
      const auto freshCorrected = scene.render(fresh);
      failures += scene.render(scene.renderer) != freshCorrected;
    };
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
    constexpr float INFINITY_VALUE = std::numeric_limits<float>::infinity();
    invalid("fogMode", "unknown");
    invalid("fogMode", 42);
    for (float bad : {-1.f, NAN_VALUE, INFINITY_VALUE}) {
      invalid("fogColor", Vec3{0.f, bad, 1.f});
      invalid("fogStart", bad);
      invalid("fogEnd", bad);
      invalid("fogDensity", bad, true);
    }
    invalid("fogEnd", 5.f);
    invalid("fogEnd", 4.f);
    invalid("fogColor", 1.f);
    invalid("fogStart", "bad");
    invalid("fogEnd", 15);
    invalid("fogDensity", "bad", true);
    validLinear();
    anari::setParameter(device, scene.renderer, "fogDensity", NAN_VALUE);
    failures +=
        !scene.expect("linear ignores density", {0.5f, 0.5f, 0.5f, 1.f});
    anari::setParameter(device, scene.renderer, "fogMode", "exp2");
    anari::setParameter(device, scene.renderer, "fogDensity", 0.1f);
    anari::setParameter(device, scene.renderer, "fogStart", NAN_VALUE);
    anari::setParameter(
        device, scene.renderer, "fogEnd", "inactive wrong type");
    failures += !scene.expect("exp2 ignores linear inputs",
        {0.63212056f, 0.63212056f, 0.63212056f, 1.f});
    anari::setParameter(device, scene.renderer, "fogMode", "none");
    anari::setParameter(device,
        scene.renderer,
        "fogColor",
        Vec3{NAN_VALUE, -1.f, INFINITY_VALUE});
    anari::setParameter(device, scene.renderer, "fogDensity", -1.f);
    warnings.clear();
    failures +=
        !scene.expect("none ignores invalid inputs", {0.f, 0.f, 0.f, 1.f});
    failures += !warnings.empty();
    anari::unsetParameter(device, scene.renderer, "fogMode");
    failures += !scene.expect(
        "omitted mode ignores invalid inputs", {0.f, 0.f, 0.f, 1.f});
    failures += !warnings.empty();
    failures += testLifecycle(scene);
    failures += testColor(scene);
    failures += testDistanceSweep(scene);
  }
  failures += testDiagnosticRenderers(device);
  printf("Renderer fog: %d failure(s)\n", failures);
  return failures ? 1 : 0;
}

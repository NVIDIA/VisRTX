// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "RendererFogSupport.h"

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using namespace visrtx::fogtest;

constexpr unsigned WIDTH = 16;
constexpr unsigned PIXELS = WIDTH * WIDTH;
constexpr Vec3 RADIANCE = {0.8f, 0.4f, 0.2f};
constexpr Vec3 FOG_COLOR = {0.2f, 0.6f, 1.f};

/* All observations are mapped public ANARI channels. Each render starts with
 * frameID zero, pairing random samples across fog edits, not just mean colors.
 */
struct Scene
{
  Scene(anari::Device d, const char *subtype);
  ~Scene();
  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;
  void attach(anari::Surface surface = {},
      anari::Light light = {},
      anari::Volume volume = {});
  void fog(const char *mode, float start = 0.f, float end = 20.f);
  Image render(int samples = 1);

  anari::Device d;
  const char *subtype;
  anari::World world;
  anari::Camera camera;
  anari::Renderer renderer;
};

Scene::Scene(anari::Device device, const char *type) : d(device), subtype(type)
{
  world = anari::newObject<anari::World>(d);
  ObjectOwner worldOwner(d, world);
  camera = anari::newObject<anari::Camera>(d, "orthographic");
  ObjectOwner cameraOwner(d, camera);
  anari::setParameter(d, camera, "position", Vec3{0.f, 0.f, 0.f});
  anari::setParameter(d, camera, "direction", Vec3{0.f, 0.f, 1.f});
  anari::setParameter(d, camera, "up", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, camera, "height", 1.f);
  anari::commitParameters(d, camera);
  renderer = anari::newObject<anari::Renderer>(d, subtype);
  ObjectOwner rendererOwner(d, renderer);
  anari::setParameter(d, renderer, "denoise", false);
  anari::setParameter(d, renderer, "fireflyFilterMode", "none");
  anari::setParameter(d, renderer, "ambientSamples", 0);
  anari::setParameter(d, renderer, "ambientRadiance", 1.f);
  anari::setParameter(d, renderer, "background", Vec4{0.f, 0.f, 0.f, 0.f});
  anari::setParameter(d, renderer, "fogDistanceMetric", "viewDepth");
  anari::setParameter(d, renderer, "fogColorSource", "constant");
  anari::setParameter(d, renderer, "fogColor", FOG_COLOR);
  fog("none");
  worldOwner.release();
  cameraOwner.release();
  rendererOwner.release();
}

Scene::~Scene()
{
  anari::release(d, renderer);
  anari::release(d, camera);
  anari::release(d, world);
}

void Scene::attach(
    anari::Surface surface, anari::Light light, anari::Volume volume)
{
  auto set = [&](const char *name, auto object) {
    if (object)
      anari::setParameterArray1D(d, world, name, &object, 1);
    else
      anari::unsetParameter(d, world, name);
  };
  set("surface", surface);
  set("light", light);
  set("volume", volume);
  anari::commitParameters(d, world);
}

void Scene::fog(const char *mode, float start, float end)
{
  anari::setParameter(d, renderer, "fogMode", mode);
  anari::setParameter(d, renderer, "fogStart", start);
  anari::setParameter(d, renderer, "fogEnd", end);
  anari::commitParameters(d, renderer);
}

Image Scene::render(int samples)
{
  anari::setParameter(d, renderer, "pixelSamples", samples);
  anari::commitParameters(d, renderer);
  auto frame = anari::newObject<anari::Frame>(d);
  const ObjectOwner frameOwner(d, frame);
  anari::setParameter(d, frame, "size", UVec2{WIDTH, WIDTH});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, frame, "channel.normal", ANARI_FLOAT32_VEC3);
  anari::setParameter(d, frame, "channel.albedo", ANARI_FLOAT32_VEC3);
  anari::setParameter(d, frame, "channel.depth", ANARI_FLOAT32);
  for (const auto *name :
      {"channel.objectId", "channel.primitiveId", "channel.instanceId"})
    anari::setParameter(d, frame, name, ANARI_UINT32);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", camera);
  anari::setParameter(d, frame, "renderer", renderer);
  anari::commitParameters(d, frame);

  anari::render(d, frame);

  anari::wait(d, frame);
  return readImage(d, frame, {WIDTH, WIDTH});
}

bool check(bool value, const Scene &scene, const char *label)
{
  if (!value)
    fprintf(stderr, "%s: %s failed\n", scene.subtype, label);
  return value;
}

bool expectColor(const Scene &scene, const Image &image, const Vec4 &ref)
{
  for (const auto &pixel : image.color) {
    for (int c = 0; c < 4; ++c) {
      if (!std::isfinite(pixel[c])
          || std::abs(double(pixel[c]) - ref[c])
              > 1e-5 + 1e-4 * std::abs(double(ref[c]))) {
        fprintf(stderr,
            "%s: color (%g,%g,%g,%g), expected (%g,%g,%g,%g)\n",
            scene.subtype,
            pixel[0],
            pixel[1],
            pixel[2],
            pixel[3],
            ref[0],
            ref[1],
            ref[2],
            ref[3]);
        return false;
      }
    }
  }
  return true;
}

anari::Surface plane(anari::Device d,
    float z,
    Vec3 color,
    float slope = 0.f,
    bool emissive = false,
    float extent = 2.f,
    float opacity = 1.f)
{
  const std::array<Vec3, 4> positions = {
      Vec3{-extent, -extent, z - extent * slope},
      Vec3{extent, -extent, z + extent * slope},
      Vec3{extent, extent, z + extent * slope},
      Vec3{-extent, extent, z - extent * slope}};
  auto geometry = anari::newObject<anari::Geometry>(d, "quad");
  const ObjectOwner geometryOwner(d, geometry);
  anari::setParameterArray1D(
      d, geometry, "vertex.position", positions.data(), positions.size());
  anari::commitParameters(d, geometry);
  auto material = anari::newObject<anari::Material>(
      d, emissive ? "physicallyBased" : "matte");
  const ObjectOwner materialOwner(d, material);
  if (emissive) {
    anari::setParameter(d, material, "baseColor", Vec3{});
    anari::setParameter(d, material, "specular", 0.f);
    anari::setParameter(d, material, "emissive", color);
  } else
    anari::setParameter(d, material, "color", color);
  anari::setParameter(d, material, "opacity", opacity);
  anari::commitParameters(d, material);
  auto surface = anari::newObject<anari::Surface>(d);
  ObjectOwner surfaceOwner(d, surface);
  anari::setParameter(d, surface, "geometry", geometry);
  anari::setParameter(d, surface, "material", material);
  anari::setParameter(d, surface, "id", 42u);
  anari::commitParameters(d, surface);
  surfaceOwner.release();
  return surface;
}

anari::Light quadLight(anari::Device d)
{
  auto light = anari::newObject<anari::Light>(d, "quad");
  ObjectOwner lightOwner(d, light);
  anari::setParameter(d, light, "position", Vec3{-2.f, -2.f, 10.f});
  anari::setParameter(d, light, "edge1", Vec3{4.f, 0.f, 0.f});
  anari::setParameter(d, light, "edge2", Vec3{0.f, 4.f, 0.f});
  anari::setParameter(d, light, "color", RADIANCE);
  anari::setParameter(d, light, "intensity", 1.f);
  anari::commitParameters(d, light);
  lightOwner.release();
  return light;
}

int testProxy(Scene &scene)
{
  logCase(std::string(scene.subtype) + "/proxy");
  int failures = 0;
  const auto d = scene.d;
  auto light = quadLight(d);
  const ObjectOwner lightOwner(d, light);
  const bool quality = std::string(scene.subtype) == "quality";
  if (quality)
    anari::setParameter(d, scene.renderer, "ambientRadiance", 0.f);
  scene.attach({}, light);
  scene.fog("none");
  const auto baseline = scene.render();
  const bool visible = std::string(scene.subtype) != "fast";
  failures += !expectColor(
      scene, baseline, visible ? Vec4{0.8f, 0.4f, 0.2f, 1.f} : Vec4{});
  for (const float end : {20.f, 10.f}) {
    scene.fog("linear", 0.f, end);
    const auto fogged = scene.render();
    const Vec4 expected =
        end == 20.f ? Vec4{0.5f, 0.5f, 0.6f, 1.f} : Vec4{0.2f, 0.6f, 1.f, 1.f};
    failures += !expectColor(scene, fogged, visible ? expected : Vec4{});
    failures += !sameAuxiliary(baseline, fogged, scene.subtype);
    for (unsigned p = 0; p < PIXELS; ++p) {
      failures += !check(fogged.object[p] == ~0u && fogged.primitive[p] == ~0u
              && fogged.instance[p] == ~0u,
          scene,
          "proxy remains unpickable");
      // Quality does not write picking depth for proxies; preserve that
      // existing policy rather than assigning Interactive's depth semantics.
      if (visible && !quality)
        failures += !check(fogged.depth[p] == 10.f, scene, "proxy hit depth");
    }
    // A known opaque contribution at the same depth, from emission in Quality
    // or constant ambient shading otherwise, gets the same camera operation.
    auto reference =
        plane(d, 10.f, RADIANCE, 0.f, std::string(scene.subtype) == "quality");
    const ObjectOwner referenceOwner(d, reference);
    scene.attach(reference);
    failures += !expectColor(scene, scene.render(), expected);
    scene.attach({}, light);
  }
  anari::setParameter(d, light, "visible", false);
  anari::commitParameters(d, light);
  scene.fog("none");
  const auto hidden = scene.render();
  scene.fog("linear", 0.f, 1.f);
  const auto hiddenFog = scene.render();
  failures += !expectColor(scene, hiddenFog, Vec4{});
  failures += !check(
      hidden.color == hiddenFog.color, scene, "hidden light stays hidden");
  failures += !sameAuxiliary(hidden, hiddenFog, scene.subtype);
  return failures;
}

// The receiver is wholly before fogStart, but the off-screen light is beyond
// it. Tilting the receiver lets a camera-facing normal see that deeper light.
// This isolates light transport from fog on the receiver itself.
int testIllumination(Scene &scene)
{
  logCase(std::string(scene.subtype) + "/illumination");
  if (std::string(scene.subtype) == "fast")
    return 0; // Fast uses headlight shading, not analytic-light illumination.
  const auto d = scene.d;
  int failures = 0;
  auto receiver = plane(d, 2.f, Vec3{0.6f, 0.6f, 0.6f}, 2.f);
  const ObjectOwner receiverOwner(d, receiver);
  auto light = quadLight(d);
  const ObjectOwner lightOwner(d, light);
  anari::setParameter(d, light, "position", Vec3{4.f, -0.5f, 5.5f});
  anari::setParameter(d, light, "edge1", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, light, "edge2", Vec3{0.f, 0.f, 1.f});
  anari::setParameter(d, light, "intensity", 20.f);
  anari::commitParameters(d, light);
  anari::setParameter(d, scene.renderer, "ambientRadiance", 0.f);
  scene.attach(receiver, light);
  scene.fog("none");
  const auto baseline = scene.render(64);
  double mean = 0.;
  for (const auto &p : baseline.color)
    mean += p[0] / PIXELS;
  failures +=
      !check(mean > 0.01, scene, "receiver is illuminated by the light");
  for (const bool visible : {true, false}) {
    anari::setParameter(d, light, "visible", visible);
    anari::commitParameters(d, light);
    scene.fog("linear", 3.1f, 4.f);
    const auto active = scene.render(64);
    failures += !check(active.color == baseline.color,
        scene,
        "fog and proxy visibility leave receiver illumination unchanged");
    failures += !sameAuxiliary(baseline, active, scene.subtype);
  }
  anari::setParameter(d, light, "intensity", 0.f);
  anari::commitParameters(d, light);
  failures += !expectColor(scene, scene.render(64), Vec4{0.f, 0.f, 0.f, 1.f});
  printf("%s unaffected receiver mean red = %.9g (64 samples/pixel)\n",
      scene.subtype,
      mean);
  anari::setParameter(d, scene.renderer, "ambientRadiance", 1.f);
  return failures;
}

anari::Volume slab(anari::Device d,
    float thickness,
    Vec3 color = {0.1f, 0.2f, 0.3f},
    float extent = 1.f)
{
  auto field = anari::newObject<anari::SpatialField>(d, "structuredRegular");
  const ObjectOwner fieldOwner(d, field);
  // ANARI-owned storage outlives this helper; no borrowed application pointer.
  auto data = anari::newArray3D(d, ANARI_FLOAT32, 17, 17, 17);
  const ObjectOwner dataOwner(d, data);
  {
    const ArrayMapping<float> values(d, data);
    std::fill(values.data(), values.data() + 17 * 17 * 17, 0.5f);
  }
  anari::setParameter(d, field, "data", data);
  anari::setParameter(d, field, "origin", Vec3{-extent, -extent, 3.f});
  anari::setParameter(
      d, field, "spacing", Vec3{extent / 8.f, extent / 8.f, thickness / 16.f});
  anari::commitParameters(d, field);
  auto volume = anari::newObject<anari::Volume>(d, "transferFunction1D");
  ObjectOwner volumeOwner(d, volume);
  anari::setParameter(d, volume, "value", field);
  anari::setParameter(d, volume, "color", color);
  anari::setParameter(d, volume, "opacity", 0.5f);
  anari::setParameter(d, volume, "unitDistance", 1.f);
  anari::setParameter(d, volume, "id", 77u);
  anari::commitParameters(d, volume);
  volumeOwner.release();
  return volume;
}

int testVolume(Scene &scene)
{
  logCase(std::string(scene.subtype) + "/volume");
  const auto d = scene.d;
  int failures = 0;
  auto volume = slab(d, 2.f);
  const ObjectOwner volumeOwner(d, volume);
  anari::setParameter(d, scene.renderer, "volumeSamplingRate", 1.f);
  scene.attach({}, {}, volume);
  scene.fog("none");
  const auto only = scene.render();
  // Homogeneous emission-absorption: T = (1 - 0.5)^(2/1) = 1/4.
  // The slab spans exactly 32 march steps, so initial sample jitter cannot
  // change this integral. No statistical allowance is needed in this case.
  failures += !expectColor(scene, only, Vec4{0.075f, 0.15f, 0.225f, 0.75f});
  scene.fog("linear", 0.f, 1.f);
  const auto fogOnly = scene.render();
  failures +=
      !check(only.color == fogOnly.color, scene, "volume-only RGB unchanged");
  failures += !sameAuxiliary(only, fogOnly, scene.subtype);
  for (unsigned p = 0; p < PIXELS; ++p) {
    failures += !check(only.object[p] == 77u && only.primitive[p] == 77u
            && only.depth[p] >= 3.f && only.depth[p] < 3.0625f,
        scene,
        "volume retains IDs and sampled integration depth");
  }

  auto surface = plane(d, 10.f, RADIANCE);
  const ObjectOwner surfaceOwner(d, surface);
  auto light = quadLight(d);
  const ObjectOwner lightOwner(d, light);
  // Check ordinary surfaces in every renderer and proxies where already
  // displayed. Volume entry/sample depth (~3) must not replace surface
  // depth 10.
  for (const bool proxy : {false, true}) {
    if (proxy && std::string(scene.subtype) == "fast")
      continue;
    scene.attach(proxy ? anari::Surface{} : surface,
        proxy ? light : anari::Light{},
        volume);
    scene.fog("none");
    const auto baseline = scene.render();
    failures += !expectColor(scene, baseline, Vec4{0.275f, 0.25f, 0.275f, 1.f});
    scene.fog("linear", 0.f, 20.f);
    const auto partial = scene.render();
    failures += !expectColor(scene, partial, Vec4{0.2f, 0.275f, 0.375f, 1.f});
    failures += !sameAuxiliary(baseline, partial, scene.subtype);
    scene.fog("linear", 0.f, 10.f);
    const auto full = scene.render();
    failures += !expectColor(scene, full, Vec4{0.125f, 0.3f, 0.475f, 1.f});
    failures += !sameAuxiliary(baseline, full, scene.subtype);
  }
  return failures;
}

bool expectSlabMean(const Scene &scene,
    const Image &image,
    const Vec3 &surfaceColor,
    bool hasSurface)
{
  // Independent homogeneous emission-absorption reference. This slab spans
  // 32.5 steps of 1/16 world unit: jitter gives 32 or 33 samples. The exact
  // physical transmittance is 2^(-2.03125). Bound the lattice quadrature bias
  // by h^2 max(T'')/8, and sampling error by Hoeffding with failure probability
  // 1e-9 for N=256*64 independent camera samples. No observed error sets the
  // tolerance. An exact integrator also satisfies this reference.
  constexpr double LENGTH = 2.03125;
  constexpr double STEP = 0.0625;
  constexpr int SAMPLES = 64;
  const double mu = std::log(2.);
  const double transmission = std::exp(-mu * LENGTH);
  constexpr double UPPER = 0.25;
  const double lower = std::exp(-mu * (2. + STEP));
  const double quadrature = STEP * STEP * mu * mu * UPPER / 8.;
  const double sampling = (UPPER - lower)
      * std::sqrt(std::log(2. / 1e-9) / (2. * PIXELS * SAMPLES));
  const double allowance = quadrature + sampling;
  constexpr Vec3 VOLUME_COLOR = {0.1f, 0.2f, 0.3f};
  std::array<double, 4> mean{};
  for (const auto &p : image.color)
    for (int c = 0; c < 4; ++c)
      mean[c] += double(p[c]) / PIXELS;
  bool ok = true;
  double maxError = 0.;
  for (int c = 0; c < 4; ++c) {
    const double foreground = c == 3 ? 1. : VOLUME_COLOR[c];
    const double background = c == 3 ? double(hasSurface) : surfaceColor[c];
    const double expected =
        foreground * (1. - transmission) + background * transmission;
    const double error = std::abs(mean[c] - expected);
    maxError = std::max(maxError, error);
    ok &= std::isfinite(mean[c])
        && error <= std::abs(background - foreground) * allowance + 1e-5
                + 1e-4 * std::abs(expected);
  }
  printf(
      "%s %s slab: max mean error %.9g, T allowance %.9g "
      "(quadrature %.9g + sampling %.9g; %u samples)\n",
      scene.subtype,
      hasSurface ? "surface-behind" : "volume-only",
      maxError,
      allowance,
      quadrature,
      sampling,
      PIXELS * SAMPLES);
  return check(ok, scene, "statistical homogeneous-slab reference");
}

int testJitteredVolume(Scene &scene)
{
  logCase(std::string(scene.subtype) + "/jittered-volume");
  const auto d = scene.d;
  int failures = 0;
  auto volume = slab(d, 2.03125f);
  const ObjectOwner volumeOwner(d, volume);
  auto surface = plane(d, 10.f, RADIANCE);
  const ObjectOwner surfaceOwner(d, surface);
  anari::setParameter(d, scene.renderer, "volumeSamplingRate", 1.f);
  for (const bool hasSurface : {false, true}) {
    scene.attach(hasSurface ? surface : anari::Surface{}, {}, volume);
    scene.fog("none");
    const auto baseline = scene.render(64);
    failures += !expectSlabMean(
        scene, baseline, hasSurface ? RADIANCE : Vec3{}, hasSurface);
    for (const float end : {20.f, 10.f}) {
      scene.fog("linear", 0.f, end);
      const auto active = scene.render(64);
      failures += !sameAuxiliary(baseline, active, scene.subtype);
      if (hasSurface) {
        failures += !expectSlabMean(scene,
            active,
            end == 20.f ? Vec3{0.5f, 0.5f, 0.6f} : FOG_COLOR,
            true);
      } else {
        failures += !check(active.color == baseline.color,
            scene,
            "jittered volume-only RGB remains bit-identical");
        failures += !expectSlabMean(scene, active, Vec3{}, false);
      }
    }
  }
  return failures;
}

// Quality uses sampled extinction, not the straight-through marcher's
// emission-absorption estimator. Its reference is Beer survival probability;
// after a real collision, no later hit is a new camera-visible fog event.
constexpr int QUALITY_SAMPLES = 4096;
constexpr Vec4 BACKDROP = {0.13f, 0.27f, 0.41f, 0.3f};

bool isMiss(unsigned p)
{
  return p / WIDTH == WIDTH - 1;
}

void configureObliqueRayBuffer(Scene &scene)
{
  const auto d = scene.d;
  auto replacement = anari::newObject<anari::Camera>(d, "rayBuffer");
  const ObjectOwner oldCameraOwner(d, scene.camera);
  scene.camera = replacement;
  auto origins = anari::newArray2D(d, ANARI_FLOAT32_VEC3, WIDTH, WIDTH);
  const ObjectOwner originsOwner(d, origins);
  auto directions = anari::newArray2D(d, ANARI_FLOAT32_VEC3, WIDTH, WIDTH);
  const ObjectOwner directionsOwner(d, directions);
  {
    const ArrayMapping<Vec3> o(d, origins);
    const ArrayMapping<Vec3> v(d, directions);
    for (unsigned p = 0; p < PIXELS; ++p) {
      o.data()[p] = {0.f, 0.f, float(2 * (p % 2))};
      v.data()[p] = isMiss(p) ? Vec3{0.f, 0.f, -1.f} : Vec3{3.f, 0.f, 4.f};
    }
  }
  anari::setParameter(d, scene.camera, "ray.org", origins);
  anari::setParameter(d, scene.camera, "ray.dir", directions);
  anari::setParameter(d, scene.camera, "position", Vec3{0.f, 0.f, -1.f});
  anari::setParameter(d, scene.camera, "direction", Vec3{0.f, 0.f, 1.f});
  anari::commitParameters(d, scene.camera);
  anari::setParameter(d, scene.renderer, "background", BACKDROP);
  anari::setParameter(d, scene.renderer, "premultiplyBackground", false);
  anari::setParameter(d, scene.renderer, "ambientRadiance", 0.f);
  anari::setParameter(d, scene.renderer, "maxRayDepth", 4);
  anari::commitParameters(d, scene.renderer);
}

double qualityFraction(unsigned p, float z, bool radial, bool full)
{
  // Camera reference z=-1, supplied ray origins z=0/2, unit ray z=4/5.
  const double distance = radial ? (z - 2 * (p % 2)) * 1.25 : z + 1.;
  return full ? 1. : distance / 20.;
}

Vec3 qualityTarget(bool background)
{
  return background ? Vec3{BACKDROP[0], BACKDROP[1], BACKDROP[2]} : FOG_COLOR;
}

struct FogCase
{
  const char *name;
  const char *mode;
  bool radial;
  bool background;
  bool full;
  bool representativeOnly = false;
};

// Complete selector products belong to the absorbing surface representative.
// Other paths distinguish both distances/sources locally, and retain both full
// colors; at saturation the metrics are equivalent for every supplied hit.
constexpr FogCase QUALITY_CASES[] = {
    {"linear-view-constant", "linear", false, false, false},
    {"linear-ray-background", "linear", true, true, false},
    {"linear-view-background", "linear", false, true, false, true},
    {"linear-ray-constant", "linear", true, false, false, true},
    {"full-constant", "linear", false, false, true},
    {"full-background", "linear", true, true, true}};

// All curves still exercise volume/proxy composition on every contribution
// path. Only the volume+surface representative repeats the other source at
// each metric. Independent cheap formula/selector anchors cover those products
// without multiplying every physical interaction (see the coverage ledger).
constexpr FogCase STRAIGHT_CASES[] = {
    {"linear-view-constant", "linear", false, false, false},
    {"linear-ray-background", "linear", true, true, false},
    {"exp-view-constant", "exp", false, false, false},
    {"exp-ray-background", "exp", true, true, false},
    {"exp2-view-constant", "exp2", false, false, false},
    {"exp2-ray-background", "exp2", true, true, false},
    {"linear-view-background", "linear", false, true, false, true},
    {"linear-ray-constant", "linear", true, false, false, true},
    {"exp-view-background", "exp", false, true, false, true},
    {"exp-ray-constant", "exp", true, false, false, true},
    {"exp2-view-background", "exp2", false, true, false, true},
    {"exp2-ray-constant", "exp2", true, false, false, true},
    {"full-constant", "linear", false, false, true},
    {"full-background", "linear", true, true, true},
    {"disabled-restoration", "none", true, true, false}};

void caseLabel(const Scene &scene, const std::string &path, const char *name)
{
  fflush(stdout);
  fprintf(stderr,
      "[fog-interaction] %s/%s/%s\n",
      scene.subtype,
      path.c_str(),
      name);
  fflush(stderr);
}

void qualityFog(Scene &scene, bool radial, bool background, bool full)
{
  anari::setParameter(scene.d,
      scene.renderer,
      "fogDistanceMetric",
      radial ? "rayDistance" : "viewDepth");
  anari::setParameter(scene.d,
      scene.renderer,
      "fogColorSource",
      background ? "background" : "constant");
  scene.fog("linear", 0.f, full ? 0.01f : 20.f);
}

bool expectQuality(const char *label,
    const Image &image,
    const std::vector<Vec4> &expected,
    double sampleRange = 0.,
    int samples = QUALITY_SAMPLES)
{
  // A Bernoulli camera survival/coverage decision has range <= sampleRange.
  // Hoeffding, two-sided failure probability 1e-9 per assertion, gives the
  // per-pixel and per-origin-group bounds independently of measured error.
  // Each group has 120 pixels * samples. Paired volume-lighting estimates
  // cancel when testing the added surface term, so only survival is random.
  // All nonzero-range assertions retain 4096 spp. Zero-opacity primary proxies
  // are deterministic and use 1 spp with the same original numeric criterion.
  const double pixelBound =
      sampleRange * std::sqrt(std::log(2. / 1e-9) / (2. * samples));
  const double groupBound = pixelBound / std::sqrt(120.);
  double meanError[2][4] = {};
  double maxError = 0., maxMean = 0., maxBoundRatio = 0.;
  bool ok = true;
  for (unsigned p = 0; p < PIXELS; ++p) {
    for (unsigned c = 0; c < 4; ++c) {
      const double error = double(image.color[p][c]) - expected[p][c];
      const double numeric = 1e-5 + 1e-4 * std::abs(expected[p][c]);
      const double bound = numeric + (isMiss(p) ? 0. : pixelBound);
      ok &= std::isfinite(error) && std::abs(error) <= bound;
      maxBoundRatio = std::max(maxBoundRatio, std::abs(error) / bound);
      maxError = std::max(maxError, std::abs(error));
      if (!isMiss(p))
        meanError[p % 2][c] += error / 120.;
    }
  }
  for (const auto &group : meanError)
    for (double error : group) {
      maxMean = std::max(maxMean, std::abs(error));
      ok &= std::abs(error) <= groupBound + 0.00011;
      maxBoundRatio =
          std::max(maxBoundRatio, std::abs(error) / (groupBound + 0.00011));
    }
  printf(
      "quality %s: max error=%g mean=%g bounds=%g/%g (%d spp) max-bound-ratio=%g%s\n",
      label,
      maxError,
      maxMean,
      pixelBound,
      groupBound,
      samples,
      maxBoundRatio,
      ok ? "" : " FAILED");
  fflush(stdout);
  return ok;
}

int testQualityProxyCoverage(Scene &scene)
{
  logCase("quality/proxy-coverage");
  const auto d = scene.d;
  configureObliqueRayBuffer(scene);
  auto light = quadLight(d);
  const ObjectOwner lightOwner(d, light);
  anari::setParameter(d, light, "position", Vec3{-20.f, -20.f, 10.f});
  anari::setParameter(d, light, "edge1", Vec3{40.f, 0.f, 0.f});
  anari::setParameter(d, light, "edge2", Vec3{0.f, 40.f, 0.f});
  anari::commitParameters(d, light);
  int failures = 0;
  for (float coverage : {0.f, 0.25f}) {
    // Zero-opacity camera pass-through has no sampled coverage decision.
    // The fractional witness keeps its independent 4096-spp coverage anchor.
    const int samples = coverage == 0.f ? 1 : QUALITY_SAMPLES;
    const std::string path =
        coverage == 0.f ? "proxy-pass-through" : "proxy-fractional";
    caseLabel(scene, path, "baseline");
    auto front = plane(d, 2.5f, {}, 0.f, false, 20.f, coverage);
    const ObjectOwner frontOwner(d, front);
    scene.attach(front, light);
    scene.fog("none");
    const auto baseline = scene.render(samples);
    auto expected = baseline.color;
    for (unsigned p = 0; p < PIXELS; ++p) {
      expected[p] = BACKDROP;
      if (!isMiss(p)) {
        for (unsigned c = 0; c < 3; ++c)
          expected[p][c] = (1. - coverage) * RADIANCE[c];
        expected[p][3] = 1.f;
      }
    }
    failures += !expectQuality("proxy coverage baseline",
        baseline,
        expected,
        coverage > 0.f ? 1. : 0.,
        samples);
    for (const auto &setting : QUALITY_CASES) {
      if (setting.representativeOnly)
        continue;
      caseLabel(scene, path, setting.name);
      qualityFog(scene, setting.radial, setting.background, setting.full);
      const auto image = scene.render(samples);
      const auto target = qualityTarget(setting.background);
      for (unsigned p = 0; p < PIXELS; ++p) {
        if (isMiss(p))
          continue;
        const double f = qualityFraction(p, 10.f, setting.radial, setting.full);
        const double frontF =
            qualityFraction(p, 2.5f, setting.radial, setting.full);
        for (unsigned c = 0; c < 3; ++c)
          expected[p][c] = coverage * frontF * target[c]
              + (1. - coverage) * ((1. - f) * RADIANCE[c] + f * target[c]);
      }
      failures += !expectQuality("proxy through coverage",
          image,
          expected,
          coverage > 0.f ? 1. : 0.,
          samples);
      failures += !sameAuxiliary(baseline, image, scene.subtype);
    }
    caseLabel(scene, path, "disabled-restoration");
    scene.fog("none");
    const auto restored = scene.render(samples);
    failures += !check(
        restored.color == baseline.color, scene, "proxy disable restores RGB");
    failures += !sameAuxiliary(baseline, restored, scene.subtype);
    // Primary visibility must survive even a zero-opacity pass-through.
    anari::setParameter(d, light, "visible", false);
    anari::commitParameters(d, light);
    scene.fog("none");
    caseLabel(scene, path, "hidden-baseline");
    const auto hidden = scene.render(samples);
    qualityFog(scene, true, false, true);
    caseLabel(scene, path, "hidden-full");
    const auto hiddenFog = scene.render(samples);
    expected = hidden.color;
    for (unsigned p = 0; p < PIXELS; ++p)
      if (!isMiss(p))
        for (unsigned c = 0; c < 3; ++c)
          expected[p][c] += coverage * FOG_COLOR[c];
    failures += !expectQuality(
        "hidden proxy through coverage", hiddenFog, expected, 0., samples);
    failures += !sameAuxiliary(hidden, hiddenFog, scene.subtype);
    anari::setParameter(d, light, "visible", true);
    anari::commitParameters(d, light);
  }
  return failures;
}

int testQualityVolume(Scene &scene, bool scattering)
{
  logCase(
      std::string("quality/volume/scattering=") + std::to_string(scattering));
  const auto d = scene.d;
  configureObliqueRayBuffer(scene);
  anari::setParameter(
      d, scene.renderer, "ambientRadiance", scattering ? 1.f : 0.f);
  auto volume =
      slab(d, 2.f, scattering ? Vec3{0.4f, 0.3f, 0.2f} : Vec3{}, 20.f);
  const ObjectOwner volumeOwner(d, volume);
  scene.attach({}, {}, volume);
  scene.fog("none");
  const std::string medium = scattering ? "scattering" : "absorbing";
  caseLabel(scene, medium + "/volume-only", "baseline");
  const auto only = scene.render(QUALITY_SAMPLES);
  // sigma_t = -log(1-opacity)/unitDistance = log(2). The oblique rays
  // cross a two-unit slab with cos(theta)=4/5: P(no collision)=2^(-2.5).
  const double survival = std::pow(2., -2.5);
  auto expected = only.color;
  double volumeRadiance = 0.;
  bool metadata = true;
  for (unsigned p = 0; p < PIXELS; ++p) {
    if (isMiss(p)) {
      expected[p] = BACKDROP;
      continue;
    }
    volumeRadiance += (only.color[p][0] - survival * BACKDROP[0]) / 240.;
    metadata &= only.object[p] == 77u && only.primitive[p] == 77u
        && std::isfinite(only.depth[p]) && only.depth[p] > 0.f;
    for (unsigned c = 0; c < 3; ++c)
      if (!scattering)
        expected[p][c] = survival * BACKDROP[c];
    expected[p][3] = 1. - survival * (1. - BACKDROP[3]);
  }
  int failures =
      !expectQuality("volume-only Beer survival", only, expected, 1.);
  failures += !check(metadata, scene, "volume first-collision metadata");
  if (scattering)
    failures += !check(
        volumeRadiance > 0.01, scene, "nonzero scattered volume radiance");
  printf("quality volume-only scattering=%d: excess red=%g survival=%g\n",
      int(scattering),
      volumeRadiance,
      survival);
  for (const auto &setting : QUALITY_CASES) {
    if (!setting.full)
      continue;
    caseLabel(scene, medium + "/volume-only", setting.name);
    qualityFog(scene, setting.radial, setting.background, true);
    const auto active = scene.render(QUALITY_SAMPLES);
    failures += !check(
        only.color == active.color, scene, "sampled volume-only RGB unchanged");
    failures += !sameAuxiliary(only, active, scene.subtype);
  }

  // Absorbing media isolate the transmitted emitter against a fully
  // independent radiance reference. Scattering media instead use a black
  // receiver: its unfogged radiance is zero, and volume lighting cancels in
  // paired images. Only the unscattered camera fraction gains survival * F * B.
  // For a proxy of known radiance S the paired difference is survival * F *
  // (B-S); its illumination of the volume still cancels. A later surface hit
  // after a volume scatter must NOT inject its own fog color.
  auto surface =
      plane(d, 10.f, scattering ? Vec3{} : RADIANCE, 0.f, !scattering, 20.f);
  const ObjectOwner surfaceOwner(d, surface);
  auto light = quadLight(d);
  const ObjectOwner lightOwner(d, light);
  anari::setParameter(d, light, "position", Vec3{-20.f, -20.f, 10.f});
  anari::setParameter(d, light, "edge1", Vec3{40.f, 0.f, 0.f});
  anari::setParameter(d, light, "edge2", Vec3{0.f, 40.f, 0.f});
  anari::commitParameters(d, light);
  for (bool proxy : {false, true}) {
    const std::string path = medium + (proxy ? "/proxy" : "/surface");
    caseLabel(scene, path, "baseline");
    scene.attach(proxy ? anari::Surface{} : surface,
        proxy ? light : anari::Light{},
        volume);
    scene.fog("none");
    const auto baseline = scene.render(QUALITY_SAMPLES);
    if (!scattering) {
      expected = baseline.color;
      for (unsigned p = 0; p < PIXELS; ++p)
        if (!isMiss(p))
          expected[p] = {float(survival * RADIANCE[0]),
              float(survival * RADIANCE[1]),
              float(survival * RADIANCE[2]),
              1.f};
      failures += !expectQuality(
          proxy ? "absorbed proxy baseline" : "absorbed surface baseline",
          baseline,
          expected,
          1.);
    }
    for (const auto &setting : QUALITY_CASES) {
      if (setting.representativeOnly && (scattering || proxy))
        continue;
      caseLabel(scene, path, setting.name);
      qualityFog(scene, setting.radial, setting.background, setting.full);
      const auto image = scene.render(QUALITY_SAMPLES);
      const auto target = qualityTarget(setting.background);
      expected = baseline.color;
      for (unsigned p = 0; p < PIXELS; ++p) {
        if (isMiss(p))
          continue;
        const double f = qualityFraction(p, 10.f, setting.radial, setting.full);
        for (unsigned c = 0; c < 3; ++c)
          expected[p][c] = scattering
              ? baseline.color[p][c]
                  + survival * f * (target[c] - (proxy ? RADIANCE[c] : 0.f))
              : survival * ((1. - f) * RADIANCE[c] + f * target[c]);
      }
      failures += !expectQuality(scattering
              ? (proxy ? "proxy with scattering" : "surface after scattering")
              : proxy ? "proxy behind absorbing volume"
                      : "surface behind absorbing volume",
          image,
          expected,
          1.);
      failures += !sameAuxiliary(baseline, image, scene.subtype);
    }
    caseLabel(scene, path, "disabled-restoration");
    scene.fog("none");
    const auto restored = scene.render(QUALITY_SAMPLES);
    failures += !check(restored.color == baseline.color,
        scene,
        "volume path disable restores RGB");
    failures += !sameAuxiliary(baseline, restored, scene.subtype);
  }
  return failures;
}

enum class Contribution
{
  BACKDROP,
  SURFACE,
  PROXY
};

// Named composition contracts for straight-through Renderers. Oblique supplied
// rays distinguish the two metrics; the black homogeneous slab has exactly
// forty march steps and transmission 2^(-2.5), independent of sample jitter.
int testStraightThroughContracts(Scene &scene)
{
  logCase(std::string(scene.subtype) + "/straight-through");
  const auto d = scene.d;
  configureObliqueRayBuffer(scene);
  anari::setParameter(d, scene.renderer, "ambientRadiance", 1.f);
  anari::setParameter(d, scene.renderer, "volumeSamplingRate", 1.f);
  anari::setParameter(d, scene.renderer, "fogDensity", 0.05f);
  auto volume = slab(d, 2.f, {}, 20.f);
  const ObjectOwner volumeOwner(d, volume);
  auto surface = plane(d, 10.f, RADIANCE, 0.f, false, 20.f);
  const ObjectOwner surfaceOwner(d, surface);
  auto light = quadLight(d);
  const ObjectOwner lightOwner(d, light);
  anari::setParameter(d, light, "position", Vec3{-20.f, -20.f, 10.f});
  anari::setParameter(d, light, "edge1", Vec3{40.f, 0.f, 0.f});
  anari::setParameter(d, light, "edge2", Vec3{0.f, 40.f, 0.f});
  anari::commitParameters(d, light);
  int failures = 0;
  double maximumError = 0.;
  unsigned cases = 0;
  for (bool hasVolume : {false, true}) {
    for (Contribution contribution :
        {Contribution::BACKDROP, Contribution::SURFACE, Contribution::PROXY}) {
      // Fast's geometry-only visibility is asserted as backdrop, not skipped.
      const bool hit = contribution == Contribution::SURFACE
          || (contribution == Contribution::PROXY
              && std::string(scene.subtype) != "fast");
      const std::string path = std::string(hasVolume ? "volume/" : "clear/")
          + (contribution == Contribution::SURFACE      ? "surface"
                  : contribution == Contribution::PROXY ? "proxy"
                                                        : "backdrop");
      caseLabel(scene, path, "baseline");
      scene.attach(
          contribution == Contribution::SURFACE ? surface : anari::Surface{},
          contribution == Contribution::PROXY ? light : anari::Light{},
          hasVolume ? volume : anari::Volume{});
      scene.fog("none");
      const auto baseline = scene.render();
      for (const auto &setting : STRAIGHT_CASES) {
        if (setting.representativeOnly
            && !(hasVolume && contribution == Contribution::SURFACE))
          continue;
        caseLabel(scene, path, setting.name);
        const auto *mode = setting.mode;
        const bool radial = setting.radial;
        const bool background = setting.background;
        const bool full = setting.full;
        qualityFog(scene, radial, background, full);
        scene.fog(mode, 0.f, full ? 0.01f : 20.f);
        const auto image = scene.render();
        const auto target = qualityTarget(background);
        bool ok = true;
        for (unsigned p = 0; p < PIXELS; ++p) {
          Vec4 expected = BACKDROP;
          if (!isMiss(p)) {
            const double survival = hasVolume ? std::pow(2., -2.5) : 1.;
            const double distance = radial ? (10. - 2 * (p % 2)) * 1.25 : 11.;
            const std::string curve(mode);
            const double visibility = curve == "none" ? 1.
                : full                                ? 0.
                : curve == "linear"
                ? 1. - distance / 20.
                : std::exp(curve == "exp" ? -0.05 * distance
                                          : -std::pow(0.05 * distance, 2));
            for (unsigned c = 0; c < 3; ++c)
              expected[c] = survival
                  * (hit ? visibility * RADIANCE[c]
                              + (1. - visibility) * target[c]
                         : BACKDROP[c]);
            expected[3] = hit ? 1. : 1. - survival * (1. - BACKDROP[3]);
          }
          for (unsigned c = 0; c < 4; ++c) {
            const double error =
                std::abs(double(image.color[p][c]) - expected[c]);
            maximumError = std::max(maximumError, error);
            ok &= std::isfinite(error)
                && error <= 1e-5 + 1e-4 * std::abs(expected[c]);
          }
        }
        if (!ok)
          fprintf(stderr,
              "%s cross interaction: volume=%d contribution=%d metric=%s source=%s curve=%s\n",
              scene.subtype,
              int(hasVolume),
              int(contribution),
              radial ? "rayDistance" : "viewDepth",
              background ? "background" : "constant",
              mode);
        failures += !ok;
        failures += !sameAuxiliary(baseline, image, scene.subtype);
        if (!hit || std::string(mode) == "none")
          failures += !check(image.color == baseline.color,
              scene,
              "volume-only/miss or disabled output unchanged");
        ++cases;
      }
    }
  }
  printf(
      "%s straight-through interaction contracts: %u cases, max absolute error=%g\n",
      scene.subtype,
      cases,
      maximumError);
  return failures;
}

} // namespace

int main()
{
  auto d = makeVisRTXDevice(exitOnErrorStatus);
  const ObjectOwner deviceOwner(d, d);
  if (!d)
    return 1;
  anari::commitParameters(d, d);
  int failures = requireRendererFogSupport(d);
  for (const char *subtype : {"fast", "interactive", "default"}) {
    Scene scene(d, subtype);
    failures += testProxy(scene);
    failures += testIllumination(scene);
    failures += testVolume(scene);
    failures += testJitteredVolume(scene);
    failures += testStraightThroughContracts(scene);
  }
  {
    Scene scene(d, "quality");
    failures += testProxy(scene);
    failures += testIllumination(scene);
    failures += testQualityProxyCoverage(scene);
    failures += testQualityVolume(scene, false);
    failures += testQualityVolume(scene, true);
  }
  printf("Fog interactions: %d failures\n", failures);
  return failures ? 1 : 0;
}

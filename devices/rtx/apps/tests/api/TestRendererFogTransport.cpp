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

constexpr UVec2 SIZE{8, 8};
constexpr unsigned PIXELS = SIZE[0] * SIZE[1];
constexpr Vec3 FOG{0.3f, 0.9f, 0.2f};
constexpr Vec3 EMISSION{1.6f, 0.8f, 1.2f};
constexpr Vec3 TINT{0.6f, 0.8f, 0.4f};
constexpr int SAMPLES = 4096;
constexpr float FOG_END = 20.f;
constexpr float FOG_DENSITY = 0.1f;

/* Public-ANARI transport fixture with paired camera samples and controlled
 * indirect radiance; no production fog evaluator defines its references.
 */
struct Scene
{
  Scene(anari::Device device);
  ~Scene();
  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;
  anari::Material plane(float z,
      Vec3 tint,
      Vec3 emission,
      float transmission = 0.f,
      float opacity = 1.f,
      bool metal = false);
  void fog(const char *mode, const char *metric, bool background);
  Image render();
  Vec3 backdrop(unsigned pixel) const;

  anari::Device d;
  anari::World world;
  anari::Renderer renderer;
  anari::Camera camera;
  anari::Frame frame;
  std::vector<anari::Surface> surfaces;
  std::vector<anari::Material> materials;
};

Scene::Scene(anari::Device device) : d(device)
{
  world = anari::newObject<anari::World>(d);
  ObjectOwner worldOwner(d, world);
  renderer = anari::newObject<anari::Renderer>(d, "quality");
  ObjectOwner rendererOwner(d, renderer);
  anari::setParameter(d, renderer, "denoise", false);
  anari::setParameter(d, renderer, "fireflyFilterMode", "none");
  anari::setParameter(d, renderer, "ambientRadiance", 0.f);
  anari::setParameter(d, renderer, "pixelSamples", SAMPLES);
  anari::setParameter(d, renderer, "maxRayDepth", 8);
  anari::setParameter(d, renderer, "fogColor", FOG);
  // A spatially varying, transparent image keeps the background out of
  // transmitted-path alpha while making screen-location drift observable.
  auto image = anari::newArray2D(d, ANARI_FLOAT32_VEC4, SIZE[0], SIZE[1]);
  const ObjectOwner imageOwner(d, image);
  {
    const ArrayMapping<Vec4> pixels(d, image);
    for (unsigned p = 0; p < PIXELS; ++p) {
      const auto rgb = backdrop(p);
      pixels.data()[p] = {rgb[0], rgb[1], rgb[2], 0.f};
    }
  }
  anari::setParameter(d, renderer, "background", image);
  anari::setParameter(d, renderer, "premultiplyBackground", false);
  anari::commitParameters(d, renderer);

  camera = anari::newObject<anari::Camera>(d, "rayBuffer");
  ObjectOwner cameraOwner(d, camera);
  auto origins = anari::newArray2D(d, ANARI_FLOAT32_VEC3, SIZE[0], SIZE[1]);
  const ObjectOwner originsOwner(d, origins);
  auto directions = anari::newArray2D(d, ANARI_FLOAT32_VEC3, SIZE[0], SIZE[1]);
  const ObjectOwner directionsOwner(d, directions);
  {
    const ArrayMapping<Vec3> o(d, origins);
    const ArrayMapping<Vec3> v(d, directions);
    for (unsigned p = 0; p < PIXELS; ++p) {
      o.data()[p] = {0.f, 0.f, float(2 * (p % 2))};
      v.data()[p] = {3.f, 0.f, 4.f};
    }
  }
  anari::setParameter(d, camera, "ray.org", origins);
  anari::setParameter(d, camera, "ray.dir", directions);
  anari::setParameter(d, camera, "position", Vec3{0.f, 0.f, -1.f});
  anari::setParameter(d, camera, "direction", Vec3{0.f, 0.f, 1.f});
  anari::commitParameters(d, camera);
  frame = anari::newObject<anari::Frame>(d);
  ObjectOwner frameOwner(d, frame);
  anari::setParameter(d, frame, "size", SIZE);
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(d, frame, "channel.normal", ANARI_FLOAT32_VEC3);
  anari::setParameter(d, frame, "channel.albedo", ANARI_FLOAT32_VEC3);
  for (const char *name :
      {"channel.primitiveId", "channel.objectId", "channel.instanceId"})
    anari::setParameter(d, frame, name, ANARI_UINT32);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", camera);
  anari::setParameter(d, frame, "renderer", renderer);
  anari::commitParameters(d, frame);
  worldOwner.release();
  rendererOwner.release();
  cameraOwner.release();
  frameOwner.release();
}

Scene::~Scene()
{
  anari::release(d, frame);
  anari::release(d, camera);
  anari::release(d, renderer);
  anari::release(d, world);
  for (auto surface : surfaces)
    anari::release(d, surface);
  for (auto material : materials)
    anari::release(d, material);
}

Vec3 Scene::backdrop(unsigned p) const
{
  return {0.1f + 0.05f * (p % SIZE[0]), 0.2f + 0.03f * (p / SIZE[0]), 0.7f};
}

anari::Material Scene::plane(float z,
    Vec3 tint,
    Vec3 emission,
    float transmission,
    float opacity,
    bool metal)
{
  auto geometry = anari::newObject<anari::Geometry>(d, "quad");
  const ObjectOwner geometryOwner(d, geometry);
  // Face the camera: each dielectric interface uses the entering IOR ratio,
  // rather than repeated exiting interfaces that would cause total reflection.
  const std::array<Vec3, 4> vertices = {Vec3{-100.f, -100.f, z},
      Vec3{-100.f, 100.f, z},
      Vec3{100.f, 100.f, z},
      Vec3{100.f, -100.f, z}};
  anari::setParameterArray1D(
      d, geometry, "vertex.position", vertices.data(), 4);
  anari::commitParameters(d, geometry);
  auto material = anari::newObject<anari::Material>(d, "physicallyBased");
  ObjectOwner materialOwner(d, material);
  anari::setParameter(d, material, "baseColor", tint);
  anari::setParameter(d, material, "emissive", emission);
  anari::setParameter(d, material, "metallic", metal ? 1.f : 0.f);
  anari::setParameter(d, material, "specular", 0.f);
  anari::setParameter(d, material, "roughness", 0.02f);
  anari::setParameter(d, material, "ior", 1.5f);
  anari::setParameter(d, material, "transmission", transmission);
  anari::setParameter(d, material, "opacity", opacity);
  anari::commitParameters(d, material);
  materials.push_back(material);
  materialOwner.release();
  auto surface = anari::newObject<anari::Surface>(d);
  ObjectOwner surfaceOwner(d, surface);
  anari::setParameter(d, surface, "geometry", geometry);
  anari::setParameter(d, surface, "material", material);
  anari::setParameter(d, surface, "id", unsigned(100 + surfaces.size()));
  anari::commitParameters(d, surface);
  surfaces.push_back(surface);
  surfaceOwner.release();
  // Explicit instance ID, so all ID-channel comparisons are non-vacuous.
  auto group = anari::newObject<anari::Group>(d);
  const ObjectOwner groupOwner(d, group);
  anari::setParameterArray1D(
      d, group, "surface", surfaces.data(), surfaces.size());
  anari::commitParameters(d, group);
  auto instance = anari::newObject<anari::Instance>(d, "transform");
  const ObjectOwner instanceOwner(d, instance);
  anari::setParameter(d, instance, "group", group);
  anari::setParameter(d, instance, "id", 301u);
  anari::commitParameters(d, instance);
  anari::setParameterArray1D(d, world, "instance", &instance, 1);
  anari::commitParameters(d, world);
  return material;
}

void Scene::fog(const char *mode, const char *metric, bool background)
{
  anari::setParameter(
      d, renderer, "fogMode", std::string(mode) == "full" ? "linear" : mode);
  anari::setParameter(
      d, renderer, "fogEnd", std::string(mode) == "full" ? 0.01f : FOG_END);
  anari::setParameter(d, renderer, "fogDensity", FOG_DENSITY);
  anari::setParameter(d, renderer, "fogDistanceMetric", metric);
  anari::setParameter(
      d, renderer, "fogColorSource", background ? "background" : "constant");
  anari::commitParameters(d, renderer);
}

Image Scene::render()
{
  anari::render(d, frame);
  anari::wait(d, frame);
  return readImage(d, frame, SIZE);
}

double visibility(const char *mode, double distance)
{
  if (std::string(mode) == "full")
    return 0.;
  return referenceVisibility(mode, distance, 0., FOG_END, FOG_DENSITY);
}

bool expect(const std::string &label,
    const Image &image,
    const std::vector<Vec4> &reference,
    bool stochastic = false,
    int samples = SAMPLES)
{
  double maxError = 0., meanErrors[2][3] = {};
  bool passed = true;
  for (unsigned p = 0; p < PIXELS; ++p) {
    for (unsigned c = 0; c < 3; ++c) {
      const double error = image.color[p][c] - reference[p][c];
      if (!std::isfinite(error))
        return false;
      maxError = std::fmax(maxError, std::abs(error));
      meanErrors[p % 2][c] += error / (PIXELS / 2);
      const double limit =
          stochastic ? 0.05 : 1e-5 + 1e-4 * std::abs(reference[p][c]);
      passed &= std::abs(error) <= limit;
    }
  }
  double maxMeanError = 0.;
  for (auto &group : meanErrors)
    for (double error : group)
      maxMeanError = std::fmax(maxMeanError, std::abs(error));
  if (stochastic)
    passed &= maxMeanError <= 0.009;
  printf("%s: %d spp, max RGB error=%g, grouped mean error=%g%s\n",
      label.c_str(),
      samples,
      maxError,
      maxMeanError,
      passed ? "" : " FAILED");
  return passed;
}

// The same committed scene/renderer begins at accumulation frame zero for each
// setting, pairing transport random streams without a private seed interface.
// The expected affine operation is applied to the *unfogged* shaded result;
// no production fog helper or resolved-depth channel defines the reference.
int testTransport(anari::Device d,
    bool transmission,
    float emitterZ,
    float coverage,
    unsigned panes,
    bool proxy = false)
{
  Scene scene(d);
  // Coverage rejection is camera visibility, including numerical re-origining.
  // The zero-opacity case also ensures a discarded contribution injects no fog.
  scene.plane(3.f, {}, {}, 0.f, coverage);
  for (unsigned i = 0; i < panes; ++i)
    scene.plane(
        5.f + 2.f * i, TINT, {}, transmission ? 1.f : 0.f, 1.f, !transmission);
  if (proxy) {
    auto light = anari::newObject<anari::Light>(d, "quad");
    const ObjectOwner lightOwner(d, light);
    anari::setParameter(d, light, "position", Vec3{-100.f, -100.f, emitterZ});
    anari::setParameter(d, light, "edge1", Vec3{200.f, 0.f, 0.f});
    anari::setParameter(d, light, "edge2", Vec3{0.f, 200.f, 0.f});
    anari::setParameter(d, light, "side", "both");
    anari::setParameter(d, light, "visible", false);
    anari::setParameter(d, light, "color", EMISSION);
    anari::setParameter(d, light, "intensity", 1.f);
    anari::commitParameters(d, light);
    anari::setParameterArray1D(d, scene.world, "light", &light, 1);
    anari::commitParameters(d, scene.world);
  } else
    scene.plane(emitterZ, {}, EMISSION);
  scene.fog("none", "viewDepth", false);
  const auto baseline = scene.render();
  int failures = 0;
  if (baseline.object[0] != 100 || baseline.instance[0] != 301
      || baseline.primitive[0] == ~0u || !std::isfinite(baseline.depth[0])) {
    fprintf(stderr, "transport fixture has no valid first-hit metadata\n");
    ++failures;
  }
  // The unlit near plane cannot return the emitter without a real BSDF event.
  anari::setParameter(d, scene.renderer, "maxRayDepth", 1);
  scene.fog("none", "viewDepth", false);
  const auto directOnly = scene.render();
  anari::setParameter(d, scene.renderer, "maxRayDepth", 8);
  double indirect = 0.;
  for (unsigned p = 0; p < PIXELS; ++p) {
    const double background = scene.backdrop(p)[0];
    const double shaded =
        baseline.color[p][0] - (1. - baseline.color[p][3]) * background;
    const double direct =
        directOnly.color[p][0] - (1. - directOnly.color[p][3]) * background;
    indirect += (shaded - direct) / PIXELS;
  }
  if (indirect < 0.05) {
    fprintf(
        stderr, "transport fixture lacks indirect radiance: %g\n", indirect);
    ++failures;
  }
  printf("%s %s z=%g coverage=%g panes=%u: indirect red=%g\n",
      transmission ? "refraction" : "reflection",
      proxy ? "hidden proxy" : "emitter",
      emitterZ,
      coverage,
      panes,
      indirect);

  // A constant-radiance emitter fills the complete reflected/refracted field.
  // Moving it must not produce inverse-square dimming or change camera depth.
  // For the narrow GGX reflection, Schlick's grazing correction is < .001;
  // pure transmission has tint per pane. RR at the fourth pane is stochastic:
  // the .025 mean limit exceeds six conservative SEs for 64*4096 samples of
  // bounded <=1.6 radiance. This independent oracle also pins nonzero
  // transport.
  if (coverage == 0.f) {
    for (unsigned c = 0; c < 3; ++c) {
      double mean = 0.;
      for (unsigned p = 0; p < PIXELS; ++p)
        mean += (baseline.color[p][c]
                    - (1. - baseline.color[p][3]) * scene.backdrop(p)[c])
            / PIXELS;
      const double reference = EMISSION[c] * std::pow(TINT[c], panes);
      printf("controlled radiance channel %u: mean=%g expected=%g\n",
          c,
          mean,
          reference);
      if (std::abs(mean - reference) > 0.025)
        ++failures;
    }
  }

  for (bool background : {false, true}) {
    for (const char *metric : {"viewDepth", "rayDistance"}) {
      for (const char *mode : {"linear", "exp", "exp2", "full", "none"}) {
        scene.fog(mode, metric, background);
        const auto image = scene.render();
        auto expected = baseline.color;
        for (unsigned p = 0; p < PIXELS; ++p) {
          const bool radial = std::string(metric) == "rayDistance";
          const double distance = radial ? (5. - 2 * (p % 2)) * 1.25 : 6.;
          const double frontDistance = radial ? (3. - 2 * (p % 2)) * 1.25 : 4.;
          const double t = visibility(mode, distance);
          const double frontT = visibility(mode, frontDistance);
          const auto target = background ? scene.backdrop(p) : FOG;
          for (unsigned c = 0; c < 3; ++c) {
            // Existing frame backdrop compositing is not surface shading.
            const double miss =
                (1. - baseline.color[p][3]) * scene.backdrop(p)[c];
            expected[p][c] = t * (baseline.color[p][c] - miss) + miss
                + (coverage * (1. - frontT) + (1. - coverage) * (1. - t))
                    * target[c];
          }
        }
        // Only the additive rear-layer coverage term is unpaired. Its range
        // <=.9 gives six-SE bounds .043/pixel and .0075 per 32-pixel origin
        // group at 4096 spp; use .05/.009 including float accumulation error.
        const std::string label =
            std::string(transmission ? "refraction " : "reflection ") + mode
            + " " + metric + (background ? " image" : " constant")
            + " coverage=" + std::to_string(coverage) + " panes="
            + std::to_string(panes) + (proxy ? " hidden proxy" : " emitter");
        failures += !expect(label, image, expected, coverage > 0.f);
        failures += !sameAuxiliary(baseline, image, "transport");
        if (std::string(mode) == "none" && image.color != baseline.color) {
          fprintf(stderr, "disabled fog changed paired transport RGB\n");
          ++failures;
        }
      }
    }
  }
  return failures;
}

int testHdrRadiance(anari::Device d)
{
  int failures = 0;
  constexpr Vec3 HDR_EMISSION{1e38f, 5e37f, 2e37f};
  for (bool indirect : {false, true}) {
    Scene scene(d);
    // One paired sample avoids overflow in the unfogged accumulation sum.
    // The accepted surface is at view depth 10, including for reflection.
    anari::setParameter(d, scene.renderer, "pixelSamples", 1);
    anari::setParameter(d, scene.renderer, "fogColor", Vec3{});
    if (indirect)
      scene.plane(9.f, TINT, {}, 0.f, 1.f, true);
    scene.plane(indirect ? -3.f : 9.f, {}, HDR_EMISSION);
    scene.fog("none", "viewDepth", false);
    const auto baseline = scene.render();
    const std::string label = indirect ? "HDR reflection" : "HDR emission";
    if (indirect) {
      // Establish a live reflected HDR signal without a noisy mean oracle:
      // every paired pixel must see at least half of tint * emitter radiance.
      for (const auto &pixel : baseline.color)
        for (unsigned c = 0; c < 3; ++c)
          failures += !std::isfinite(pixel[c])
              || pixel[c] < 0.5 * TINT[c] * HDR_EMISSION[c];
    } else {
      const std::vector<Vec4> expected(
          PIXELS, {HDR_EMISSION[0], HDR_EMISSION[1], HDR_EMISSION[2], 1.f});
      failures += !expect(label + " unfogged", baseline, expected, false, 1);
    }
    for (const char *mode : {"exp", "exp2"}) {
      const float density = std::string(mode) == "exp" ? 9.f : 0.95f;
      scene.fog(mode, "viewDepth", false);
      anari::setParameter(d, scene.renderer, "fogDensity", density);
      anari::commitParameters(d, scene.renderer);
      const auto image = scene.render();
      // exp(-90), or exp(-(0.95*10)^2): FLOAT32 subnormal visibility,
      // but a normal, visible result after multiplication by HDR radiance.
      // Compute the public affine reference in long double, never through
      // the production evaluator or an intermediate FLOAT32 visibility.
      const long double opticalDistance = 10.L * density;
      const long double t = std::exp(std::string(mode) == "exp"
              ? -opticalDistance
              : -opticalDistance * opticalDistance);
      auto expected = baseline.color;
      for (unsigned p = 0; p < PIXELS; ++p)
        for (unsigned c = 0; c < 3; ++c)
          expected[p][c] = t * baseline.color[p][c];
      failures += !expect(label + " " + mode, image, expected, false, 1);
      failures += !sameAuxiliary(baseline, image, "transport");
    }
    scene.fog("none", "viewDepth", false);
    const auto restored = scene.render();
    failures += restored.color != baseline.color;
    failures += !sameAuxiliary(baseline, restored, "transport");
  }
  return failures;
}

int testIndirectVolume(anari::Device d)
{
  Scene scene(d);
  scene.plane(3.f, {}, {}, 0.f, 0.f);
  scene.plane(5.f, TINT, {}, 1.f);
  auto field = anari::newObject<anari::SpatialField>(d, "structuredRegular");
  const ObjectOwner fieldOwner(d, field);
  auto data = anari::newArray3D(d, ANARI_FLOAT32, 17, 17, 17);
  const ObjectOwner dataOwner(d, data);
  {
    const ArrayMapping<float> values(d, data);
    std::fill(values.data(), values.data() + 17 * 17 * 17, 0.5f);
  }
  anari::setParameter(d, field, "data", data);
  anari::setParameter(d, field, "origin", Vec3{-20.f, -20.f, 8.f});
  anari::setParameter(d, field, "spacing", Vec3{2.5f, 2.5f, 0.125f});
  anari::commitParameters(d, field);
  auto volume = anari::newObject<anari::Volume>(d, "transferFunction1D");
  const ObjectOwner volumeOwner(d, volume);
  anari::setParameter(d, volume, "value", field);
  anari::setParameter(d, volume, "color", Vec3{0.4f, 0.3f, 0.2f});
  anari::setParameter(d, volume, "opacity", 0.5f);
  anari::setParameter(d, volume, "unitDistance", 1.f);
  anari::setParameter(d, volume, "id", 77u);
  anari::commitParameters(d, volume);
  anari::setParameterArray1D(d, scene.world, "volume", &volume, 1);
  anari::commitParameters(d, scene.world);
  anari::setParameter(d, scene.renderer, "ambientRadiance", 1.f);
  scene.fog("none", "viewDepth", false);
  const auto baseline = scene.render();
  anari::setParameter(d, scene.renderer, "maxRayDepth", 1);
  scene.fog("none", "viewDepth", false);
  const auto directOnly = scene.render();
  anari::setParameter(d, scene.renderer, "maxRayDepth", 8);
  double indirect = 0.;
  for (unsigned p = 0; p < PIXELS; ++p) {
    const double backdrop = scene.backdrop(p)[0];
    indirect +=
        (baseline.color[p][0] - directOnly.color[p][0]
            + (baseline.color[p][3] - directOnly.color[p][3]) * backdrop)
        / PIXELS;
  }
  printf("refracted volume: indirect red=%g (%d spp)\n", indirect, SAMPLES);
  int failures = 0;
  if (!(indirect > 0.01)) {
    fprintf(
        stderr, "refracted volume fixture lacks scattered indirect radiance\n");
    ++failures;
  }
  // Analytic affine reference of the complete paired shaded result, including
  // volume NEE and subsequent scattering/RR. Only camera-surface distance is
  // used, never a sampled volume position. Existing backdrop compositing is
  // not surface radiance; leave that term unchanged even through refraction.
  for (bool background : {false, true}) {
    for (const char *metric : {"viewDepth", "rayDistance"}) {
      for (const char *mode : {"linear", "exp", "exp2", "full", "none"}) {
        scene.fog(mode, metric, background);
        const auto image = scene.render();
        auto expected = baseline.color;
        for (unsigned p = 0; p < PIXELS; ++p) {
          const double distance = std::string(metric) == "rayDistance"
              ? (5. - 2 * (p % 2)) * 1.25
              : 6.;
          const double t = visibility(mode, distance);
          const auto target = background ? scene.backdrop(p) : FOG;
          for (unsigned c = 0; c < 3; ++c) {
            const double miss =
                (1. - baseline.color[p][3]) * scene.backdrop(p)[c];
            expected[p][c] =
                t * (baseline.color[p][c] - miss) + miss + (1. - t) * target[c];
          }
        }
        failures += !expect(std::string("indirect volume ") + mode + " "
                + metric + (background ? " image" : " constant"),
            image,
            expected);
        failures += !sameAuxiliary(baseline, image, "transport");
        if (std::string(mode) == "none" && image.color != baseline.color)
          ++failures;
      }
    }
  }
  return failures;
}

} // namespace

int main()
{
  auto device = makeVisRTXDevice(exitOnErrorStatus);
  const ObjectOwner deviceOwner(device, device);
  int failures = requireRendererFogSupport(device);
  for (float emitterZ : {-3.f, -19.f})
    failures += testTransport(device, false, emitterZ, 0.f, 1);
  for (float emitterZ : {13.f, 29.f})
    failures += testTransport(device, true, emitterZ, 0.f, 1);
  failures += testTransport(device, false, -19.f, 0.25f, 1);
  failures += testTransport(device, true, 29.f, 0.25f, 1);
  failures += testTransport(device, true, 29.f, 0.f, 4);
  for (float z : {-3.f, -19.f})
    failures += testTransport(device, false, z, 0.f, 1, true);
  for (float z : {13.f, 29.f})
    failures += testTransport(device, true, z, 0.f, 1, true);
  failures += testIndirectVolume(device);
  failures += testHdrRadiance(device);
  if (failures)
    fprintf(stderr, "%d fog transport failure(s)\n", failures);
  else
    printf("quality fog transport passed\n");
  return failures ? 1 : 0;
}

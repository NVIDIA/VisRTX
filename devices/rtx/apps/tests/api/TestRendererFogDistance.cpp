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
#include <string>
#include <vector>

namespace {

using namespace visrtx::fogtest;

constexpr UVec2 SIZE = {4, 4};
constexpr unsigned PIXELS = SIZE[0] * SIZE[1];
constexpr float FOG_END = 20.f;
constexpr float FOG_DENSITY = 0.05f;

/* Small public-ANARI fixture, deliberately independent of device headers.
 * Black full-coverage planes isolate the fog fraction from shading estimators.
 */
struct Scene
{
  Scene(anari::Device device, const char *subtype, const char *cameraType);
  ~Scene();
  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;
  void plane(float z, Vec3 translation = {}, float scale = 1.f);
  void addCoveragePlane(float z, float opacity);
  std::vector<Vec4> render();
  void fog(const char *mode, const char *metric, float scale = 1.f);

  anari::Device device;
  const char *subtype;
  anari::World world;
  anari::Geometry geometry;
  anari::Surface surface;
  anari::Camera camera;
  anari::Renderer renderer;
  anari::Frame frame;
};

Scene::Scene(anari::Device d, const char *s, const char *cameraType)
    : device(d), subtype(s)
{
  auto material = anari::newObject<anari::Material>(d, "matte");
  const ObjectOwner materialOwner(d, material);
  anari::setParameter(d, material, "color", Vec3{0.f, 0.f, 0.f});
  anari::commitParameters(d, material);
  geometry = anari::newObject<anari::Geometry>(d, "quad");
  ObjectOwner geometryOwner(d, geometry);
  plane(10.f);
  surface = anari::newObject<anari::Surface>(d);
  ObjectOwner surfaceOwner(d, surface);
  anari::setParameter(d, surface, "geometry", geometry);
  anari::setParameter(d, surface, "material", material);
  anari::setParameter(d, surface, "id", 42u);
  anari::commitParameters(d, surface);
  world = anari::newObject<anari::World>(d);
  ObjectOwner worldOwner(d, world);
  anari::setParameterArray1D(d, world, "surface", &surface, 1);
  anari::commitParameters(d, world);
  camera = anari::newObject<anari::Camera>(d, cameraType);
  ObjectOwner cameraOwner(d, camera);
  anari::setParameter(d, camera, "position", Vec3{0.f, 0.f, 0.f});
  anari::setParameter(d, camera, "direction", Vec3{0.f, 0.f, 3.f});
  anari::setParameter(d, camera, "up", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, camera, "fovy", float(std::acos(-1.) / 2.));
  anari::setParameter(d, camera, "height", 20.f);
  anari::commitParameters(d, camera);
  renderer = anari::newObject<anari::Renderer>(d, s);
  ObjectOwner rendererOwner(d, renderer);
  anari::setParameter(d, renderer, "denoise", false);
  anari::setParameter(d, renderer, "fireflyFilterMode", "none");
  anari::setParameter(d, renderer, "ambientSamples", 0);
  anari::setParameter(d, renderer, "background", Vec4{0.f, 0.f, 0.f, 0.f});
  anari::commitParameters(d, renderer);
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
  geometryOwner.release();
  surfaceOwner.release();
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
  anari::release(device, geometry);
  anari::release(device, surface);
}

void Scene::plane(float z, Vec3 t, float scale)
{
  const std::array<Vec3, 4> positions = {
      Vec3{t[0] - 100.f * scale, t[1] - 100.f * scale, t[2] + z * scale},
      Vec3{t[0] + 100.f * scale, t[1] - 100.f * scale, t[2] + z * scale},
      Vec3{t[0] + 100.f * scale, t[1] + 100.f * scale, t[2] + z * scale},
      Vec3{t[0] - 100.f * scale, t[1] + 100.f * scale, t[2] + z * scale}};
  anari::setParameterArray1D(
      device, geometry, "vertex.position", positions.data(), positions.size());
  anari::commitParameters(device, geometry);
}

void Scene::addCoveragePlane(float z, float opacity)
{
  auto g = anari::newObject<anari::Geometry>(device, "quad");
  const ObjectOwner geometryOwner(device, g);
  const std::array<Vec3, 4> positions = {Vec3{-100.f, -100.f, z},
      Vec3{100.f, -100.f, z},
      Vec3{100.f, 100.f, z},
      Vec3{-100.f, 100.f, z}};
  anari::setParameterArray1D(
      device, g, "vertex.position", positions.data(), positions.size());
  anari::commitParameters(device, g);
  auto m = anari::newObject<anari::Material>(device, "matte");
  const ObjectOwner materialOwner(device, m);
  anari::setParameter(device, m, "color", Vec3{0.f, 0.f, 0.f});
  anari::setParameter(device, m, "opacity", opacity);
  anari::commitParameters(device, m);
  auto front = anari::newObject<anari::Surface>(device);
  const ObjectOwner frontOwner(device, front);
  anari::setParameter(device, front, "geometry", g);
  anari::setParameter(device, front, "material", m);
  anari::commitParameters(device, front);
  const std::array<anari::Surface, 2> surfaces = {front, surface};
  anari::setParameterArray1D(
      device, world, "surface", surfaces.data(), surfaces.size());
  anari::commitParameters(device, world);
}

std::vector<Vec4> Scene::render()
{
  anari::render(device, frame);
  anari::wait(device, frame);
  return readChannel<Vec4>(device, frame, "channel.color", SIZE[0], SIZE[1]);
}

void Scene::fog(const char *mode, const char *metric, float scale)
{
  anari::setParameter(device, renderer, "fogMode", mode);
  if (metric)
    anari::setParameter(device, renderer, "fogDistanceMetric", metric);
  else
    anari::unsetParameter(device, renderer, "fogDistanceMetric");
  anari::setParameter(device, renderer, "fogEnd", FOG_END * scale);
  anari::setParameter(device, renderer, "fogDensity", FOG_DENSITY / scale);
  anari::commitParameters(device, renderer);
}

bool near(double actual, double expected)
{
  return std::isfinite(actual)
      && std::abs(actual - expected) <= 1e-5 + 1e-4 * std::abs(expected);
}

bool expect(const Scene &scene,
    const char *label,
    const std::vector<Vec4> &image,
    const std::vector<double> &fractions,
    const std::vector<double> &alpha = {})
{
  for (unsigned p = 0; p < PIXELS; ++p) {
    const auto &v = image[p];
    if (!near(v[0], fractions[p]) || !near(v[1], fractions[p])
        || !near(v[2], fractions[p])
        || !near(v[3], alpha.empty() ? 1. : alpha[p])) {
      fprintf(stderr,
          "%s %s pixel %u: (%g,%g,%g,%g), expected F=%g\n",
          scene.subtype,
          label,
          p,
          v[0],
          v[1],
          v[2],
          v[3],
          fractions[p]);
      return false;
    }
  }
  return true;
}

double fraction(const char *mode, double distance)
{
  return referenceFraction(mode, distance, 0., FOG_END, FOG_DENSITY);
}

int testPlanes(anari::Device d, const char *subtype)
{
  int failures = 0;
  for (const char *cameraType : {"perspective", "orthographic"}) {
    Scene scene(d, subtype, cameraType);
    for (const char *mode : {"linear", "exp", "exp2"}) {
      for (const char *metric : {"viewDepth", "rayDistance"}) {
        std::vector<double> expected(PIXELS);
        // 90-degree FOV: pixel-center intersections are (10*u,10*v,10).
        // Orthographic rays instead originate at each pixel's (x,y,0).
        for (unsigned y = 0; y < SIZE[1]; ++y) {
          for (unsigned x = 0; x < SIZE[0]; ++x) {
            const double u = 2. * (x + 0.5) / SIZE[0] - 1.;
            const double v = 2. * (y + 0.5) / SIZE[1] - 1.;
            const bool radial = std::string(cameraType) == "perspective"
                && std::string(metric) == "rayDistance";
            const double distance =
                radial ? 10. * std::sqrt(1. + u * u + v * v) : 10.;
            // The linear interval [2,22] exercises scaling of both endpoints.
            expected[y * SIZE[0] + x] = std::string(mode) == "linear"
                ? (distance - 2.) / 20.
                : fraction(mode, distance);
          }
        }
        for (const Vec3 translation : {Vec3{}, Vec3{13.f, -7.f, 31.f}}) {
          for (float scale : {1.f, 0.125f, 8.f}) {
            scene.plane(10.f, translation, scale);
            anari::setParameter(d, scene.camera, "position", translation);
            anari::setParameter(d, scene.camera, "height", 20.f * scale);
            anari::commitParameters(d, scene.camera);
            scene.fog(mode, metric, scale);
            anari::setParameter(d, scene.renderer, "fogStart", 2.f * scale);
            anari::setParameter(d, scene.renderer, "fogEnd", 22.f * scale);
            anari::commitParameters(d, scene.renderer);
            const std::string label = std::string(cameraType) + " " + mode + " "
                + metric + " translated/scaled=" + std::to_string(scale);
            failures += !expect(scene, label.c_str(), scene.render(), expected);
          }
        }
      }
    }
  }
  return failures;
}

Image auxiliary(const Scene &scene)
{
  return readImage(scene.device, scene.frame, SIZE);
}

template <typename T>
void buffer(Scene &scene, const char *name, const std::vector<T> &data)
{
  auto a = anari::newArray2D(scene.device, data.data(), SIZE[0], SIZE[1]);
  const ObjectOwner arrayOwner(scene.device, a);
  anari::setParameter(scene.device, scene.camera, name, a);
}

int testRayBuffer(anari::Device d, const char *subtype)
{
  Scene scene(d, subtype, "rayBuffer");
  std::vector<Vec3> origins(PIXELS), directions(PIXELS);
  std::vector<float> lower(PIXELS), upper(PIXELS, 100.f);
  for (unsigned p = 0; p < PIXELS; ++p) {
    origins[p] = {float(p % 4) - 1.5f, 0.f, float(p / 4)};
    directions[p] = {p % 2 ? 3.f : 0.f, 0.f, 4.f};
    const double distance = (10. - origins[p][2]) * (p % 2 ? 1.25 : 1.);
    lower[p] = float(distance * 0.4);
    if (p % 4 == 2)
      upper[p] = float(distance - 0.25);
    if (p % 4 == 3)
      lower[p] = float(distance + 0.25);
  }
  buffer(scene, "ray.org", origins);
  buffer(scene, "ray.dir", directions);
  buffer(scene, "ray.tmin", lower);
  buffer(scene, "ray.tmax", upper);
  int failures = 0;
  for (bool coverage : {false, true}) {
    // Quality coverage references belong to the stochastic-layer test suite.
    if (coverage && std::string(subtype) == "quality")
      continue;
    if (coverage) {
      // Keep the farther plane visible through 25% coverage. Advancing past
      // the front hit must not measure the back hit from that surface.
      scene.addCoveragePlane(5.f, 0.25f);
      lower.assign(PIXELS, 0.f);
      upper.assign(PIXELS, 100.f);
      buffer(scene, "ray.tmin", lower);
      buffer(scene, "ray.tmax", upper);
    }
    for (int reference = 0; reference < 3; ++reference) {
      const Vec3 position = reference == 0 ? Vec3{}
          : reference == 1                 ? Vec3{2.f, 0.f, 1.f}
                                           : Vec3{0.f, 0.f, 20.f};
      const Vec3 direction =
          reference == 1 ? Vec3{3.f, 0.f, 4.f} : Vec3{0.f, 0.f, 5.f};
      anari::setParameter(d, scene.camera, "position", position);
      anari::setParameter(d, scene.camera, "direction", direction);
      anari::commitParameters(d, scene.camera);
      for (bool cut : {false, true}) {
        // Isolate interval clipping from cutting-plane re-origining.
        if (!coverage && cut)
          continue;
        anari::setParameter(d,
            scene.renderer,
            "cutPlane",
            cut ? Vec4{0.f, 0.f, 1.f, -4.f} : Vec4{});
        scene.fog("none", nullptr);
        const auto baseline = scene.render();
        const auto baselineAux = auxiliary(scene);
        for (const char *metric : {"viewDepth", "rayDistance"}) {
          for (const char *mode : {"linear", "exp", "exp2"}) {
            scene.fog(mode, metric);
            std::vector<double> expected(PIXELS), alpha(PIXELS, 1.);
            for (unsigned p = 0; p < PIXELS; ++p) {
              if (!coverage && p % 4 >= 2) {
                alpha[p] = 0.;
                continue;
              }
              auto distanceAt = [&](double z) {
                if (std::string(metric) == "rayDistance")
                  return (z - origins[p][2]) * (p % 2 ? 1.25 : 1.);
                const double x = origins[p][0]
                    + (z - origins[p][2]) * directions[p][0] / directions[p][2];
                return std::fmax(0.,
                    ((x - position[0]) * direction[0]
                        + (z - position[2]) * direction[2])
                        / 5.);
              };
              expected[p] = coverage ? 0.25 * fraction(mode, distanceAt(5.))
                      + 0.75 * fraction(mode, distanceAt(10.))
                                     : fraction(mode, distanceAt(10.));
            }
            const std::string label = std::string("rayBuffer ") + mode + " "
                + metric + " reference=" + std::to_string(reference)
                + (coverage ? " coverage" : " intervals") + (cut ? " cut" : "");
            failures +=
                !expect(scene, label.c_str(), scene.render(), expected, alpha);
            if (!sameAuxiliary(baselineAux, auxiliary(scene), scene.subtype)) {
              fprintf(stderr,
                  "%s %s changed auxiliary channels\n",
                  subtype,
                  label.c_str());
              ++failures;
            }
          }
          scene.fog("none", metric);
          failures += scene.render() != baseline;
          failures +=
              !sameAuxiliary(baselineAux, auxiliary(scene), scene.subtype);
        }
        // Full fog must still retain the no-fog geometric metadata.
        scene.fog("linear", "rayDistance");
        anari::setParameter(d, scene.renderer, "fogEnd", 0.01f);
        anari::commitParameters(d, scene.renderer);
        scene.render();
        failures +=
            !sameAuxiliary(baselineAux, auxiliary(scene), scene.subtype);
      }
    }
  }
  return failures;
}

int testOpaqueCutPlane(anari::Device d)
{
  Scene scene(d, "quality", "orthographic");
  // Cutting-plane entry at z=4 changes the traversal origin, not the original
  // orthographic camera sample. The opaque surface remains at distance 10.
  anari::setParameter(d, scene.renderer, "cutPlane", Vec4{0.f, 0.f, 1.f, -4.f});
  scene.fog("none", "rayDistance");
  const auto baseline = scene.render();
  const auto baselineAux = auxiliary(scene);
  int failures = 0;
  for (const char *metric : {"viewDepth", "rayDistance"}) {
    for (const char *mode : {"linear", "exp", "exp2"}) {
      scene.fog(mode, metric);
      failures += !expect(scene,
          "opaque cutting-plane origin",
          scene.render(),
          std::vector<double>(PIXELS, fraction(mode, 10.)));
      failures += !sameAuxiliary(baselineAux, auxiliary(scene), scene.subtype);
    }
  }
  scene.fog("none", "rayDistance");
  failures += scene.render() != baseline;
  return failures;
}

int testAperture(anari::Device d, const char *subtype)
{
  Scene scene(d, subtype, "perspective");
  // Collapse the image region to the optical axis and put the plane at the
  // focus distance. Every lens ray then ends at P=(0,0,10), while O is uniform
  // on the radius-8 lens disk. This constrains the reference analytically
  // without reproducing the device's Halton sequence or disk mapping.
  constexpr Vec4 REGION = {0.5f, 0.5f, 0.5f, 0.5f};
  anariSetParameter(
      d, scene.camera, "imageRegion", ANARI_FLOAT32_BOX2, REGION.data());
  anari::setParameter(d, scene.camera, "focusDistance", 10.f);
  anari::setParameter(d, scene.camera, "apertureRadius", 8.f);
  anari::commitParameters(d, scene.camera);
  constexpr int SAMPLES = 4096;
  anari::setParameter(d, scene.renderer, "pixelSamples", SAMPLES);
  scene.fog("none", nullptr);
  const auto baseline = scene.render();
  const auto baselineAux = auxiliary(scene);
  int failures = 0;
  for (const char *metric : {"viewDepth", "rayDistance"}) {
    for (const char *mode : {"linear", "exp", "exp2"}) {
      // For a uniform disk, d has density 2*d/R^2 on [z,sqrt(z^2+R^2)].
      // Integrate each curve in closed form, independently of GPU evaluation.
      constexpr double Z = 10., RADIUS_SQUARED = 64., K = 0.05;
      const double far = std::sqrt(Z * Z + RADIUS_SQUARED);
      double expected = fraction(mode, Z);
      if (std::string(metric) == "rayDistance") {
        if (std::string(mode) == "linear")
          expected =
              2. * (far * far * far - Z * Z * Z) / (3. * RADIUS_SQUARED * 20.);
        else if (std::string(mode) == "exp") {
          const double visibility = 2. / RADIUS_SQUARED
              * ((Z / K + 1. / (K * K)) * std::exp(-K * Z)
                  - (far / K + 1. / (K * K)) * std::exp(-K * far));
          expected = 1. - visibility;
        } else
          expected = 1.
              - (std::exp(-K * K * Z * Z) - std::exp(-K * K * far * far))
                  / (K * K * RADIUS_SQUARED);
      }
      scene.fog(mode, metric);
      const auto image = scene.render();
      if (std::string(metric) == "viewDepth") {
        failures += !expect(scene,
            "lens viewDepth",
            image,
            std::vector<double>(PIXELS, expected));
      } else {
        // 4096 lens samples/pixel, 65536 total. The largest F range here is
        // 0.141 (linear). Six standard errors using the conservative bounded
        // variance range^2/4 give 0.00661/pixel and 0.00166 for the mean.
        // Allow 0.007 and 0.002 respectively, including FP accumulation.
        // These are explicit error criteria, not an IID confidence claim for
        // the deterministic quasi-Monte-Carlo camera sampler. A nominal-origin
        // shortcut gives 0.5 instead of ~0.573 in the linear case and fails.
        double mean = 0.;
        for (const auto &pixel : image) {
          for (int c = 0; c < 3; ++c)
            failures += !std::isfinite(pixel[c])
                || std::abs(pixel[c] - expected) > 0.007;
          failures += !near(pixel[3], 1.);
          mean += pixel[0] / PIXELS;
        }
        printf("%s lens %s: %u samples, mean %.8f reference %.8f\n",
            subtype,
            mode,
            SAMPLES * PIXELS,
            mean,
            expected);
        failures += std::abs(mean - expected) > 0.002;
      }
      failures += !sameAuxiliary(baselineAux, auxiliary(scene), scene.subtype);
    }
  }
  scene.fog("none", "rayDistance");
  failures += scene.render() != baseline;
  failures += !sameAuxiliary(baselineAux, auxiliary(scene), scene.subtype);
  return failures;
}

int testMetricLifecycle(
    anari::Device d, const char *subtype, std::vector<std::string> &warnings)
{
  Scene scene(d, subtype, "perspective");
  int failures = 0;
  for (const char *metric : {"rayDistance",
           "nonsense",
           "viewDepth",
           "rayDistance",
           static_cast<const char *>(nullptr)}) {
    for (int i = 0; i < 4; ++i)
      scene.render();
    warnings.clear();
    scene.fog("linear", metric);
    const auto actual = scene.render();
    const bool invalid = metric && std::string(metric) == "nonsense";
    if (invalid) {
      failures += !expect(scene,
          "invalid metric disables fog",
          actual,
          std::vector<double>(PIXELS, 0.));
      bool warned = false;
      for (const auto &warning : warnings)
        warned |= warning.find("fogDistanceMetric") != std::string::npos;
      if (!warned) {
        fprintf(stderr, "%s missing distanceMetric warning\n", subtype);
        ++failures;
      }
    } else {
      Scene fresh(d, subtype, "perspective");
      fresh.fog("linear", metric);
      const auto reference = fresh.render();
      for (unsigned p = 0; p < PIXELS; ++p)
        for (unsigned c = 0; c < 4; ++c)
          failures += !near(actual[p][c], reference[p][c]);
      if (!metric || std::string(metric) == "viewDepth")
        failures += !expect(scene,
            "default/restored viewDepth",
            actual,
            std::vector<double>(PIXELS, 0.5));
    }
  }
  warnings.clear();
  anari::setParameter(d, scene.renderer, "fogDistanceMetric", 42.f);
  anari::commitParameters(d, scene.renderer);
  failures += !expect(scene,
      "wrong metric type disables fog",
      scene.render(),
      std::vector<double>(PIXELS, 0.));
  bool warned = false;
  for (const auto &warning : warnings)
    warned |= warning.find("fogDistanceMetric") != std::string::npos;
  failures += !warned;
  scene.fog("none", "nonsense");
  warnings.clear();
  failures += !expect(scene,
      "inactive metric ignored",
      scene.render(),
      std::vector<double>(PIXELS, 0.));
  failures += !warnings.empty();
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
    failures += testPlanes(device, subtype);
    failures += testRayBuffer(device, subtype);
    failures += testAperture(device, subtype);
    failures += testMetricLifecycle(device, subtype, warnings);
  }
  failures += testOpaqueCutPlane(device);
  printf("fog camera distance: %d failures\n", failures);
  return failures ? 1 : 0;
}

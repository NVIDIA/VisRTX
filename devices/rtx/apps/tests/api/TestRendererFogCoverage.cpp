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

constexpr UVec2 SIZE = {16, 16};
constexpr unsigned PIXELS = SIZE[0] * SIZE[1];
constexpr Vec3 FOG_COLOR = {0.8f, 0.4f, 0.2f};
constexpr float FOG_END = 20.f;
constexpr float FOG_DENSITY = 0.05f;

/* Only public ANARI objects and mapped outputs; no device test seam or helpers.
 */
struct Scene
{
  Scene(anari::Device device, const char *subtype);
  ~Scene();
  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;
  void addPlane(float z,
      Vec3 color,
      float opacity,
      float left = -100.f,
      float right = 100.f,
      float transmission = 0.f,
      bool mask = false);
  void setColor(unsigned layer, Vec3 color);
  void fog(const char *mode, const char *metric, bool full = false);
  void rayBuffer();
  Image render();
  void setSamples(int count);

  anari::Device device;
  const char *subtype;
  anari::World world;
  anari::Camera camera;
  anari::Renderer renderer;
  anari::Frame frame;
  int samples = 1;
  std::vector<anari::Material> materials;
  std::vector<anari::Instance> instances;
};

Scene::Scene(anari::Device d, const char *s) : device(d), subtype(s)
{
  world = anari::newObject<anari::World>(d);
  ObjectOwner worldOwner(d, world);
  camera = anari::newObject<anari::Camera>(d, "orthographic");
  ObjectOwner cameraOwner(d, camera);
  anari::setParameter(d, camera, "position", Vec3{});
  anari::setParameter(d, camera, "direction", Vec3{0.f, 0.f, 1.f});
  anari::setParameter(d, camera, "up", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, camera, "height", 4.f);
  anari::commitParameters(d, camera);
  renderer = anari::newObject<anari::Renderer>(d, s);
  ObjectOwner rendererOwner(d, renderer);
  anari::setParameter(d, renderer, "denoise", false);
  anari::setParameter(d, renderer, "fireflyFilterMode", "none");
  anari::setParameter(d, renderer, "ambientSamples", 0);
  anari::setParameter(d, renderer, "ambientRadiance", 1.f);
  anari::setParameter(d, renderer, "background", Vec4{});
  anari::setParameter(d, renderer, "fogColorSource", "constant");
  anari::setParameter(d, renderer, "fogColor", FOG_COLOR);
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
  for (auto m : materials)
    anari::release(device, m);
  for (auto i : instances)
    anari::release(device, i);
}

void Scene::addPlane(float z,
    Vec3 color,
    float opacity,
    float left,
    float right,
    float transmission,
    bool mask)
{
  const auto d = device;
  const unsigned index = materials.size();
  auto g = anari::newObject<anari::Geometry>(d, "quad");
  const ObjectOwner geometryOwner(d, g);
  const std::array<Vec3, 4> positions = {Vec3{left, -100.f, z},
      Vec3{right, -100.f, z},
      Vec3{right, 100.f, z},
      Vec3{left, 100.f, z}};
  anari::setParameterArray1D(d, g, "vertex.position", positions.data(), 4);
  anari::commitParameters(d, g);
  const bool emissive = std::string(subtype) == "quality";
  auto m = anari::newObject<anari::Material>(
      d, transmission > 0.f || emissive ? "physicallyBased" : "matte");
  ObjectOwner materialOwner(d, m);
  if (emissive) {
    // Controlled radiance, independent of shadows cast by the front layer.
    anari::setParameter(d, m, "baseColor", Vec3{});
    anari::setParameter(d, m, "metallic", 0.f);
    anari::setParameter(d, m, "specular", 0.f);
    anari::setParameter(d, m, "emissive", color);
  } else
    anari::setParameter(
        d, m, transmission > 0.f ? "baseColor" : "color", color);
  anari::setParameter(d, m, "opacity", opacity);
  if (transmission > 0.f) {
    anari::setParameter(d, m, "transmission", transmission);
    anari::setParameter(d, m, "metallic", 0.f);
    anari::setParameter(d, m, "specular", 0.f);
  }
  if (mask)
    anari::setParameter(d, m, "alphaCutoff", 0.5f);
  anari::commitParameters(d, m);
  materials.push_back(m);
  materialOwner.release(); // Scene owns it only after push_back succeeds.
  auto surface = anari::newObject<anari::Surface>(d);
  const ObjectOwner surfaceOwner(d, surface);
  anari::setParameter(d, surface, "geometry", g);
  anari::setParameter(d, surface, "material", m);
  anari::setParameter(d, surface, "id", 200 + index);
  anari::commitParameters(d, surface);
  auto group = anari::newObject<anari::Group>(d);
  const ObjectOwner groupOwner(d, group);
  anari::setParameterArray1D(d, group, "surface", &surface, 1);
  anari::commitParameters(d, group);
  auto instance = anari::newObject<anari::Instance>(d, "transform");
  ObjectOwner instanceOwner(d, instance);
  anari::setParameter(d, instance, "group", group);
  anari::setParameter(d, instance, "id", 300 + index);
  anari::commitParameters(d, instance);
  instances.push_back(instance);
  instanceOwner.release();
  anari::setParameterArray1D(
      d, world, "instance", instances.data(), instances.size());
  anari::commitParameters(d, world);
}

void Scene::setColor(unsigned layer, Vec3 color)
{
  anari::setParameter(device,
      materials[layer],
      std::string(subtype) == "quality" ? "emissive" : "color",
      color);
  anari::commitParameters(device, materials[layer]);
}

void Scene::fog(const char *mode, const char *metric, bool full)
{
  anari::setParameter(device, renderer, "fogMode", mode);
  anari::setParameter(device, renderer, "fogDistanceMetric", metric);
  anari::setParameter(device, renderer, "fogEnd", full ? 0.01f : FOG_END);
  anari::setParameter(device, renderer, "fogDensity", FOG_DENSITY);
  anari::commitParameters(device, renderer);
}

void Scene::rayBuffer()
{
  auto replacement = anari::newObject<anari::Camera>(device, "rayBuffer");
  const ObjectOwner oldCameraOwner(device, camera);
  camera = replacement; // Scene owns the replacement even if setup throws.
  auto org = anari::newArray2D(device, ANARI_FLOAT32_VEC3, SIZE[0], SIZE[1]);
  const ObjectOwner orgOwner(device, org);
  auto dir = anari::newArray2D(device, ANARI_FLOAT32_VEC3, SIZE[0], SIZE[1]);
  const ObjectOwner dirOwner(device, dir);
  {
    const ArrayMapping<Vec3> origins(device, org);
    const ArrayMapping<Vec3> directions(device, dir);
    for (unsigned p = 0; p < PIXELS; ++p) {
      origins.data()[p] = {0.f, 0.f, float(p % 3)};
      directions.data()[p] = {3.f, 0.f, 4.f};
    }
  }
  anari::setParameter(device, camera, "ray.org", org);
  anari::setParameter(device, camera, "ray.dir", dir);
  anari::setParameter(device, camera, "position", Vec3{0.f, 0.f, -1.f});
  anari::setParameter(device, camera, "direction", Vec3{0.f, 0.f, 1.f});
  anari::commitParameters(device, camera);
  anari::setParameter(device, frame, "camera", camera);
  anari::commitParameters(device, frame);
}

void Scene::setSamples(int count)
{
  anari::setParameter(device, renderer, "pixelSamples", count);
  samples = count;
}

Image Scene::render()
{
  anari::render(device, frame);

  anari::wait(device, frame);
  return readImage(device, frame, SIZE);
}

double visibility(const char *mode, double distance)
{
  return referenceVisibility(mode, distance, 0., FOG_END, FOG_DENSITY);
}

bool expect(const Scene &scene,
    const std::string &label,
    const Image &actual,
    const std::vector<Vec4> &expected)
{
  double maxError = 0., maxToleranceFraction = 0.;
  for (unsigned p = 0; p < PIXELS; ++p) {
    for (unsigned c = 0; c < 4; ++c) {
      const double ref = expected[p][c];
      const double error = std::abs(actual.color[p][c] - ref);
      maxError = std::fmax(maxError, error);
      maxToleranceFraction = std::fmax(
          maxToleranceFraction, error / (1e-5 + 1e-4 * std::abs(ref)));
      if (!std::isfinite(actual.color[p][c])
          || std::abs(actual.color[p][c] - ref) > 1e-5 + 1e-4 * std::abs(ref)) {
        fprintf(stderr,
            "%s %s pixel %u channel %u: got %.9g expected %.9g\n",
            scene.subtype,
            label.c_str(),
            p,
            c,
            actual.color[p][c],
            ref);
        return false;
      }
    }
  }
  printf("%s %s: %d spp, max error=%g, tolerance fraction=%g\n",
      scene.subtype,
      label.c_str(),
      scene.samples,
      maxError,
      maxToleranceFraction);
  return true;
}

// Quality analytically deposits local opacity but samples coverage for the
// continuation. Compare with the independent composited expectation, not with
// another renderer or a particular realization of the coverage RNG. At 4096
// spp, bounded [0,1] observations have sigma <= 0.5/sqrt(4096). The per-pixel
// limit .05 exceeds six standard errors; the .008 mean limit does likewise in
// each of six disjoint groups (>= 40 pixels). These are conservative error
// criteria, not an IID confidence claim about the renderer's random sequence.
bool expectCoverage(const Scene &scene,
    const std::string &label,
    const Image &actual,
    const std::vector<Vec4> &expected)
{
  if (std::string(scene.subtype) != "quality")
    return expect(scene, label, actual, expected);
  double errors[6][4] = {};
  unsigned counts[6] = {};
  double maxPixelError = 0., maxMeanError = 0.;
  for (unsigned p = 0; p < PIXELS; ++p) {
    const unsigned group = p % 3 + 3 * (p % SIZE[0] >= SIZE[0] / 2);
    ++counts[group];
    for (unsigned c = 0; c < 4; ++c) {
      const double error = actual.color[p][c] - expected[p][c];
      if (!std::isfinite(error))
        return false;
      errors[group][c] += error;
      maxPixelError = std::fmax(maxPixelError, std::abs(error));
    }
  }
  for (unsigned g = 0; g < 6; ++g)
    for (unsigned c = 0; c < 4; ++c)
      maxMeanError =
          std::fmax(maxMeanError, std::abs(errors[g][c] / counts[g]));
  printf("quality %s: 4096 spp, max pixel error=%g, grouped mean error=%g\n",
      label.c_str(),
      maxPixelError,
      maxMeanError);
  return maxPixelError <= 0.05 && maxMeanError <= 0.008;
}

struct FogCase
{
  const char *name;
  const char *mode;
  const char *metric;
  bool full = false;
};

// Oblique rays distinguish the metrics. Keep every active curve/metric pair,
// but only one saturated endpoint and one disable-after-active transition.
constexpr FogCase LAYER_FOG[] = {{"linear-view", "linear", "viewDepth"},
    {"exp-view", "exp", "viewDepth"},
    {"exp2-view", "exp2", "viewDepth"},
    {"linear-ray", "linear", "rayDistance"},
    {"exp-ray", "exp", "rayDistance"},
    {"exp2-ray", "exp2", "rayDistance"},
    {"saturated", "linear", "viewDepth", true},
    {"disabled-restoration", "none", "rayDistance"}};

// Orthographic planar depths agree under both metrics. Retain a ray-metric
// composition witness, all curves and the endpoint; cheap Distance anchors
// separately distinguish the metrics rather than repeating equal distances.
constexpr FogCase PLANAR_FOG[] = {{"linear-view", "linear", "viewDepth"},
    {"exp-view", "exp", "viewDepth"},
    {"exp2-view", "exp2", "viewDepth"},
    {"linear-ray", "linear", "rayDistance"},
    {"saturated", "linear", "viewDepth", true}};

int testLayers(anari::Device d, const char *subtype, bool background = false)
{
  int failures = 0;
  constexpr Vec3 FRONT{0.2f, 0.3f, 0.1f}, BACK{0.1f, 0.2f, 0.4f};
  const Vec3 target = background ? Vec3{0.15f, 0.65f, 0.35f} : FOG_COLOR;
  const struct LayerCase
  {
    const char *name;
    float front, rear;
    bool occlusionOnly = false;
  } cases[] = {{"empty", 0.f, 0.f},
      {"rear-partial", 0.f, 0.5f},
      {"rear-opaque", 0.f, 1.f},
      {"front-partial", 0.25f, 0.f},
      {"both-partial", 0.25f, 0.5f},
      {"partial-over-opaque", 0.25f, 1.f},
      {"opaque-over-empty", 1.f, 0.f, true},
      {"opaque-over-partial", 1.f, 0.5f, true},
      {"opaque-over-opaque", 1.f, 1.f}};
  for (const auto &layer : cases) {
    const float alpha = layer.front, backAlpha = layer.rear;
    logCase(std::string(subtype) + "/layers/" + layer.name
        + (background ? "/background" : "/constant"));
    Scene scene(d, subtype);
    if (background) {
      anari::setParameter(d,
          scene.renderer,
          "background",
          Vec4{target[0], target[1], target[2], 1.f});
      anari::setParameter(d, scene.renderer, "fogColorSource", "background");
    }
    if (std::string(subtype) == "quality")
      scene.setSamples(4096);
    scene.addPlane(4.f, FRONT, alpha);
    scene.addPlane(12.f, BACK, backAlpha);
    scene.rayBuffer();
    scene.fog("none", "viewDepth");
    const auto baseline = scene.render();
    for (const auto &fog : LAYER_FOG) {
      // Rear opacity cannot affect an opaque front. Keep live unfogged,
      // oblique partial-fog and disable witnesses at both other opacities;
      // opaque-over-opaque carries the complete setting table.
      if (layer.occlusionOnly && std::string(fog.name) != "linear-ray"
          && std::string(fog.mode) != "none")
        continue;
      fprintf(stderr, "[fog-setting] %s/%s\n", layer.name, fog.name);
      scene.fog(fog.mode, fog.metric, fog.full);
      std::vector<Vec4> expected(PIXELS);
      for (unsigned p = 0; p < PIXELS; ++p) {
        const bool radial = std::string(fog.metric) == "rayDistance";
        const double t0 = fog.full
            ? 0.
            : visibility(fog.mode, radial ? (4. - p % 3) * 1.25 : 5.);
        const double t1 = fog.full
            ? 0.
            : visibility(fog.mode, radial ? (12. - p % 3) * 1.25 : 13.);
        for (unsigned c = 0; c < 3; ++c) {
          // Independently composited premultiplied contributions at two
          // depths.
          expected[p][c] = alpha * (t0 * FRONT[c] + (1. - t0) * target[c])
              + (1. - alpha) * backAlpha
                  * (t1 * BACK[c] + (1. - t1) * target[c])
              + (background ? (1. - alpha) * (1. - backAlpha) * target[c] : 0.);
        }
        expected[p][3] = background ? 1.f : alpha + (1.f - alpha) * backAlpha;
      }
      const auto image = scene.render();
      failures += !expectCoverage(scene,
          std::string(layer.name) + (background ? " background " : " ")
              + fog.name,
          image,
          expected);
      failures += !sameAuxiliary(baseline, image, scene.subtype);
    }
    // Make ID assertions non-vacuous, including the transparent front case.
    if (baseline.primitive[0] == ~0u || baseline.object[0] != 200
        || baseline.instance[0] != 300) {
      fprintf(stderr,
          "%s unexpected baseline IDs: %u %u %u\n",
          subtype,
          baseline.primitive[0],
          baseline.object[0],
          baseline.instance[0]);
      ++failures;
    }
  }
  return failures;
}

int testTransmission(anari::Device d, const char *subtype)
{
  int failures = 0;
  constexpr Vec3 TINT{0.25f, 0.5f, 0.75f}, BACK{0.1f, 0.2f, 0.3f};
  for (float alpha : {0.5f, 1.f}) {
    logCase(
        std::string(subtype) + "/transmission/alpha=" + std::to_string(alpha));
    Scene scene(d, subtype);
    scene.addPlane(4.f, TINT, alpha, -100.f, 100.f, 0.75f);
    scene.addPlane(12.f, BACK, 0.5f);
    // Colored transmission without illumination: all deposited unfogged
    // radiance is zero, including Interactive's stochastic reflection bounce.
    anari::setParameter(d, scene.renderer, "ambientRadiance", 0.f);
    anari::setParameter(d, scene.renderer, "fixedAmbientLighting", false);
    scene.rayBuffer();
    scene.fog("none", "viewDepth");
    const auto baseline = scene.render();
    for (const auto &fog : LAYER_FOG) {
      const auto mode = fog.mode, metric = fog.metric;
      const bool full = fog.full;
      fprintf(stderr, "[fog-setting] transmission/%s\n", fog.name);
      scene.fog(mode, metric, full);
      std::vector<Vec4> expected(PIXELS);
      for (unsigned p = 0; p < PIXELS; ++p) {
        const bool radial = std::string(metric) == "rayDistance";
        const double t0 =
            full ? 0. : visibility(mode, radial ? (4. - p % 3) * 1.25 : 5.);
        const double t1 =
            full ? 0. : visibility(mode, radial ? (12. - p % 3) * 1.25 : 13.);
        for (unsigned c = 0; c < 3; ++c) {
          // Thin native PBR: no metal/volume attenuation, so the material's
          // straight-through filter is transmission * baseColor. This is
          // independent of fog visibility and of the scalar coverage alpha.
          const double through = 1. - alpha + alpha * 0.75 * TINT[c];
          expected[p][c] =
              (alpha * (1. - t0) + through * 0.5 * (1. - t1)) * FOG_COLOR[c];
        }
        expected[p][3] = alpha + (1.f - alpha) * 0.5f;
      }
      const auto image = scene.render();
      failures += !expect(scene,
          std::string("colored transmission ") + fog.name,
          image,
          expected);
      failures += !sameAuxiliary(baseline, image, scene.subtype);
    }
  }
  return failures;
}

int testCutouts(anari::Device d, const char *subtype)
{
  logCase(std::string(subtype) + "/cutouts");
  Scene scene(d, subtype);
  if (std::string(subtype) == "quality")
    scene.setSamples(4096);
  constexpr Vec3 FRONT{0.2f, 0.3f, 0.1f}, BACK{0.1f, 0.2f, 0.4f};
  // Two adjoining masked patches form a genuine discarded hole in front of a
  // half-covered rear layer, not a geometric hole or a zero-valued fog color.
  scene.addPlane(4.f, FRONT, 0.25f, -100.f, 0.f, 0.f, true);
  scene.addPlane(4.f, FRONT, 0.75f, 0.f, 100.f, 0.f, true);
  scene.addPlane(12.f, BACK, 0.5f);
  scene.fog("none", "viewDepth");
  const auto baseline = scene.render();
  int failures = 0;
  for (const auto &fog : PLANAR_FOG) {
    const auto mode = fog.mode;
    const bool full = fog.full;
    fprintf(stderr, "[fog-setting] cutouts/%s\n", fog.name);
    scene.fog(mode, fog.metric, full);
    std::vector<Vec4> expected(PIXELS);
    for (unsigned p = 0; p < PIXELS; ++p) {
      // Camera right is -X for direction +Z and up +Y.
      const bool covered = p % SIZE[0] < SIZE[0] / 2;
      const double t = full ? 0. : visibility(mode, covered ? 4. : 12.);
      const auto &color = covered ? FRONT : BACK;
      const float alpha = covered ? 1.f : 0.5f;
      for (unsigned c = 0; c < 3; ++c)
        expected[p][c] = alpha * (t * color[c] + (1. - t) * FOG_COLOR[c]);
      expected[p][3] = alpha;
    }
    const auto image = scene.render();
    failures += !expectCoverage(
        scene, std::string("masked hole ") + fog.name, image, expected);
    failures += !sameAuxiliary(baseline, image, scene.subtype);
  }
  scene.fog("none", "rayDistance");
  const auto restored = scene.render();
  failures +=
      !expect(scene, "cutout disabled restoration", restored, baseline.color);
  failures += !sameAuxiliary(baseline, restored, scene.subtype);
  return failures;
}

Vec3 cueColor(Vec3 color, double transmittance)
{
  for (unsigned c = 0; c < 3; ++c)
    color[c] = transmittance * color[c] + (1. - transmittance) * FOG_COLOR[c];
  return color;
}

int testEdges(anari::Device d, const char *subtype)
{
  int failures = 0;
  constexpr Vec3 FRONT{0.1f, 0.2f, 0.3f}, BACK{0.4f, 0.1f, 0.2f};
  for (bool depthEdge : {false, true}) {
    logCase(std::string(subtype) + (depthEdge ? "/depth-edge" : "/silhouette"));
    Scene scene(d, subtype);
    const bool layeredSilhouette =
        !depthEdge && std::string(subtype) == "quality";
    scene.addPlane(4.f, FRONT, 1.f, -100.f, layeredSilhouette ? 0.02f : 0.07f);
    if (depthEdge)
      scene.addPlane(12.f, BACK, 1.f);
    else if (layeredSilhouette)
      // Quality stores nearest-hit depth: a single constant-depth silhouette
      // happens to commute with premultiplied fog. Two adjoining depths plus
      // empty background make a resolved-depth operation distinguishable.
      scene.addPlane(12.f, BACK, 1.f, 0.02f, 0.07f);
    // 64 camera samples per pixel, paired by fresh accumulation on Commit.
    // No statistical tolerance is needed: the reference uses the same geometry
    // and sampling with independently pre-cued material colors. Orthographic
    // planar layers have constant depth under BOTH metrics for every sample.
    scene.setSamples(64);
    scene.fog("none", "viewDepth");
    const auto baseline = scene.render();
    for (const auto &fog : PLANAR_FOG) {
      const auto mode = fog.mode, metric = fog.metric;
      const bool full = fog.full;
      fprintf(stderr, "[fog-setting] edges/%s\n", fog.name);
      scene.fog(mode, metric, full);
      const auto image = scene.render();
      failures += !sameAuxiliary(baseline, image, scene.subtype);
      scene.setColor(0, cueColor(FRONT, full ? 0. : visibility(mode, 4.)));
      if (depthEdge || layeredSilhouette)
        scene.setColor(1, cueColor(BACK, full ? 0. : visibility(mode, 12.)));
      scene.fog("none", metric);
      const auto reference = scene.render();
      failures += !expect(scene,
          std::string(depthEdge ? "depth edge " : "silhouette ") + mode + " "
              + metric,
          image,
          reference.color);

      unsigned mixed = 0, distinguishable = 0;
      double maxPostError = 0.;
      for (unsigned p = 0; p < PIXELS; ++p) {
        const double coverage = depthEdge
            ? (baseline.color[p][0] - BACK[0]) / (FRONT[0] - BACK[0])
            : baseline.color[p][3];
        if (coverage <= 0.05 || coverage >= 0.95)
          continue;
        ++mixed;
        // A tempting but wrong resolved-depth implementation, evaluated on
        // unfogged output. Assert that the oracle can reject it, not just
        // that two equivalent public-ANARI renders happen to agree.
        const double t = full ? 0. : visibility(mode, baseline.depth[p]);
        double error = 0.;
        for (unsigned c = 0; c < 3; ++c) {
          const double post = t * baseline.color[p][c]
              + baseline.color[p][3] * (1. - t) * FOG_COLOR[c];
          error = std::fmax(error, std::abs(post - reference.color[p][c]));
        }
        maxPostError = std::fmax(maxPostError, error);
        distinguishable += error > 0.01;
      }
      if (mixed < SIZE[1] / 2 || (!full && distinguishable < SIZE[1] / 2)) {
        fprintf(stderr,
            "%s insensitive %s %s edge oracle: mixed=%u distinct=%u\n",
            subtype,
            depthEdge ? "depth" : "silhouette",
            mode,
            mixed,
            distinguishable);
        ++failures;
      }
      printf(
          "%s %s %s %s: 64 spp, mixed=%u distinct=%u, max resolved-depth error=%g\n",
          subtype,
          depthEdge ? "depth edge" : "silhouette",
          mode,
          metric,
          mixed,
          distinguishable,
          maxPostError);
      scene.setColor(0, FRONT);
      if (depthEdge || layeredSilhouette)
        scene.setColor(1, BACK);
    }
    scene.fog("none", "viewDepth");
    const auto restored = scene.render();
    failures +=
        !expect(scene, "edge disabled restoration", restored, baseline.color);
    failures += !sameAuxiliary(baseline, restored, scene.subtype);
  }
  return failures;
}

int testLighting(anari::Device d, const char *subtype)
{
  logCase(std::string(subtype) + "/lighting-AO");
  Scene scene(d, subtype);
  constexpr Vec3 GRAY{0.4f, 0.4f, 0.4f};
  // Pixel-aligned opaque planes: every sample in a pixel sees the same depth.
  // The front half-plane both shadows and occludes ambient light at the rear.
  scene.addPlane(4.f, GRAY, 1.f, -100.f, 0.f);
  scene.addPlane(12.f, GRAY, 1.f);
  if (std::string(subtype) == "quality") {
    for (auto material : scene.materials) {
      anari::setParameter(d, material, "baseColor", GRAY);
      anari::setParameter(d, material, "emissive", Vec3{});
      anari::commitParameters(d, material);
    }
  }
  auto light = anari::newObject<anari::Light>(d, "directional");
  const ObjectOwner lightOwner(d, light);
  anari::setParameter(d, light, "direction", Vec3{1.f, 0.f, 1.f});
  anari::setParameter(d, light, "irradiance", 2.f);
  anari::commitParameters(d, light);
  anari::setParameterArray1D(d, scene.world, "light", &light, 1);
  anari::commitParameters(d, scene.world);
  anari::setParameter(d, scene.renderer, "fixedAmbientLighting", false);
  anari::setParameter(d, scene.renderer, "ambientRadiance", 0.3f);
  anari::setParameter(d, scene.renderer, "ambientSamples", 16);
  const auto setAoDistance = [&](float distance) {
    anari::setParameter(
        d, scene.renderer, "ambientOcclusionDistance", distance);
  };
  setAoDistance(20.f);
  scene.setSamples(64);
  int failures = 0;
  // Quality path-traces ambient illumination; only straight-through renderers
  // expose the bounded AO estimator. Remove direct lighting to isolate it.
  if (std::string(subtype) != "quality") {
    anari::setParameter(d, light, "irradiance", 0.f);
    anari::commitParameters(d, light);
    scene.fog("none", "viewDepth");
    const auto occluded = scene.render();
    setAoDistance(0.25f); // Too short to reach the plane eight units away.
    scene.fog("none", "viewDepth");
    const auto unoccluded = scene.render();
    double aoDifference = 0., unoccludedMean = 0.;
    for (unsigned p = 0; p < PIXELS; ++p) {
      // The rear receiver, not the foreground occluder, is the AO control.
      if (p % SIZE[0] < SIZE[0] / 2) {
        unoccludedMean += unoccluded.color[p][0] / (PIXELS / 2);
        aoDifference +=
            (unoccluded.color[p][0] - occluded.color[p][0]) / (PIXELS / 2);
      }
    }
    if (!(std::abs(unoccludedMean - 0.12) <= 1e-5 + 1e-4 * 0.12)) {
      fprintf(stderr,
          "%s short-radius AO receiver: got %g expected 0.12\n",
          subtype,
          unoccludedMean);
      ++failures;
    }
    if (!(aoDifference > 0.02)) {
      fprintf(stderr,
          "%s AO-only radius control lacks occlusion: difference=%g\n",
          subtype,
          aoDifference);
      ++failures;
    }
    printf("%s AO-only radius control: 64 spp x 16 AO samples, difference=%g\n",
        subtype,
        aoDifference);
    setAoDistance(20.f);
    scene.fog("linear", "viewDepth");
    auto expected = occluded.color;
    for (unsigned p = 0; p < PIXELS; ++p) {
      const double z = p % SIZE[0] >= SIZE[0] / 2 ? 4. : 12.;
      const double t = visibility("linear", z);
      for (unsigned c = 0; c < 3; ++c)
        expected[p][c] = t * expected[p][c] + (1. - t) * FOG_COLOR[c];
    }
    const auto fogged = scene.render();
    failures += !expect(scene, "paired AO-only fog", fogged, expected);
    failures += !sameAuxiliary(occluded, fogged, scene.subtype);
    anari::setParameter(d, light, "irradiance", 2.f);
    anari::commitParameters(d, light);
  }
  scene.fog("none", "viewDepth");
  const auto baseline = scene.render();
  double frontMean = 0., backMean = 0.;
  for (unsigned p = 0; p < PIXELS; ++p) {
    if (p % SIZE[0] >= SIZE[0] / 2)
      frontMean += baseline.color[p][0] / (PIXELS / 2);
    else
      backMean += baseline.color[p][0] / (PIXELS / 2);
  }
  if (frontMean - backMean < 0.01) {
    fprintf(stderr,
        "%s lighting oracle lacks a shadow/AO signal: front=%g back=%g\n",
        subtype,
        frontMean,
        backMean);
    ++failures;
  }
  for (const auto &fog : PLANAR_FOG) {
    const auto mode = fog.mode, metric = fog.metric;
    const bool full = fog.full;
    fprintf(stderr, "[fog-setting] lighting/%s\n", fog.name);
    scene.fog(mode, metric, full);
    auto expected = baseline.color;
    for (unsigned p = 0; p < PIXELS; ++p) {
      const double z = p % SIZE[0] >= SIZE[0] / 2 ? 4. : 12.;
      const double t = full ? 0. : visibility(mode, z);
      for (unsigned c = 0; c < 3; ++c)
        expected[p][c] = t * expected[p][c] + (1. - t) * FOG_COLOR[c];
    }
    const auto image = scene.render();
    failures += !expect(scene,
        std::string("paired lighting/AO ") + mode + " " + metric,
        image,
        expected);
    failures += !sameAuxiliary(baseline, image, scene.subtype);
  }
  printf("%s paired lighting/AO: 64 spp x 16 AO samples, front=%g back=%g\n",
      subtype,
      frontMean,
      backMean);
  return failures;
}

int testDisabledChannels(anari::Device d, const char *subtype)
{
  logCase(std::string(subtype) + "/disabled-channels");
  Scene scene(d, subtype);
  scene.addPlane(4.f, {0.2f, 0.3f, 0.1f}, 0.25f, -100.f, 0.f);
  scene.addPlane(12.f, {0.1f, 0.2f, 0.4f}, 0.5f);
  // Paired 64-spp realizations test exact preservation, not convergence.
  scene.setSamples(64);
  anari::commitParameters(d, scene.renderer);
  const auto omitted = scene.render();
  int failures = 0;
  for (bool inactiveInvalid : {false, true}) {
    if (inactiveInvalid) {
      anari::setParameter(d, scene.renderer, "fogDistanceMetric", "invalid");
      anari::setParameter(d, scene.renderer, "fogColorSource", 42);
      anari::setParameter(
          d, scene.renderer, "fogColor", Vec3{-1.f, -2.f, -3.f});
      anari::setParameter(d, scene.renderer, "fogStart", -1.f);
      anari::setParameter(d, scene.renderer, "fogEnd", -2.f);
      anari::setParameter(
          d, scene.renderer, "fogDensity", "wrong inactive type");
    }
    for (bool unset : {false, true}) {
      if (unset)
        anari::unsetParameter(d, scene.renderer, "fogMode");
      else
        anari::setParameter(d, scene.renderer, "fogMode", "none");
      anari::commitParameters(d, scene.renderer);
      const auto image = scene.render();
      failures += image.color != omitted.color;
      failures += !sameAuxiliary(omitted, image, scene.subtype);
    }
  }
  printf(
      "%s omitted/none/inactive-invalid channels: %d failures, paired 64 spp\n",
      subtype,
      failures);
  return failures;
}

void configureLayers(
    Scene &scene, const char *mode, const char *metric, bool background)
{
  const auto d = scene.device;
  // This is a paired fresh/reused oracle, not the independent coverage mean.
  scene.setSamples(std::string(scene.subtype) == "quality" ? 64 : 1);
  anari::setParameter(d,
      scene.renderer,
      "background",
      background ? Vec4{0.15f, 0.65f, 0.35f, 1.f} : Vec4{});
  anari::setParameter(d,
      scene.renderer,
      "fogColorSource",
      background ? "background" : "constant");
  scene.fog(mode, metric);
}

int testLayeredRecommit(anari::Device d, const char *subtype)
{
  logCase(std::string(subtype) + "/layered-recommit");
  Scene reused(d, subtype);
  reused.addPlane(4.f, {0.2f, 0.3f, 0.1f}, 0.25f);
  reused.addPlane(12.f, {0.1f, 0.2f, 0.4f}, 0.5f);
  reused.rayBuffer();
  configureLayers(reused, "linear", "viewDepth", false);
  int failures = 0;
  for (const char *mode : {"exp", "exp2", "none", "linear"}) {
    // Deliberately leave old accumulated color, not merely a committed setting.
    for (unsigned i = 0; i < 4; ++i)
      reused.render();
    int oldSamples = -1;
    const bool haveOldSamples = anari::getProperty(
        d, reused.frame, "numSamples", oldSamples, ANARI_WAIT);
    const bool background = std::string(mode) == "exp";
    const char *metric = background ? "rayDistance" : "viewDepth";
    configureLayers(reused, mode, metric, background);
    const auto actual = reused.render();
    Scene fresh(d, subtype);
    fresh.addPlane(4.f, {0.2f, 0.3f, 0.1f}, 0.25f);
    fresh.addPlane(12.f, {0.1f, 0.2f, 0.4f}, 0.5f);
    fresh.rayBuffer();
    configureLayers(fresh, mode, metric, background);
    const auto reference = fresh.render();
    int actualSamples = -1, freshSamples = -1;
    const bool haveActualSamples = anari::getProperty(
        d, reused.frame, "numSamples", actualSamples, ANARI_WAIT);
    const bool haveFreshSamples = anari::getProperty(
        d, fresh.frame, "numSamples", freshSamples, ANARI_WAIT);
    // The default sampleLimit can cap the old renders. Require actual retained
    // accumulation beyond one render, not an assumed four-times-spp count.
    if (!haveOldSamples || !haveActualSamples || !haveFreshSamples
        || oldSamples <= reused.samples || actualSamples != reused.samples
        || freshSamples != reused.samples) {
      fprintf(stderr,
          "%s %s incorrect accumulation counts: old=%d changed=%d fresh=%d\n",
          subtype,
          mode,
          oldSamples,
          actualSamples,
          freshSamples);
      ++failures;
    }
    printf("%s %s accumulation counts: old=%d changed=%d fresh=%d\n",
        subtype,
        mode,
        oldSamples,
        actualSamples,
        freshSamples);
    // Fresh/reused realizations use identical accumulation-frame-zero samples;
    // unlike the analytic coverage mean above, this comparison is paired.
    failures += !expect(reused,
        std::string("layered recommit ") + mode,
        actual,
        reference.color);
    failures += !sameAuxiliary(actual, reference, reused.subtype);
  }
  printf(
      "%s layered recommits: fresh-renderer paired checks, %d spp, "
      "four old render calls per transition (subject to sampleLimit)\n",
      subtype,
      reused.samples);
  return failures;
}

} // namespace

int main()
{
  auto device = makeVisRTXDevice(exitOnErrorStatus);
  const ObjectOwner deviceOwner(device, device);
  int failures = requireRendererFogSupport(device);
  for (const char *subtype : {"fast", "interactive", "default", "quality"})
    failures += testDisabledChannels(device, subtype);
  for (const char *subtype : {"fast", "interactive", "default"}) {
    failures += testLayers(device, subtype);
    failures += testLayers(device, subtype, true);
    failures += testLayeredRecommit(device, subtype);
    failures += testTransmission(device, subtype);
    failures += testCutouts(device, subtype);
    failures += testEdges(device, subtype);
    failures += testLighting(device, subtype);
  }
  failures += testLayers(device, "quality");
  failures += testLayers(device, "quality", true);
  failures += testCutouts(device, "quality");
  failures += testEdges(device, "quality");
  failures += testLighting(device, "quality");
  failures += testLayeredRecommit(device, "quality");
  if (failures)
    fprintf(stderr, "%d fog coverage failure(s)\n", failures);
  else
    printf("fog coverage passed (fast, interactive, default, quality)\n");
  return failures ? 1 : 0;
}

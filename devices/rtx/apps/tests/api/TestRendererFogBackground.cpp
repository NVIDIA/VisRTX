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
#include <new>
#include <string>
#include <vector>

namespace {

using namespace visrtx::fogtest;

using ColorImage = std::vector<Vec4>;
constexpr unsigned WIDTH = 8;
constexpr unsigned HEIGHT = 4;

/* Public ANARI fixture. A black plane at viewDepth=10 removes all lighting
 * variance; no denoising/nonlinear filtering, one sample per accumulation
 * frame. All checked pixels are well inside the plane, even with an image
 * region.
 */
struct Scene
{
  Scene(anari::Device d, const char *subtype);
  ~Scene();
  Scene(const Scene &) = delete;
  Scene &operator=(const Scene &) = delete;
  anari::Renderer newRenderer() const;
  ColorImage render(anari::Frame target = nullptr);
  void opacity(float value);
  void showSurface(bool visible);
  bool expect(const char *label, const ColorImage &expected);
  bool expect(const char *label, Vec4 expected);

  anari::Device d;
  const char *subtype;
  anari::World world;
  anari::Camera camera;
  anari::Material material;
  anari::Surface surface;
  anari::Renderer renderer;
  anari::Frame frame;
};

Scene::Scene(anari::Device device, const char *s) : d(device), subtype(s)
{
  material = anari::newObject<anari::Material>(d, "matte");
  ObjectOwner materialOwner(d, material);
  anari::setParameter(d, material, "color", Vec3{0.f, 0.f, 0.f});
  anari::commitParameters(d, material);
  constexpr std::array<Vec3, 4> POSITIONS = {Vec3{-100.f, -100.f, 0.f},
      Vec3{100.f, -100.f, 0.f},
      Vec3{100.f, 100.f, 0.f},
      Vec3{-100.f, 100.f, 0.f}};
  auto geometry = anari::newObject<anari::Geometry>(d, "quad");
  const ObjectOwner geometryOwner(d, geometry);
  anari::setParameterArray1D(
      d, geometry, "vertex.position", POSITIONS.data(), POSITIONS.size());
  anari::commitParameters(d, geometry);
  surface = anari::newObject<anari::Surface>(d);
  ObjectOwner surfaceOwner(d, surface);
  anari::setParameter(d, surface, "geometry", geometry);
  anari::setParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);
  world = anari::newObject<anari::World>(d);
  ObjectOwner worldOwner(d, world);
  anari::setParameterArray1D(d, world, "surface", &surface, 1);
  anari::commitParameters(d, world);
  camera = anari::newObject<anari::Camera>(d, "orthographic");
  ObjectOwner cameraOwner(d, camera);
  anari::setParameter(d, camera, "position", Vec3{0.f, 0.f, -10.f});
  anari::setParameter(d, camera, "direction", Vec3{0.f, 0.f, 1.f});
  anari::setParameter(d, camera, "up", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, camera, "height", 1.f);
  anari::commitParameters(d, camera);
  renderer = newRenderer();
  ObjectOwner rendererOwner(d, renderer);
  frame = anari::newObject<anari::Frame>(d);
  ObjectOwner frameOwner(d, frame);
  anari::setParameter(d, frame, "size", UVec2{WIDTH, HEIGHT});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", camera);
  anari::setParameter(d, frame, "renderer", renderer);
  anari::commitParameters(d, frame);
  materialOwner.release();
  surfaceOwner.release();
  worldOwner.release();
  cameraOwner.release();
  rendererOwner.release();
  frameOwner.release();
}

Scene::~Scene()
{
  anari::release(d, frame);
  anari::release(d, renderer);
  anari::release(d, camera);
  anari::release(d, world);
  anari::release(d, material);
  anari::release(d, surface);
}

anari::Renderer Scene::newRenderer() const
{
  auto r = anari::newObject<anari::Renderer>(d, subtype);
  ObjectOwner rendererOwner(d, r);
  anari::setParameter(d, r, "denoise", false);
  anari::setParameter(d, r, "fireflyFilterMode", "none");
  anari::setParameter(d, r, "ambientSamples", 0);
  anari::setParameter(d, r, "ambientRadiance", 0.f);
  anari::setParameter(d, r, "fogMode", "linear");
  anari::setParameter(d, r, "fogEnd", 20.f);
  anari::setParameter(d, r, "fogColorSource", "background");
  anari::commitParameters(d, r);
  rendererOwner.release();
  return r;
}

ColorImage Scene::render(anari::Frame target)
{
  if (!target)
    target = frame;
  anari::render(d, target);
  anari::wait(d, target);
  return readChannel<Vec4>(d, target, "channel.color", WIDTH, HEIGHT);
}

void Scene::opacity(float value)
{
  anari::setParameter(d, material, "opacity", value);
  anari::commitParameters(d, material);
}

void Scene::showSurface(bool visible)
{
  if (visible)
    anari::setParameterArray1D(d, world, "surface", &surface, 1);
  else
    anari::unsetParameter(d, world, "surface");
  anari::commitParameters(d, world);
}

bool Scene::expect(const char *label, const ColorImage &expected)
{
  const auto actual = render();
  // Spec's linear-float tolerance, including alpha. No statistical allowance:
  // these full-coverage, black-surface scenes have deterministic radiance.
  for (unsigned p = 0; p < actual.size(); ++p) {
    for (int c = 0; c < 4; ++c) {
      if (!std::isfinite(actual[p][c])
          || std::abs(double(actual[p][c]) - expected[p][c])
              > 1e-5 + 1e-4 * std::abs(double(expected[p][c]))) {
        fprintf(stderr,
            "%s %s pixel %u channel %d: got %.9g, expected %.9g\n",
            subtype,
            label,
            p,
            c,
            actual[p][c],
            expected[p][c]);
        return false;
      }
    }
  }
  return true;
}

bool Scene::expect(const char *label, Vec4 expected)
{
  return expect(label, ColorImage(WIDTH * HEIGHT, expected));
}

int testFlat(Scene &s)
{
  const auto d = s.d;
  const auto r = s.renderer;
  int failures = 0;
  // Independently worked fractions at viewDepth=10: linear [0,20] -> 1/2,
  // exp density=.1 -> 1-e^-1, exp2 density=.05 -> 1-e^(-1/4).
  constexpr const char *MODES[] = {"linear", "exp", "exp2"};
  constexpr double FRACTIONS[] = {0.5, 0.6321205588285577, 0.2211992169285951};
  for (int curve = 0; curve < 3; ++curve) {
    anari::setParameter(d, r, "fogMode", MODES[curve]);
    anari::setParameter(d, r, "fogDensity", curve == 2 ? 0.05f : 0.1f);
    for (float alpha : {0.f, 0.25f, 1.f}) {
      const Vec4 bg{4.f, 2.f, 0.5f, alpha};
      anari::setParameter(d, r, "background", bg);
      for (bool premultiply : {false, true}) {
        anari::setParameter(d, r, "premultiplyBackground", premultiply);
        anari::commitParameters(d, r);
        Vec4 expected{0.f, 0.f, 0.f, 1.f};
        Vec4 miss = bg;
        for (int c = 0; c < 3; ++c) {
          miss[c] *= premultiply ? alpha : 1.f;
          expected[c] = float(FRACTIONS[curve] * miss[c]);
        }
        failures +=
            !s.expect("HDR flat background curve/premultiplication", expected);
        s.showSurface(false);
        failures += !s.expect("unfogged miss RGB and alpha", miss);
        const auto fogMiss = s.render();
        anari::setParameter(d, r, "fogMode", "none");
        anari::commitParameters(d, r);
        failures += !s.expect("disabled miss RGB and alpha", miss);
        failures += s.render() != fogMiss;
        anari::setParameter(d, r, "fogMode", MODES[curve]);
        anari::commitParameters(d, r);
        s.showSurface(true);
      }
    }
  }
  anari::setParameter(d, r, "fogMode", "linear");
  anari::setParameter(d, r, "premultiplyBackground", false);
  anari::setParameter(d, r, "background", Vec4{0.2f, 0.4f, 0.8f, 0.25f});
  anari::commitParameters(d, r);
  failures +=
      !s.expect("flat background midpoint", Vec4{0.1f, 0.2f, 0.4f, 1.f});
  // Partial coverage keeps the existing compositor convention: .5*.5*B
  // from the surface, .5*B from the backdrop, alpha=.5+.5*.25.
  // Quality's stochastic coverage needs its own convergence reference.
  if (std::string(s.subtype) != "quality") {
    s.opacity(0.5f);
    failures += !s.expect(
        "coverage preserves backdrop alpha", Vec4{0.15f, 0.3f, 0.6f, 0.625f});
    s.opacity(1.f);
  }
  return failures;
}

int testEncoding(Scene &s)
{
  const auto d = s.d;
  const auto r = s.renderer;
  anari::setParameter(d, s.frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::commitParameters(d, s.frame);
  int failures = 0;
  for (bool premultiply : {false, true}) {
    anari::setParameter(d, r, "premultiplyBackground", premultiply);
    anari::commitParameters(d, r);
    anari::render(d, s.frame);
    anari::wait(d, s.frame);
    using Pixel8 = std::array<unsigned char, 4>;
    const auto pixels = readChannel<Pixel8>(
        d, s.frame, "channel.color", WIDTH, HEIGHT, ANARI_UFIXED8_RGBA_SRGB);
    // Linear surface midpoint is (.1,.2,.4), or one quarter of that under
    // backdrop premultiplication. Encode only AFTER this linear fog blend.
    constexpr double LINEAR[] = {0.1, 0.2, 0.4};
    for (unsigned p = 0; p < WIDTH * HEIGHT; ++p) {
      for (int c = 0; c < 3; ++c) {
        const double value = LINEAR[c] * (premultiply ? 0.25 : 1.0);
        const double srgb = value <= 0.0031308
            ? 12.92 * value
            : 1.055 * std::pow(value, 1.0 / 2.4) - 0.055;
        // One code value permits UNORM8 quantization and encoder rounding,
        // not a linear-space relative error on an encoded measurement.
        if (std::abs(int(pixels[p][c]) - int(std::lround(255.0 * srgb))) > 1) {
          fprintf(stderr,
              "%s background fog encoded before linear blend\n",
              s.subtype);
          ++failures;
        }
      }
      failures += pixels[p][3] != 255;
    }
  }
  anari::setParameter(d, s.frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::commitParameters(d, s.frame);
  return failures;
}

int testImage(Scene &s)
{
  const auto d = s.d;
  const auto r = s.renderer;
  constexpr std::array<Vec4, 4> TEXELS = {Vec4{0.25f, 0.5f, 1.f, 0.5f},
      Vec4{1.25f, 0.5f, 1.5f, 0.5f},
      Vec4{0.25f, 1.5f, 2.f, 0.5f},
      Vec4{1.25f, 1.5f, 2.5f, 0.5f}};
  auto image = anari::newArray2D(d, ANARI_FLOAT32_VEC4, 2, 2);
  const ObjectOwner imageOwner(d, image);
  {
    const ArrayMapping<Vec4> initial(d, image);
    for (unsigned i = 0; i < TEXELS.size(); ++i)
      initial.data()[i] = TEXELS[i];
  }
  anari::setParameter(d, r, "background", image);
  // The frame samples texel centers at (x+.5)/8,(y+.5)/4. These are the
  // independently worked clamped bilinear weights for a 2x2 image.
  constexpr float WX[] = {0.f, 0.f, 0.125f, 0.375f, 0.625f, 0.875f, 1.f, 1.f};
  constexpr float WY[] = {0.f, 0.25f, 0.75f, 1.f};
  ColorImage backdrop;
  for (unsigned y = 0; y < HEIGHT; ++y)
    for (unsigned x = 0; x < WIDTH; ++x)
      backdrop.push_back(
          {0.25f + WX[x], 0.5f + WY[y], 1.f + 0.5f * WX[x] + WY[y], 0.5f});
  int failures = 0;
  for (bool region : {false, true}) {
    if (region) {
      constexpr float BOX[] = {0.25f, 0.125f, 0.75f, 0.625f};
      anari::setParameter(d, s.camera, "imageRegion", ANARI_FLOAT32_BOX2, BOX);
    } else
      anari::unsetParameter(d, s.camera, "imageRegion");
    anari::commitParameters(d, s.camera);
    for (bool premultiply : {false, true}) {
      anari::setParameter(d, r, "premultiplyBackground", premultiply);
      anari::commitParameters(d, r);
      ColorImage expected = backdrop;
      for (auto &pixel : expected) {
        for (int c = 0; c < 3; ++c)
          pixel[c] *= premultiply ? 0.25f : 0.5f;
        pixel[3] = 1.f;
      }
      failures += !s.expect("image midpoint at frame coordinates", expected);
      // Later jittered samples must still use the frame-center convention.
      for (int i = 0; i < 4; ++i)
        failures += !s.expect("image accumulated samples", expected);
      s.showSurface(false);
      ColorImage miss = backdrop;
      if (premultiply)
        for (auto &pixel : miss)
          for (int c = 0; c < 3; ++c)
            pixel[c] *= 0.5f;
      failures += !s.expect("image miss", miss);
      const auto before = s.render();
      anari::setParameter(d, r, "fogMode", "none");
      anari::commitParameters(d, r);
      failures += s.render() != before;
      anari::setParameter(d, r, "fogMode", "linear");
      anari::commitParameters(d, r);
      s.showSurface(true);
    }
  }
  // Populate the old *surface* accumulation after the miss checks above.
  for (int i = 0; i < 4; ++i)
    s.render();
  // Observed-array edits must refresh both texture and accumulated fog without
  // an unrelated Renderer, World, or Frame commit.
  {
    const ArrayMapping<Vec4> mapped(d, image);
    for (unsigned i = 0; i < 4; ++i)
      mapped.data()[i] = {2.f, 4.f, 6.f, 0.5f};
  }
  failures += !s.expect("observed image edit", Vec4{0.5f, 1.f, 1.5f, 1.f});
  anari::unsetParameter(d, r, "background");
  anari::commitParameters(d, r);
  failures += !s.expect(
      "unset image restores black background", Vec4{0.f, 0.f, 0.f, 1.f});
  anari::unsetParameter(d, s.camera, "imageRegion");
  anari::commitParameters(d, s.camera);
  return failures;
}

anari::Light makeHdri(anari::Device d, Vec3 radiance, bool visible)
{
  const std::array<Vec3, 4> texels = {radiance, radiance, radiance, radiance};
  auto light = anari::newObject<anari::Light>(d, "hdri");
  ObjectOwner lightOwner(d, light);
  auto image = anari::newArray2D(d, texels.data(), 2, 2);
  const ObjectOwner imageOwner(d, image);
  anari::setParameter(d, light, "radiance", image);
  anari::setParameter(d, light, "visible", visible);
  anari::commitParameters(d, light);
  lightOwner.release();
  return light;
}

int testHdri(Scene &s)
{
  const auto d = s.d;
  const auto r = s.renderer;
  anari::setParameter(d, r, "background", Vec4{0.2f, 0.4f, 0.8f, 0.25f});
  anari::setParameter(d, r, "premultiplyBackground", false);
  anari::commitParameters(d, r);
  auto first = makeHdri(d, {0.25f, 0.5f, 2.f}, false);
  const ObjectOwner firstOwner(d, first);
  anari::setParameter(d, first, "color", Vec3{0.5f, 1.f, 0.25f});
  anari::setParameter(d, first, "scale", 4.f);
  anari::commitParameters(d, first);
  anari::setParameterArray1D(d, s.world, "light", &first, 1);
  anari::commitParameters(d, s.world);
  int failures = 0;
  failures += !s.expect("invisible HDRI does not replace flat background",
      Vec4{0.1f, 0.2f, 0.4f, 1.f});
  for (int i = 0; i < 4; ++i)
    s.render();
  anari::setParameter(d, first, "visible", true);
  anari::commitParameters(d, first);
  // Radiance * tint * scale = (.5,2,2), half fog -> (.25,1,1).
  failures += !s.expect(
      "visible HDRI tint/scale and precedence", Vec4{0.25f, 1.f, 1.f, 1.f});
  constexpr std::array<Vec4, 4> TEXELS = {Vec4{4.f, 2.f, 1.f, 0.5f},
      Vec4{4.f, 2.f, 1.f, 0.5f},
      Vec4{4.f, 2.f, 1.f, 0.5f},
      Vec4{4.f, 2.f, 1.f, 0.5f}};
  auto image = anari::newArray2D(d, TEXELS.data(), 2, 2);
  const ObjectOwner imageOwner(d, image);
  anari::setParameter(d, r, "background", image);
  anari::commitParameters(d, r);
  failures +=
      !s.expect("visible HDRI overrides image", Vec4{0.25f, 1.f, 1.f, 1.f});
  auto second = makeHdri(d, {1.f, 0.5f, 0.25f}, true);
  const ObjectOwner secondOwner(d, second);
  const anari::Light lights[] = {first, second};
  anari::setParameterArray1D(d, s.world, "light", lights, 2);
  anari::commitParameters(d, s.world);
  failures +=
      !s.expect("sum of visible HDRIs", Vec4{0.75f, 1.25f, 1.125f, 1.f});
  anari::setParameter(d, r, "premultiplyBackground", true);
  anari::commitParameters(d, r);
  failures += !s.expect("HDRI unaffected by flat alpha premultiplication",
      Vec4{0.75f, 1.25f, 1.125f, 1.f});
  s.showSurface(false);
  failures += !s.expect(
      "HDRI miss is opaque and unfogged", Vec4{1.5f, 2.5f, 2.25f, 1.f});
  const auto before = s.render();
  anari::setParameter(d, r, "fogMode", "none");
  anari::commitParameters(d, r);
  failures += s.render() != before;
  anari::setParameter(d, r, "fogMode", "linear");
  anari::commitParameters(d, r);
  s.showSurface(true);
  anari::setParameter(d, first, "visible", false);
  anari::commitParameters(d, first);
  failures +=
      !s.expect("skip invisible first HDRI", Vec4{0.5f, 0.25f, 0.125f, 1.f});
  anari::setParameter(d, second, "visible", false);
  anari::commitParameters(d, second);
  failures += !s.expect(
      "hidden HDRIs reveal premultiplied image", Vec4{1.f, 0.5f, 0.25f, 1.f});
  anari::unsetParameter(d, r, "background");
  anari::commitParameters(d, r);
  anari::unsetParameter(d, s.world, "light");
  anari::commitParameters(d, s.world);
  return failures;
}

int testDirectionalHdri(Scene &s)
{
  const auto d = s.d;
  const auto r = s.renderer;
  // Wide constant angular bands keep all tested directions away from texture
  // transitions. +Z sees the middle band with direction=+Z/up=+Y; -Z sees
  // the outer band. This independent angular construction avoids depending on
  // the production environment evaluator as the expected-color oracle.
  std::vector<Vec3> texels(16 * 8);
  for (unsigned y = 0; y < 8; ++y)
    for (unsigned x = 0; x < 16; ++x)
      texels[y * 16 + x] =
          x >= 4 && x < 12 ? Vec3{2.f, 0.5f, 0.25f} : Vec3{0.25f, 1.f, 3.f};
  auto light = anari::newObject<anari::Light>(d, "hdri");
  const ObjectOwner lightOwner(d, light);
  auto image = anari::newArray2D(d, texels.data(), 16, 8);
  const ObjectOwner imageOwner(d, image);
  anari::setParameter(d, light, "radiance", image);
  anari::setParameter(d, light, "direction", Vec3{0.f, 0.f, 1.f});
  anari::setParameter(d, light, "up", Vec3{0.f, 1.f, 0.f});
  anari::commitParameters(d, light);
  auto group = anari::newObject<anari::Group>(d);
  const ObjectOwner groupOwner(d, group);
  anari::setParameterArray1D(d, group, "light", &light, 1);
  anari::commitParameters(d, group);
  auto instance = anari::newObject<anari::Instance>(d, "transform");
  const ObjectOwner instanceOwner(d, instance);
  anari::setParameter(d, instance, "group", group);
  anari::commitParameters(d, instance);
  anari::setParameterArray1D(d, s.world, "instance", &instance, 1);
  anari::commitParameters(d, s.world);
  int failures = 0;
  failures += !s.expect("directional HDRI", Vec4{1.f, 0.25f, 0.125f, 1.f});
  anari::setParameter(d, light, "direction", Vec3{0.f, 0.f, -1.f});
  anari::commitParameters(d, light);
  failures +=
      !s.expect("light orientation edit", Vec4{0.125f, 0.5f, 1.5f, 1.f});
  // Column-major rigid 180-degree Y rotation plus translation. Translation
  // must not alter an infinite environment lookup.
  constexpr float TRANSFORM[] = {
      -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 7, 3, -4, 1};
  anari::setParameter(d, instance, "transform", ANARI_FLOAT32_MAT4, TRANSFORM);
  anari::commitParameters(d, instance);
  failures += !s.expect(
      "composed HDRI/instance rotations", Vec4{1.f, 0.25f, 0.125f, 1.f});
  anari::setParameter(d, light, "direction", Vec3{0.f, 0.f, 1.f});
  anari::commitParameters(d, light);
  failures += !s.expect("instance-rotated HDRI", Vec4{0.125f, 0.5f, 1.5f, 1.f});
  s.showSurface(false);
  failures += !s.expect("transformed HDRI miss", Vec4{0.25f, 1.f, 3.f, 1.f});
  s.showSurface(true);
  for (int i = 0; i < 4; ++i)
    s.render();
  {
    const ArrayMapping<Vec3> editedRadiance(d, image);
    for (unsigned i = 0; i < 16 * 8; ++i)
      for (int c = 0; c < 3; ++c)
        editedRadiance.data()[i][c] *= 2.f;
  }
  failures += !s.expect("observed HDRI image edit clears accumulated fog",
      Vec4{0.25f, 1.f, 3.f, 1.f});

  // Perspective samples point to opposite sides of a split panorama, while
  // the camera's reference direction is identical for every pixel. Use full
  // fog on a reflective material: the answer must be the camera backdrop,
  // never the HDRI reached by the material's reflected ray.
  anari::unsetParameter(d, instance, "transform");
  anari::commitParameters(d, instance);
  {
    const ArrayMapping<Vec3> mapped(d, image);
    for (unsigned y = 0; y < 8; ++y)
      for (unsigned x = 0; x < 16; ++x)
        mapped.data()[y * 16 + x] =
            x < 8 ? Vec3{2.f, 0.5f, 0.25f} : Vec3{0.25f, 1.f, 3.f};
  }
  auto perspective = anari::newObject<anari::Camera>(d, "perspective");
  const ObjectOwner perspectiveOwner(d, perspective);
  anari::setParameter(d, perspective, "position", Vec3{0.f, 0.f, -10.f});
  anari::setParameter(d, perspective, "direction", Vec3{0.f, 0.f, 1.f});
  anari::setParameter(d, perspective, "up", Vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, perspective, "fovy", float(1.5707963267948966));
  anari::commitParameters(d, perspective);
  auto mirror = anari::newObject<anari::Material>(d, "physicallyBased");
  const ObjectOwner mirrorOwner(d, mirror);
  anari::setParameter(d, mirror, "baseColor", Vec3{1.f, 1.f, 1.f});
  anari::setParameter(d, mirror, "metallic", 1.f);
  anari::setParameter(d, mirror, "roughness", 0.f);
  anari::commitParameters(d, mirror);
  anari::setParameter(d, s.surface, "material", mirror);
  anari::commitParameters(d, s.surface);
  anari::setParameter(d, s.frame, "camera", perspective);
  anari::commitParameters(d, s.frame);
  anari::setParameter(d, r, "fogEnd", 1.f);
  anari::commitParameters(d, r);
  ColorImage expected;
  for (unsigned y = 0; y < HEIGHT; ++y)
    for (unsigned x = 0; x < WIDTH; ++x)
      expected.push_back(x < WIDTH / 2 ? Vec4{2.f, 0.5f, 0.25f, 1.f}
                                       : Vec4{0.25f, 1.f, 3.f, 1.f});
  failures +=
      !s.expect("original perspective direction, not reflection", expected);
  // Full fog is numerically exact here; only the first centered camera sample
  // is used, so no stochastic tolerance is necessary even for this mirror.
  anari::setParameter(d, s.surface, "material", s.material);
  anari::commitParameters(d, s.surface);
  anari::setParameter(d, s.frame, "camera", s.camera);
  anari::commitParameters(d, s.frame);
  anari::setParameter(d, r, "fogEnd", 20.f);
  anari::commitParameters(d, r);
  anari::unsetParameter(d, s.world, "instance");
  anari::commitParameters(d, s.world);
  return failures;
}

int testLifecycle(
    Scene &s, std::vector<std::string> &warnings, const char *metric)
{
  const auto d = s.d;
  const auto r = s.renderer;
  int failures = 0;
  constexpr Vec4 BG{0.2f, 0.4f, 0.8f, 0.25f};
  anari::setParameter(d, r, "background", BG);
  anari::setParameter(d, r, "premultiplyBackground", false);
  anari::commitParameters(d, r);
  for (const Vec3 color : {Vec3{10.f, 0.1f, 2.f},
           Vec3{-1.f,
               std::numeric_limits<float>::quiet_NaN(),
               std::numeric_limits<float>::infinity()}}) {
    warnings.clear();
    anari::setParameter(d, r, "fogColor", color);
    anari::commitParameters(d, r);
    failures +=
        !s.expect("background ignores fogColor, including invalid values",
            Vec4{0.1f, 0.2f, 0.4f, 1.f});
    failures += !warnings.empty();
  }
  warnings.clear();
  anari::setParameter(d, r, "fogColor", "ignored wrong type");
  anari::commitParameters(d, r);
  failures += !s.expect(
      "background ignores fogColor type", Vec4{0.1f, 0.2f, 0.4f, 1.f});
  failures += !warnings.empty();

  auto secondFrame = anari::newObject<anari::Frame>(d);
  const ObjectOwner secondFrameOwner(d, secondFrame);
  anari::setParameter(d, secondFrame, "size", UVec2{WIDTH, HEIGHT});
  anari::setParameter(d, secondFrame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, secondFrame, "world", s.world);
  anari::setParameter(d, secondFrame, "camera", s.camera);
  constexpr const char *SOURCES[] = {
      "background", "constant", nullptr, "invalid", "background"};
  for (unsigned step = 0; step < 5; ++step) {
    for (int i = 0; i < 4; ++i)
      s.render();
    const Vec4 background = step == 4 ? Vec4{2.f, 4.f, 6.f, 0.5f} : BG;
    auto configure = [&](anari::Renderer target) {
      anari::setParameter(d, target, "fogDistanceMetric", metric);
      anari::setParameter(d, target, "background", background);
      anari::setParameter(d, target, "premultiplyBackground", false);
      anari::setParameter(d, target, "fogColor", Vec3{0.8f, 0.6f, 0.4f});
      if (SOURCES[step])
        anari::setParameter(d, target, "fogColorSource", SOURCES[step]);
      else
        anari::unsetParameter(d, target, "fogColorSource");
      anari::commitParameters(d, target);
    };
    warnings.clear();
    configure(r);
    Vec4 expected{0.4f, 0.3f, 0.2f, 1.f};
    if (step == 0)
      expected = {0.1f, 0.2f, 0.4f, 1.f};
    else if (step == 3)
      expected = {0.f, 0.f, 0.f, 1.f};
    else if (step == 4)
      expected = {1.f, 2.f, 3.f, 1.f};
    failures += !s.expect("source switch/unset/invalid/recovery", expected);
    if (step == 3) {
      bool warned = false;
      for (const auto &message : warnings)
        warned |= message.find("fogColorSource") != std::string::npos;
      if (!warned) {
        fprintf(stderr, "%s invalid selector did not warn\n", s.subtype);
        ++failures;
      }
      s.showSurface(false);
      failures += !s.expect("invalid source preserves unrelated backdrop", BG);
      s.showSurface(true);
    } else
      failures += !warnings.empty();
    auto fresh = s.newRenderer();
    const ObjectOwner freshOwner(d, fresh);
    configure(fresh);
    anari::setParameter(d, secondFrame, "renderer", fresh);
    anari::commitParameters(d, secondFrame);
    if (s.render(secondFrame) != s.render()) {
      fprintf(stderr,
          "%s reused/fresh source state differs at step %u\n",
          s.subtype,
          step);
      ++failures;
    }
    anari::setParameter(d, fresh, "fogColorSource", "constant");
    anari::setParameter(d, fresh, "fogColor", Vec3{6.f, 4.f, 2.f});
    anari::commitParameters(d, fresh);
    const ColorImage otherView(WIDTH * HEIGHT, Vec4{3.f, 2.f, 1.f, 1.f});
    failures += s.render(secondFrame) != otherView;
    failures += !s.expect("two concurrent Frames sharing a World", expected);
  }
  // Wrong selector type is an active error; none must ignore it entirely.
  warnings.clear();
  anari::setParameter(d, r, "fogColorSource", 42);
  anari::commitParameters(d, r);
  failures +=
      !s.expect("wrong source type disables fog", Vec4{0.f, 0.f, 0.f, 1.f});
  bool warned = false;
  for (const auto &message : warnings)
    warned |= message.find("fogColorSource") != std::string::npos;
  failures += !warned;
  warnings.clear();
  anari::setParameter(d, r, "fogMode", "none");
  anari::commitParameters(d, r);
  failures +=
      !s.expect("none ignores invalid source", Vec4{0.f, 0.f, 0.f, 1.f});
  failures += !warnings.empty();
  anari::setParameter(d, r, "fogMode", "linear");
  anari::setParameter(d, r, "fogColorSource", "background");
  anari::commitParameters(d, r);
  failures +=
      !s.expect("corrected selector recovers", Vec4{1.f, 2.f, 3.f, 1.f});
  // Changing only the backdrop after several old samples must clear history.
  for (int i = 0; i < 4; ++i)
    s.render();
  anari::setParameter(d, r, "background", BG);
  anari::commitParameters(d, r);
  failures += !s.expect(
      "flat backdrop edit clears accumulated fog", Vec4{0.1f, 0.2f, 0.4f, 1.f});
  return failures;
}

int testQualityIndirect(anari::Device d)
{
  Scene s(d, "quality");
  const auto r = s.renderer;
  auto mirror = anari::newObject<anari::Material>(d, "physicallyBased");
  const ObjectOwner mirrorOwner(d, mirror);
  anari::setParameter(d, mirror, "baseColor", Vec3{0.8f, 0.5f, 0.25f});
  anari::setParameter(d, mirror, "metallic", 1.f);
  anari::setParameter(d, mirror, "roughness", 0.2f);
  anari::setParameter(d, mirror, "emissive", Vec3{0.1f, 0.2f, 0.4f});
  anari::commitParameters(d, mirror);
  anari::setParameter(d, s.surface, "material", mirror);
  anari::setParameter(d, s.surface, "id", 42u);
  anari::commitParameters(d, s.surface);
  constexpr Vec3 RADIANCE{2.f, 0.5f, 1.f};
  auto light = makeHdri(d, RADIANCE, false);
  const ObjectOwner lightOwner(d, light);
  anari::setParameterArray1D(d, s.world, "light", &light, 1);
  anari::commitParameters(d, s.world);
  anari::setParameter(d, s.frame, "channel.depth", ANARI_FLOAT32);
  for (const char *name : {"channel.normal", "channel.albedo"})
    anari::setParameter(d, s.frame, name, ANARI_FLOAT32_VEC3);
  for (const char *name :
      {"channel.primitiveId", "channel.objectId", "channel.instanceId"})
    anari::setParameter(d, s.frame, name, ANARI_UINT32);
  anari::commitParameters(d, s.frame);
  constexpr Vec4 BACKDROP{0.4f, 0.8f, 0.2f, 0.25f};
  constexpr Vec3 CONSTANT{4.f, 2.f, 0.5f};
  anari::setParameter(d, r, "background", BACKDROP);
  anari::setParameter(d, r, "premultiplyBackground", false);
  anari::setParameter(d, r, "pixelSamples", 64);
  anari::setParameter(d, r, "fogMode", "none");
  anari::setParameter(d, r, "maxRayDepth", 1);
  anari::commitParameters(d, r);
  const auto directOnly = s.render();
  anari::setParameter(d, r, "maxRayDepth", 5);
  anari::commitParameters(d, r);
  const auto baseline = s.render();
  double indirectMean = 0.;
  for (unsigned p = 0; p < baseline.size(); ++p)
    indirectMean += (baseline[p][0] - directOnly[p][0]) / baseline.size();
  int failures = 0;
  // This fixture must actually return radiance from the secondary environment,
  // not merely pass an affine check on direct lighting or a black surface.
  if (!(indirectMean > 0.5)) {
    fprintf(
        stderr, "quality indirect fixture returned only %g\n", indirectMean);
    ++failures;
  }
  const auto baselineAux = readImage(d, s.frame, {WIDTH, HEIGHT});
  auto sameAuxiliary = [&]() {
    return visrtx::fogtest::sameAuxiliary(
        baselineAux, readImage(d, s.frame, {WIDTH, HEIGHT}), s.subtype);
  };
  // These are paired 64-sample runs: each commit restarts the same public Frame
  // accumulation/seed sequence. The reference is the specified affine operation
  // on the unfogged shaded RGB, NOT another renderer's transport estimator.
  // Thus the ordinary 1e-5 + 1e-4*abs(reference) tolerance applies, without a
  // stochastic relaxation. Altering transport random decisions breaks pairing.
  constexpr const char *MODES[] = {"linear", "exp", "exp2", "linear"};
  constexpr double FRACTIONS[] = {
      0.5, 0.6321205588285577, 0.2211992169285951, 1.};
  for (bool visible : {false, true}) {
    anari::setParameter(d, light, "visible", visible);
    anari::commitParameters(d, light);
    for (const char *source : {"constant", "background"}) {
      for (const char *metric : {"viewDepth", "rayDistance"}) {
        for (int curve = 0; curve < 4; ++curve) {
          anari::setParameter(d, r, "fogMode", MODES[curve]);
          anari::setParameter(d, r, "fogColorSource", source);
          anari::setParameter(d, r, "fogDistanceMetric", metric);
          anari::setParameter(d, r, "fogColor", CONSTANT);
          anari::setParameter(d, r, "fogEnd", curve == 3 ? 1.f : 20.f);
          anari::setParameter(d, r, "fogDensity", curve == 2 ? 0.05f : 0.1f);
          anari::commitParameters(d, r);
          ColorImage expected = baseline;
          for (auto &pixel : expected) {
            for (int c = 0; c < 3; ++c) {
              const double color = std::string(source) == "constant"
                  ? CONSTANT[c]
                  : visible ? RADIANCE[c]
                            : BACKDROP[c];
              pixel[c] = float((1. - FRACTIONS[curve]) * pixel[c]
                  + FRACTIONS[curve] * color);
            }
          }
          failures +=
              !s.expect("opaque indirect radiance fogged once", expected);
          failures += !sameAuxiliary();
        }
      }
    }
    anari::setParameter(d, r, "fogMode", "none");
    anari::commitParameters(d, r);
    // Visibility affects only the camera backdrop, never reflected
    // illumination.
    failures += s.render() != baseline;
    failures += !sameAuxiliary();
  }
  printf("quality opaque indirect: 64 spp, secondary red contribution %.8f\n",
      indirectMean);
  return failures;
}

void countReleasedArray(const void *user, const void *)
{
  ++*static_cast<unsigned *>(const_cast<void *>(user));
}

int testResourceUnwinding(anari::Device d, const char *subtype)
{
  // Observe ownership via ANARI's application-memory deleter, not a mock or
  // an internal reference count. A client allocation failure must still free
  // an unattached temporary array exactly once.
  unsigned releases = 0;
  std::array<float, 1> storage{};
  try {
    auto array = anari::newArray1D(
        d, storage.data(), countReleasedArray, &releases, storage.size());
    const ObjectOwner owner(d, array);
    throw std::bad_alloc();
  } catch (const std::bad_alloc &) {
  }
  int failures = releases != 1;

  Scene scene(d, subtype);
  auto image = anari::newArray2D(d, ANARI_FLOAT32_VEC4, 1, 1);
  const ObjectOwner imageOwner(d, image);
  {
    const ArrayMapping<Vec4> pixels(d, image);
    pixels.data()[0] = {0.f, 0.f, 0.f, 1.f};
  }
  anari::setParameter(d, scene.renderer, "background", image);
  anari::commitParameters(d, scene.renderer);
  failures +=
      !scene.expect("array before interrupted edit", Vec4{0.f, 0.f, 0.f, 1.f});
  try {
    const ArrayMapping<Vec4> pixels(d, image);
    pixels.data()[0] = {1.f, 0.5f, 0.25f, 1.f};
    throw std::bad_alloc();
  } catch (const std::bad_alloc &) {
  }
  // No recommit: unmapping must publish the edit and invalidate old color.
  failures += !scene.expect(
      "array edit survives unwinding", Vec4{0.5f, 0.25f, 0.125f, 1.f});
  const auto baseline = scene.render();
  try {
    const FrameMapping<Vec4> mapping(d, scene.frame, "channel.color");
    mapping.checkedData(WIDTH, HEIGHT, ANARI_FLOAT32_VEC4);
    // Same unwind path as an allocating channel copy that runs out of memory.
    throw std::bad_alloc();
  } catch (const std::bad_alloc &) {
  }
  failures += scene.render() != baseline;

  // Bad dimensions/type must fail before a copy can read beyond the mapping.
  // Both validation exceptions must also leave the Frame usable.
  for (bool wrongType : {false, true}) {
    bool rejected = false;
    try {
      readChannel<Vec4>(d,
          scene.frame,
          "channel.color",
          wrongType ? WIDTH : WIDTH + 1,
          HEIGHT,
          wrongType ? ANARI_FLOAT32_VEC3 : ANARI_FLOAT32_VEC4);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    failures += !rejected;
    failures += scene.render() != baseline;
  }
  printf("%s public-ANARI resource unwinding/readback recovery: %d failures\n",
      subtype,
      failures);
  return failures;
}

} // namespace

int main()
{
  std::vector<std::string> warnings;
  auto d = makeVisRTXDevice(collectWarningsStatus, &warnings);
  const ObjectOwner deviceOwner(d, d);
  anari::commitParameters(d, d);
  int failures = requireRendererFogSupport(d);
  for (const char *subtype : {"fast", "interactive", "default", "quality"}) {
    failures += testResourceUnwinding(d, subtype);
    for (const char *metric : {"viewDepth", "rayDistance"}) {
      Scene s(d, subtype);
      anari::setParameter(d, s.renderer, "fogDistanceMetric", metric);
      anari::commitParameters(d, s.renderer);
      failures += testFlat(s);
      failures += testEncoding(s);
      failures += testImage(s);
      failures += testHdri(s);
      failures += testDirectionalHdri(s);
      failures += testLifecycle(s, warnings, metric);
      printf("%s background forms, invalidation and lifecycle: %s\n",
          subtype,
          metric);
    }
  }
  failures += testQualityIndirect(d);
  printf("background fog: %d failures\n", failures);
  return failures ? 1 : 0;
}

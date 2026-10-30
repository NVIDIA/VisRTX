/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// Camera 'near'/'far' clip planes and Miss Depth (ADR 0011), on every renderer.
// A pixel whose camera ray hits nothing reports the camera's Far Limit as its
// depth: the distance along that ray to the 'far' plane, the ray buffer's
// 'ray.tmax', or FLT_MAX when nothing bounds the ray. Clipping the ray short
// any other way (a renderer cutting plane, an invalid ray buffer) does not
// change what a Miss reports.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/anari_cpp.hpp>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static int g_clipWarnings = 0;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject source,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL][%p] %s\n", source, message);
    std::exit(1);
  } else if (severity == ANARI_SEVERITY_ERROR)
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
  else if (severity == ANARI_SEVERITY_WARNING && std::strstr(message, "'near'")
      && std::strstr(message, "'far'"))
    g_clipWarnings++;
}

static constexpr uvec2 IMAGE_SIZE = {64, 48};
static constexpr unsigned NUM_PIXELS = IMAGE_SIZE[0] * IMAGE_SIZE[1];
static constexpr unsigned MISS = ~0u;
static constexpr float FLOAT_MAX = std::numeric_limits<float>::max();

static constexpr vec3 CAM_POS = {0.f, 0.f, 5.f};
static constexpr vec3 CAM_DIR = {0.f, 0.f, -1.f};
static constexpr vec3 CAM_UP = {0.f, 1.f, 0.f};
static constexpr float CAM_FOVY = 0.8f;
static constexpr float ORTHO_HEIGHT = 6.f;

// Sphere 0 sits ~4.4 from the camera; sphere 1 is larger, partly behind it,
// and ~12 away. 'far' between them clips sphere 1; 'near' between them clips
// sphere 0.
static constexpr float FAR_SPHERE_FRONT = 12.f;
static constexpr float CLIP_BETWEEN = 8.f;

static vec3 operator+(vec3 a, vec3 b)
{
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
static vec3 operator*(float s, vec3 a)
{
  return {s * a[0], s * a[1], s * a[2]};
}
static float dot(vec3 a, vec3 b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static vec3 cross(vec3 a, vec3 b)
{
  return {a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]};
}
static vec3 normalize(vec3 a)
{
  return (1.f / std::sqrt(dot(a, a))) * a;
}

struct Image
{
  std::vector<float> depth;
  std::vector<unsigned> primID;
};

static anari::World makeWorld(ANARIDevice device)
{
  const std::array<vec3, 2> centers = {
      vec3{0.f, 0.f, 0.f}, vec3{1.5f, 0.f, -10.f}};
  const std::array<float, 2> radii = {0.6f, 3.f};

  auto geom = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setParameterArray1D(
      device, geom, "vertex.position", centers.data(), centers.size());
  anari::setParameterArray1D(
      device, geom, "vertex.radius", radii.data(), radii.size());
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "matte");
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);
  return world;
}

static Image render(ANARIDevice device,
    const char *rendererType,
    anari::World world,
    anari::Camera cam,
    const vec4 *cutPlane = nullptr)
{
  auto renderer = anari::newObject<anari::Renderer>(device, rendererType);
  anari::setParameter(device, renderer, "pixelSamples", 1);
  if (cutPlane)
    anari::setParameter(device, renderer, "cutPlane", *cutPlane);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", IMAGE_SIZE);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(device, frame, "channel.primitiveId", ANARI_UINT32);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", cam);
  anari::setAndReleaseParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  // The first sample of the first frame is pixel-centered for every camera.
  anari::render(device, frame);
  anari::wait(device, frame);

  Image img;
  auto depth = anari::map<float>(device, frame, "channel.depth");
  img.depth.assign(depth.data, depth.data + NUM_PIXELS);
  anari::unmap(device, frame, "channel.depth");
  auto ids = anari::map<uint32_t>(device, frame, "channel.primitiveId");
  img.primID.assign(ids.data, ids.data + NUM_PIXELS);
  anari::unmap(device, frame, "channel.primitiveId");

  anari::release(device, frame);
  return img;
}

static anari::Camera makeCamera(ANARIDevice device, const char *type)
{
  auto cam = anari::newObject<anari::Camera>(device, type);
  anari::setParameter(device, cam, "position", CAM_POS);
  anari::setParameter(device, cam, "direction", CAM_DIR);
  anari::setParameter(device, cam, "up", CAM_UP);
  anari::setParameter(
      device, cam, "aspect", IMAGE_SIZE[0] / float(IMAGE_SIZE[1]));
  if (std::strcmp(type, "perspective") == 0)
    anari::setParameter(device, cam, "fovy", CAM_FOVY);
  else
    anari::setParameter(device, cam, "height", ORTHO_HEIGHT);
  return cam;
}

// Pixel-center ray directions of the perspective camera, row-major.
static std::vector<vec3> perspectiveDirections()
{
  const float h = 2.f * std::tan(0.5f * CAM_FOVY);
  const float w = h * IMAGE_SIZE[0] / float(IMAGE_SIZE[1]);
  const vec3 du = w * normalize(cross(CAM_DIR, CAM_UP));
  const vec3 dv = h * normalize(cross(du, CAM_DIR));
  const vec3 d00 = CAM_DIR + -0.5f * du + -0.5f * dv;

  std::vector<vec3> dirs(NUM_PIXELS);
  for (unsigned y = 0; y < IMAGE_SIZE[1]; y++) {
    for (unsigned x = 0; x < IMAGE_SIZE[0]; x++) {
      const float sx = (x + 0.5f) / IMAGE_SIZE[0];
      const float sy = (y + 0.5f) / IMAGE_SIZE[1];
      dirs[y * IMAGE_SIZE[0] + x] = normalize(d00 + sx * du + sy * dv);
    }
  }
  return dirs;
}

// Far Limit of each perspective pixel ray for a far plane at 'farPlane'.
static std::vector<float> perspectiveFarLimits(float farPlane)
{
  std::vector<float> limits;
  for (const vec3 &d : perspectiveDirections())
    limits.push_back(farPlane / dot(d, CAM_DIR));
  return limits;
}

static bool nearlyEqual(float a, float b)
{
  return std::fabs(a - b) <= 1e-4f * std::fmax(1.f, std::fabs(b));
}

struct Counts
{
  unsigned misses{0};
  unsigned badMissDepth{0};
  unsigned nearHits{0};
  unsigned farHits{0};
  unsigned badHitDepth{0};
};

// Every Miss must report missDepth[i]; every hit must lie in [lo, hi].
static Counts tally(const Image &img,
    const std::vector<float> &missDepth,
    float lo = 0.f,
    float hi = FLOAT_MAX)
{
  Counts c;
  for (unsigned i = 0; i < NUM_PIXELS; i++) {
    if (img.primID[i] == MISS) {
      c.misses++;
      c.badMissDepth += !nearlyEqual(img.depth[i], missDepth[i]);
    } else {
      c.nearHits += img.primID[i] == 0;
      c.farHits += img.primID[i] == 1;
      c.badHitDepth += img.depth[i] < lo || img.depth[i] > hi;
    }
  }
  return c;
}

static bool check(bool cond, const char *renderer, const char *what)
{
  printf("  %-12s %-50s %s\n", renderer, what, cond ? "ok" : "FAIL");
  return cond;
}

static bool testRenderer(
    ANARIDevice device, anari::World world, const char *rendererType)
{
  bool ok = true;
  const std::vector<float> unbounded(NUM_PIXELS, FLOAT_MAX);
  const bool hasCutPlane = std::strcmp(rendererType, "debug") != 0;

  // 1. No 'far': nothing is clipped and a Miss reports FLT_MAX.
  {
    auto cam = makeCamera(device, "perspective");
    anari::commitParameters(device, cam);
    const Counts c = tally(render(device, rendererType, world, cam), unbounded);
    anari::release(device, cam);
    ok &= check(c.misses > 0 && c.nearHits > 0 && c.farHits > 0,
        rendererType,
        "unclipped: both spheres and misses visible");
    ok &= check(c.badMissDepth == 0, rendererType, "unclipped: Miss = FLT_MAX");
  }

  // 2. Perspective 'far' between the spheres: the far sphere is clipped and a
  //    Miss reports the distance to the far plane along its ray.
  {
    auto cam = makeCamera(device, "perspective");
    anari::setParameter(device, cam, "far", CLIP_BETWEEN);
    anari::commitParameters(device, cam);
    const Counts c = tally(render(device, rendererType, world, cam),
        perspectiveFarLimits(CLIP_BETWEEN),
        0.f,
        CLIP_BETWEEN);
    anari::release(device, cam);
    ok &= check(c.nearHits > 0 && c.farHits == 0 && c.badHitDepth == 0,
        rendererType,
        "perspective far clips the far sphere");
    ok &=
        check(c.badMissDepth == 0, rendererType, "perspective Miss = far/cos");
  }

  // 3. Perspective 'near' between the spheres: the near sphere is clipped.
  {
    auto cam = makeCamera(device, "perspective");
    anari::setParameter(device, cam, "near", CLIP_BETWEEN);
    anari::commitParameters(device, cam);
    const Counts c = tally(render(device, rendererType, world, cam),
        unbounded,
        FAR_SPHERE_FRONT - 1.f);
    anari::release(device, cam);
    ok &= check(c.nearHits == 0 && c.farHits > 0 && c.badHitDepth == 0
            && c.badMissDepth == 0,
        rendererType,
        "perspective near clips the near sphere");
  }

  // 4. Orthographic 'far': every Miss reports exactly 'far'.
  {
    auto cam = makeCamera(device, "orthographic");
    anari::setParameter(device, cam, "far", CLIP_BETWEEN);
    anari::commitParameters(device, cam);
    const Counts c = tally(render(device, rendererType, world, cam),
        std::vector<float>(NUM_PIXELS, CLIP_BETWEEN));
    anari::release(device, cam);
    ok &= check(c.nearHits > 0 && c.farHits == 0 && c.badMissDepth == 0,
        rendererType,
        "orthographic Miss = far");
  }

  // 5. A cutting plane hiding everything past z = 2 turns every pixel into a
  //    Miss, which still reports the Far Limit, not the cut distance.
  if (hasCutPlane) {
    const vec4 cut = {0.f, 0.f, 1.f, -2.f};
    auto cam = makeCamera(device, "perspective");
    anari::setParameter(device, cam, "far", CLIP_BETWEEN);
    anari::commitParameters(device, cam);
    const Counts c = tally(render(device, rendererType, world, cam, &cut),
        perspectiveFarLimits(CLIP_BETWEEN));
    anari::release(device, cam);
    ok &= check(c.misses == NUM_PIXELS && c.badMissDepth == 0,
        rendererType,
        "cutting plane Miss = Far Limit");
  }

  // 6. Ray buffer: a Miss reports its own ray.tmax (all short of the spheres).
  {
    const auto dirs = perspectiveDirections();
    std::vector<float> tmax(NUM_PIXELS);
    for (unsigned i = 0; i < NUM_PIXELS; i++)
      tmax[i] = 1.f + float(i % 3);

    auto cam = anari::newObject<anari::Camera>(device, "rayBuffer");
    anari::setParameter(device, cam, "position", CAM_POS);
    anari::setAndReleaseParameter(device,
        cam,
        "ray.dir",
        anari::newArray2D(device, dirs.data(), IMAGE_SIZE[0], IMAGE_SIZE[1]));
    anari::setAndReleaseParameter(device,
        cam,
        "ray.tmax",
        anari::newArray2D(device, tmax.data(), IMAGE_SIZE[0], IMAGE_SIZE[1]));
    anari::commitParameters(device, cam);
    const Counts c = tally(render(device, rendererType, world, cam), tmax);
    anari::release(device, cam);
    ok &= check(c.misses == NUM_PIXELS && c.badMissDepth == 0,
        rendererType,
        "ray buffer Miss = ray.tmax");
  }

  // 7. Ray buffer sized unlike the frame: every ray misses with FLT_MAX.
  {
    const std::vector<vec3> dirs(4, CAM_DIR);
    auto cam = anari::newObject<anari::Camera>(device, "rayBuffer");
    anari::setParameter(device, cam, "position", CAM_POS);
    anari::setAndReleaseParameter(
        device, cam, "ray.dir", anari::newArray2D(device, dirs.data(), 2, 2));
    anari::commitParameters(device, cam);
    const Counts c = tally(render(device, rendererType, world, cam), unbounded);
    anari::release(device, cam);
    ok &= check(c.misses == NUM_PIXELS && c.badMissDepth == 0,
        rendererType,
        "invalid ray buffer Miss = FLT_MAX");
  }

  return ok;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  auto world = makeWorld(device);

  bool ok = true;

  for (const char *renderer :
      {"default", "debug", "fast", "interactive", "quality"})
    ok &= testRenderer(device, world, renderer);

  // Invalid pairs warn and leave the camera unclipped.
  {
    struct Case
    {
      const char *camera;
      float nearPlane;
      float farPlane;
    };
    const Case cases[] = {
        {"perspective", CLIP_BETWEEN, 2.f}, // near > far
        {"perspective", 0.f, CLIP_BETWEEN}, // perspective needs near > 0
        {"orthographic", -1.f, CLIP_BETWEEN}, // negative near
    };
    for (const Case &k : cases) {
      g_clipWarnings = 0;
      auto cam = makeCamera(device, k.camera);
      anari::setParameter(device, cam, "near", k.nearPlane);
      anari::setParameter(device, cam, "far", k.farPlane);
      anari::commitParameters(device, cam);
      const Counts c = tally(render(device, "default", world, cam),
          std::vector<float>(NUM_PIXELS, FLOAT_MAX));
      anari::release(device, cam);
      ok &= check(g_clipWarnings > 0 && c.nearHits > 0 && c.farHits > 0
              && c.badMissDepth == 0,
          k.camera,
          "invalid near/far warns and is ignored");
    }
  }

  // Orthographic allows near == 0.
  {
    g_clipWarnings = 0;
    auto cam = makeCamera(device, "orthographic");
    anari::setParameter(device, cam, "near", 0.f);
    anari::setParameter(device, cam, "far", CLIP_BETWEEN);
    anari::commitParameters(device, cam);
    const Counts c = tally(render(device, "default", world, cam),
        std::vector<float>(NUM_PIXELS, CLIP_BETWEEN));
    anari::release(device, cam);
    ok &= check(g_clipWarnings == 0 && c.farHits == 0 && c.badMissDepth == 0,
        "orthographic",
        "near == 0 is valid");
  }

  anari::release(device, world);
  anari::release(device, device);

  if (!ok) {
    fprintf(stderr, "FAIL: camera near/far\n");
    return 1;
  }
  printf("camera near/far test passed\n");
  return 0;
}

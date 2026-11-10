// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Public ANARI regression. A full-coverage quad has a linear vertex-color
// gradient, so its albedo directly observes the camera's subpixel coordinates.
// This checks actual raygen ordinals, mapping and normalization, not an image
// comparison between different PCG-dependent batching schedules.
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include "../unit/haltonReference.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;
using UVec2 = std::array<unsigned, 2>;
constexpr unsigned SIZE = 8;

static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

struct Image
{
  std::vector<Vec3> albedo;
  std::vector<Vec4> color;
  bool operator==(const Image &other) const
  {
    return albedo == other.albedo && color == other.color;
  }
};

struct Scene
{
  anari::Device d;
  anari::World world;
  anari::Camera camera;
  anari::Renderer renderer;

  Scene(anari::Device device,
      const char *subtype,
      int spp,
      bool checkerboard,
      bool perspective = false,
      float aperture = 0.f)
      : d(device)
  {
    const std::array<Vec3, 4> positions = {Vec3{-1.f, -1.f, 0.f},
        Vec3{1.f, -1.f, 0.f},
        Vec3{1.f, 1.f, 0.f},
        Vec3{-1.f, 1.f, 0.f}};
    const std::array<Vec4, 4> colors = {Vec4{0.f, 0.f, .25f, 1.f},
        Vec4{1.f, 0.f, .25f, 1.f},
        Vec4{1.f, 1.f, .25f, 1.f},
        Vec4{0.f, 1.f, .25f, 1.f}};
    auto geometry = anari::newObject<anari::Geometry>(d, "quad");
    anari::setParameterArray1D(
        d, geometry, "vertex.position", positions.data(), 4);
    anari::setParameterArray1D(d, geometry, "vertex.color", colors.data(), 4);
    anari::commitParameters(d, geometry);
    auto material = anari::newObject<anari::Material>(d, "matte");
    anari::setParameter(d, material, "color", "color");
    anari::commitParameters(d, material);
    auto surface = anari::newObject<anari::Surface>(d);
    anari::setAndReleaseParameter(d, surface, "geometry", geometry);
    anari::setAndReleaseParameter(d, surface, "material", material);
    anari::commitParameters(d, surface);
    world = anari::newObject<anari::World>(d);
    anari::setParameterArray1D(d, world, "surface", &surface, 1);
    anari::release(d, surface);
    anari::commitParameters(d, world);
    camera = anari::newObject<anari::Camera>(
        d, perspective ? "perspective" : "orthographic");
    // Plane depth is 1, focus depth is 2: lens displacement survives at the
    // plane instead of cancelling as it would on the focal plane.
    anari::setParameter(d, camera, "fovy", float(std::acos(-1.) / 3.));
    anari::setParameter(d, camera, "focusDistance", 2.f);
    anari::setParameter(d, camera, "apertureRadius", aperture);
    anari::setParameter(d, camera, "position", Vec3{0.f, 0.f, 1.f});
    anari::setParameter(d, camera, "direction", Vec3{0.f, 0.f, -1.f});
    anari::setParameter(d, camera, "up", Vec3{0.f, 1.f, 0.f});
    anari::setParameter(d, camera, "height", 2.f);
    anari::setParameter(d, camera, "aspect", 1.f);
    anari::commitParameters(d, camera);
    renderer = anari::newObject<anari::Renderer>(d, subtype);
    anari::setParameter(d, renderer, "pixelSamples", spp);
    anari::setParameter(d, renderer, "checkerboarding", checkerboard);
    anari::setParameter(d, renderer, "denoise", false);
    anari::setParameter(d, renderer, "fireflyFilterMode", "none");
    anari::setParameter(d, renderer, "ambientRadiance", 1.f);
    anari::setParameter(d, renderer, "ambientSamples", 0);
    anari::setParameter(d, renderer, "sampleLimit", 0);
    anari::commitParameters(d, renderer);
  }
  ~Scene()
  {
    anari::release(d, renderer);
    anari::release(d, camera);
    anari::release(d, world);
  }
  anari::Frame newFrame()
  {
    auto f = anari::newObject<anari::Frame>(d);
    anari::setParameter(d, f, "size", UVec2{SIZE, SIZE});
    anari::setParameter(d, f, "channel.color", ANARI_FLOAT32_VEC4);
    anari::setParameter(d, f, "channel.albedo", ANARI_FLOAT32_VEC3);
    anari::setParameter(d, f, "world", world);
    anari::setParameter(d, f, "camera", camera);
    anari::setParameter(d, f, "renderer", renderer);
    // Explicit public restart avoids unrelated deferred scene finalizations.
    anari::setParameter(d, f, "accumulationVersion", uint64_t(1));
    anari::commitParameters(d, f);
    return f;
  }
  Image render(anari::Frame f)
  {
    anari::render(d, f);
    anari::wait(d, f);
    Image image;
    auto a = anari::map<Vec3>(d, f, "channel.albedo");
    auto c = anari::map<Vec4>(d, f, "channel.color");
    if (!a.data || !c.data || a.width != SIZE || a.height != SIZE
        || c.width != SIZE || c.height != SIZE) {
      std::fprintf(stderr, "missing or incorrectly sized image channel\n");
      std::exit(1);
    }
    image.albedo.assign(a.data, a.data + SIZE * SIZE);
    image.color.assign(c.data, c.data + SIZE * SIZE);
    anari::unmap(d, f, "channel.albedo");
    anari::unmap(d, f, "channel.color");
    return image;
  }
};

// The expected *prefix* enumerates indices from zero directly. It deliberately
// does not call sampleOrdinal or reproduce the launch's addressing expression.
static Vec3 expected(unsigned x,
    unsigned y,
    int count,
    bool perspective = false,
    float aperture = 0.f,
    int lensMode = 0)
{
  Vec3 result{0.f, 0.f, .25f};
  // Frozen uint32 mixer answers for this 8x8 fixture, evaluated offline with
  // integer modulo-2^32 arithmetic. Do not call a production sampler to form
  // camera expectations (including its pixel offset).
  static constexpr uint32_t offsets[SIZE * SIZE] = {0xf1a85082u,
      0xa3dca20eu,
      0x1b663695u,
      0x7a812a2du,
      0x893a185cu,
      0xdbd6d9b0u,
      0x4a5b740cu,
      0xc9b47bf5u,
      0x22d29479u,
      0xb524f74eu,
      0xb8fc918du,
      0xf6568bbbu,
      0x1101ab2fu,
      0xb81bc7cbu,
      0x17cc3ccfu,
      0x0c6869beu,
      0x32d981ffu,
      0x339b28e2u,
      0x76818b25u,
      0x4e603b86u,
      0x331c4a7fu,
      0x0815fdbeu,
      0x0d79c747u,
      0x83010ff4u,
      0xe7a498bdu,
      0xbaa53ff9u,
      0x4e09abdfu,
      0x0ac5cc56u,
      0xcecf5e74u,
      0xf2e2503cu,
      0xc231861cu,
      0x31e2b933u,
      0xf9de2d0au,
      0xadfb9945u,
      0x2803a2d8u,
      0x0aad144eu,
      0x3b70d7c6u,
      0x1571c214u,
      0x77d23c2bu,
      0x7c921d9au,
      0x6b790870u,
      0x086baf46u,
      0x6df03b98u,
      0x39039429u,
      0x166335b1u,
      0x67390d72u,
      0xcb0f58eau,
      0xccf66b12u,
      0x7ab1db16u,
      0xa8ac6395u,
      0x04823722u,
      0xbd84c548u,
      0x966d1870u,
      0x9a5f2fe7u,
      0x4f726fafu,
      0x10c6afa9u,
      0x88147784u,
      0x418fb0a2u,
      0x80c7e4a8u,
      0xe135426fu,
      0x35909ff5u,
      0xcf1236f6u,
      0xeee40c54u,
      0x9ab71d6eu};
  const auto offset = offsets[y * SIZE + x];
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    const auto h = halton_reference::point(uint32_t(ordinal) + offset);
    const double sx = (x + (ordinal == 0 ? .5 : h[0])) / SIZE;
    const double sy = (y + (ordinal == 0 ? .5 : h[1])) / SIZE;
    if (!perspective) {
      result[0] += sx;
      result[1] += sy;
    } else {
      // Independent thin-lens geometry: a ray from lens L to focal-plane
      // point F intersects the depth-1 plane at F/2 + L/2. Vertex color
      // maps plane [-1,1] to [0,1]. No production camera/disk helper here.
      double u = h[2], v = h[3];
      if (lensMode == 1) {
        u = 0.;
        v = 0.;
      } // bypass lens
      if (lensMode == 2) {
        u = h[3];
        v = h[2];
      } // swap u/v
      if (lensMode == 3 && ordinal == 0) {
        u = .5;
        v = .5;
      }
      if (lensMode == 4) {
        u = h[0];
        v = h[1];
      } // wrong bases
      const double radius = aperture * std::sqrt(u);
      const double phi = 2. * std::acos(-1.) * v;
      result[0] += .5 + (sx - .5) * std::tan(std::acos(-1.) / 6.)
          + .25 * radius * std::cos(phi);
      result[1] += .5 + (sy - .5) * std::tan(std::acos(-1.) / 6.)
          + .25 * radius * std::sin(phi);
    }
  }
  result[0] /= count;
  result[1] /= count;
  return result;
}

static bool run(anari::Device d,
    const char *subtype,
    int spp,
    bool checkerboard,
    bool perspective = false,
    float aperture = 0.f)
{
  Scene scene(d, subtype, spp, checkerboard, perspective, aperture);
  // Ensure this fixture can distinguish each lens failure from the intended
  // answer, even at ordinal zero (whose jitter alone must be centered).
  if (perspective && aperture > 0.f) {
    for (int mode = 1; mode <= 4; ++mode) {
      double separation = 0.;
      for (unsigned y = 0; y < SIZE; ++y)
        for (unsigned x = 0; x < SIZE; ++x) {
          const auto a = expected(x, y, 1, true, aperture);
          const auto b = expected(x, y, 1, true, aperture, mode);
          for (int c = 0; c < 2; ++c)
            separation = std::max(separation, double(std::abs(a[c] - b[c])));
        }
      if (separation < .01) {
        std::fprintf(
            stderr, "lens fixture insensitive to failure mode %d\n", mode);
        return false;
      }
    }
  }
  auto frame = scene.newFrame();
  std::vector<Image> reference;
  // Frame's initial reset precedes its first version latch. Observe the fresh
  // image, then let the public version mechanism latch before recording the
  // progressive trace. Do not change Frame's inherited reset lifecycle here.
  Image fresh = scene.render(frame);
  bool pending = false;
  anari::getProperty(d, frame, "nextFrameReset", pending, ANARI_WAIT);
  bool ok = pending;
  const int launches = checkerboard ? 12 : 3;
  for (int run = 0; run < 3; ++run) {
    if (run == 1) {
      anari::setParameter(d, frame, "accumulationVersion", uint64_t(2));
      anari::commitParameters(d, frame);
    } else if (run == 2) {
      anari::release(d, frame);
      frame = scene.newFrame();
      fresh = scene.render(frame);
      anari::getProperty(d, frame, "nextFrameReset", pending, ANARI_WAIT);
      ok &= pending;
    }
    for (int launch = 0; launch < launches; ++launch) {
      const Image actual = scene.render(frame);
      int samples = -1;
      if (!anari::getProperty(d, frame, "numSamples", samples, ANARI_WAIT)
          || samples
              != (checkerboard ? (launch + 1) / 4 : (launch + 1) * spp)) {
        std::fprintf(stderr,
            "%s spp=%d checker=%d launch=%d: wrong sample count %d\n",
            subtype,
            spp,
            checkerboard,
            launch,
            samples);
        ok = false;
      }
      for (unsigned y = 0; y < SIZE; ++y) {
        for (unsigned x = 0; x < SIZE; ++x) {
          int count = (launch + 1) * spp;
          const int divisor = checkerboard ? launch / 4 + 1 : launch * spp + 1;
          if (checkerboard) {
            const int tile = (x & 1) | ((y & 1) << 1);
            count = launch / 4 + int(tile <= launch % 4);
          }
          // The mapped albedo channel retains its existing pre-launch
          // (completedSamples+1) divisor, including sparse checkerboard tiles.
          // Color compositing has its own normalization/initial tile redirect.
          // Preserve both; addressing must not repair counter normalization.
          Vec3 want = count ? expected(x, y, count, perspective, aperture)
                            : Vec3{0.f, 0.f, 0.f};
          for (float &v : want)
            v *= float(count) / divisor;
          for (int c = 0; c < 3; ++c) {
            const float got = actual.albedo[y * SIZE + x][c];
            // Float ray setup, trig, intersection and barycentrics versus
            // double rational/analytic oracle: absolute albedo tolerance,
            // not inappropriate CPU/GPU bit identity.
            if (!std::isfinite(got) || std::abs(got - want[c]) > 2e-6f) {
              std::fprintf(stderr,
                  "%s spp=%d checker=%d run=%d launch=%d pixel=(%u,%u) channel=%d: %.9g != %.9g\n",
                  subtype,
                  spp,
                  checkerboard,
                  run,
                  launch,
                  x,
                  y,
                  c,
                  got,
                  want[c]);
              anari::release(d, frame);
              return false;
            }
          }
        }
      }
      if (launch == 0 && run != 1 && !(fresh == actual)) {
        std::fprintf(stderr,
            "%s fresh first launch differs from reset ordinal zero\n",
            subtype);
        ok = false;
      }
      if (run == 0)
        reference.push_back(actual);
      else if (!(actual == reference[launch])) {
        std::fprintf(stderr,
            "%s spp=%d checker=%d run=%d launch=%d not repeatable\n",
            subtype,
            spp,
            checkerboard,
            run,
            launch);
        ok = false;
      }
    }
  }
  anari::release(d, frame);
  std::printf("%s spp=%d checker=%d perspective=%d aperture=%g: %s\n",
      subtype,
      spp,
      checkerboard,
      perspective,
      aperture,
      ok ? "ok" : "FAIL");
  return ok;
}

int main()
{
  auto device = makeVisRTXDevice(status);
  bool ok = true;
  for (const char *subtype : {"quality", "interactive"}) {
    for (int spp : {1, 2, 3, 4, 8})
      ok &= run(device, subtype, spp, false);
    // Requested spp=8 must be forced to effective spp=1 by the real renderer.
    ok &= run(device, subtype, 8, true);
    for (float aperture : {0.f, .4f}) {
      for (int spp : {1, 3})
        ok &= run(device, subtype, spp, false, true, aperture);
      ok &= run(device, subtype, 8, true, true, aperture);
    }
  }
  anari::release(device, device);
  return ok ? 0 : 1;
}

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

// Transcode-outage recovery, three assertions in one flow:
// 1. Push-actually-fires: switching to a BROKEN root (passes the
//    libraries/mdl probe, transcode fails — no nodedefs) visibly degrades an
//    UNTOUCHED committed material to the fallback. A silently-dead push would
//    keep rendering the old compiled material.
// 2. Failure-branch freeze: the failed retranscode must not tear down the
//    routed args holding the user's constant red.
// 3. Outage-window staging: a sampler bound WHILE broken must survive (routing
//    is deferred until a material exists to route into) and take effect on
//    recovery.

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <array>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>

using vec2 = std::array<float, 2>;
using vec3 = std::array<float, 3>;
using uvec3 = std::array<unsigned int, 3>;
using uvec2 = std::array<unsigned int, 2>;

static void statusFunc(const void *, ANARIDevice, ANARIObject, ANARIDataType,
    ANARIStatusSeverity s, ANARIStatusCode, const char *m)
{ if (s <= ANARI_SEVERITY_WARNING) std::fprintf(stderr, "[anari] %s\n", m); }

static anari::Sampler makeGreenSampler(anari::Device d)
{
  auto img = anari::newArray2D(d, ANARI_FLOAT32_VEC3, 1, 1);
  *anari::map<vec3>(d, img) = vec3{0.f, 1.f, 0.f};
  anari::unmap(d, img);
  auto s = anari::newObject<anari::Sampler>(d, "image2D");
  anari::setAndReleaseParameter(d, s, "image", img);
  anari::setParameter(d, s, "inAttribute", "attribute0");
  anari::setParameter(d, s, "wrapMode1", "clampToEdge");
  anari::setParameter(d, s, "wrapMode2", "clampToEdge");
  anari::commitParameters(d, s);
  return s;
}

static vec3 centerPixel(anari::Device d, anari::Frame frame, uvec2 size)
{
  anari::render(d, frame); anari::wait(d, frame);
  auto fb = anari::map<float>(d, frame, "channel.color");
  const float *px = fb.data + 4 * (size[0] * (size[1] / 2) + size[0] / 2);
  vec3 c{px[0], px[1], px[2]};
  anari::unmap(d, frame, "channel.color");
  return c;
}
static bool isRed(vec3 c)
{ return c[0] > 0.3f && c[0] > c[1] + 0.1f && c[0] > c[2] + 0.1f; }
static bool isGreen(vec3 c)
{ return c[1] > 0.3f && c[1] > c[0] + 0.1f && c[1] > c[2] + 0.1f; }

static const char *kStandardSurfaceDoc = R"(<?xml version="1.0"?>
<materialx version="1.39">
  <standard_surface name="surface" type="surfaceshader" />
  <surfacematerial name="StandardSurface" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";

namespace fs = std::filesystem;

// A root whose libraries/ holds ONLY the mdl implementation modules: passes
// the chain's libraries/mdl probe, but stdlib nodedef loading finds nothing,
// so every transcode against it fails.
static fs::path makeBrokenRoot()
{
  const fs::path realLibraries(MATERIALX_LIBRARIES_DIR);
  auto root = fs::temp_directory_path()
      / ("visrtx-mtlx-outage-broken-" + std::to_string(getpid()));
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root / "libraries");
  fs::create_directory_symlink(realLibraries / "mdl", root / "libraries" / "mdl", ec);
  if (ec) {
    std::printf("FAIL: cannot create mdl symlink: %s\n", ec.message().c_str());
    std::exit(1);
  }
  return root;
}

static fs::path makeGoodRoot()
{
  const fs::path realLibraries(MATERIALX_LIBRARIES_DIR);
  auto root = fs::temp_directory_path()
      / ("visrtx-mtlx-outage-good-" + std::to_string(getpid()));
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root);
  fs::create_directory_symlink(realLibraries, root / "libraries", ec);
  if (ec) {
    std::printf("FAIL: cannot create libraries symlink: %s\n", ec.message().c_str());
    std::exit(1);
  }
  return root;
}

static void setRoot(anari::Device d, const fs::path &root)
{
  anari::setParameter(d, d, "materialxSearchPaths", root.string());
  anari::commitParameters(d, d);
}

int main()
{
  const auto rootGood = makeGoodRoot();
  const auto rootBroken = makeBrokenRoot();

  auto d = anari::Device(makeVisRTXDevice(statusFunc));
  anari::setParameter(d, d, "forceInit", true);
  setRoot(d, rootGood);

  auto mat = anari::newObject<anari::Material>(d, "materialx");
  anari::setParameter(d, mat, "sourceType", std::string("documentInline"));
  anari::setParameter(d, mat, "source", std::string(kStandardSurfaceDoc));
  anari::setParameter(d, mat, "materialName", std::string("StandardSurface"));
  anari::setParameter(d, mat, "base_color", vec3{1.f, 0.f, 0.f}); // constant red
  anari::commitParameters(d, mat);

  std::array<vec3, 4> pos = {vec3{-1,-1,0}, {1,-1,0}, {1,1,0}, {-1,1,0}};
  std::array<vec2, 4> uv = {vec2{0,0}, {1,0}, {1,1}, {0,1}};
  std::array<uvec3, 2> idx = {uvec3{0,1,2}, uvec3{0,2,3}};
  auto geom = anari::newObject<anari::Geometry>(d, "triangle");
  anari::setParameterArray1D(d, geom, "vertex.position", pos.data(), pos.size());
  anari::setParameterArray1D(d, geom, "vertex.attribute0", uv.data(), uv.size());
  anari::setParameterArray1D(d, geom, "primitive.index", idx.data(), idx.size());
  anari::commitParameters(d, geom);

  auto surf = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surf, "geometry", geom);
  anari::setParameter(d, surf, "material", mat);
  anari::commitParameters(d, surf);

  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, world, "surface", &surf, 1);
  anari::commitParameters(d, world);
  anari::release(d, surf);

  auto cam = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, cam, "position", vec3{0, 0, 3});
  anari::setParameter(d, cam, "direction", vec3{0, 0, -1});
  anari::setParameter(d, cam, "up", vec3{0, 1, 0});
  anari::commitParameters(d, cam);

  auto rnd = anari::newObject<anari::Renderer>(d, "default");
  anari::setParameter(d, rnd, "ambientRadiance", 1.f);
  anari::commitParameters(d, rnd);

  auto frame = anari::newObject<anari::Frame>(d);
  uvec2 size = {64, 64};
  anari::setParameter(d, frame, "size", size);
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(d, frame, "world", world);
  anari::setAndReleaseParameter(d, frame, "camera", cam);
  anari::setAndReleaseParameter(d, frame, "renderer", rnd);
  anari::commitParameters(d, frame);

  bool ok = true;
  auto c = centerPixel(d, frame, size);
  if (!isRed(c)) {
    std::printf("baseline: not red (%.2f,%.2f,%.2f)\n", c[0], c[1], c[2]);
    ok = false;
  }

  // 1. Broken root: the device push must retranscode the UNTOUCHED material,
  // fail, and visibly degrade to the fallback (proves the push fires).
  setRoot(d, rootBroken);
  c = centerPixel(d, frame, size);
  if (isRed(c)) {
    std::printf("outage: still red — push never fired (%.2f,%.2f,%.2f)\n",
        c[0], c[1], c[2]);
    ok = false;
  }

  // 2. Recovery with NOTHING touched: the routed constant (only copy of the
  // user's red — the clean param was consumed on the first commit) must have
  // survived the failed retranscode.
  setRoot(d, rootGood);
  c = centerPixel(d, frame, size);
  if (!isRed(c)) {
    std::printf("value recovery: not red — routed constant lost "
                "(%.2f,%.2f,%.2f)\n", c[0], c[1], c[2]);
    ok = false;
  }

  // 3. Second outage; bind a sampler to the (previously constant) input
  // DURING it. Routing must defer, not consume-and-drop it.
  setRoot(d, rootBroken);
  auto green = makeGreenSampler(d);
  anari::setParameter(d, mat, "base_color", green);
  anari::commitParameters(d, mat);
  centerPixel(d, frame, size); // flush the outage commit; still fallback

  // 4. Recovery: the sampler staged during the outage must route and win.
  setRoot(d, rootGood);
  c = centerPixel(d, frame, size);
  if (!isGreen(c)) {
    std::printf("sampler recovery: not green — outage-staged sampler lost "
                "(%.2f,%.2f,%.2f)\n", c[0], c[1], c[2]);
    ok = false;
  }

  anari::release(d, green);
  anari::release(d, mat);
  anari::release(d, frame);
  anari::release(d, d);

  std::error_code ec;
  fs::remove_all(rootGood, ec);
  fs::remove_all(rootBroken, ec);

  if (!ok) { std::printf("FAIL\n"); return 1; }
  std::printf("PASS\n");
  return 0;
}

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

// A distribution-root change (materialxSearchPaths recommit -> generation
// bump -> retranscode) must PRESERVE a bound textured input. The clean
// sampler param was consumed by routing on the first commit, so the persisted
// textured set is the only record of the binding — a wiped set silently
// reverts the material to constants. Also exercises materialName="" (treated
// as no selection, the TSD default-constructed param).

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

// A 1x1 green image2D sampler (linear float, so no colorspace decode involved).
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

static bool centerIsGreen(anari::Device d, anari::Frame frame, uvec2 size,
    const char *label)
{
  anari::render(d, frame); anari::wait(d, frame);
  auto fb = anari::map<float>(d, frame, "channel.color");
  const float *px = fb.data + 4 * (size[0] * (size[1] / 2) + size[0] / 2);
  bool g = px[1] > 0.3f && px[1] > px[0] + 0.1f && px[1] > px[2] + 0.1f;
  if (!g)
    std::printf("%s: not green: (%.2f,%.2f,%.2f)\n", label, px[0], px[1], px[2]);
  anari::unmap(d, frame, "channel.color");
  return g;
}

// An application-authored instantiation (ADR 0008): the standard_surface
// nodedef comes from the MaterialX distribution the device resolves at runtime.
static const char *kStandardSurfaceDoc = R"(<?xml version="1.0"?>
<materialx version="1.39">
  <standard_surface name="surface" type="surfaceshader" />
  <surfacematerial name="StandardSurface" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";

namespace fs = std::filesystem;

static fs::path makeSymlinkRoot(const char *tag)
{
  const fs::path realLibraries(MATERIALX_LIBRARIES_DIR);
  auto root = fs::temp_directory_path()
      / ("visrtx-mtlx-rootchange-" + std::string(tag) + "-"
          + std::to_string(getpid()));
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root);
  fs::create_directory_symlink(realLibraries, root / "libraries", ec);
  if (ec) {
    std::printf("FAIL: cannot create libraries symlink: %s\n",
        ec.message().c_str());
    std::exit(1);
  }
  return root;
}

int main()
{
  const auto rootA = makeSymlinkRoot("a");
  const auto rootB = makeSymlinkRoot("b");

  auto d = anari::Device(makeVisRTXDevice(statusFunc));
  anari::setParameter(d, d, "materialxSearchPaths", rootA.string());
  anari::setParameter(d, d, "forceInit", true);
  anari::commitParameters(d, d);

  auto mat = anari::newObject<anari::Material>(d, "materialx");
  anari::setParameter(d, mat, "sourceType", std::string("documentInline"));
  anari::setParameter(d, mat, "source", std::string(kStandardSurfaceDoc));
  // "" = no selection (the TSD default-constructed param); the single
  // material in the document is picked.
  anari::setParameter(d, mat, "materialName", std::string(""));
  auto green = makeGreenSampler(d);
  anari::setParameter(d, mat, "base_color", green); // sampler on a clean input
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

  bool ok = centerIsGreen(d, frame, size, "before root change");

  // Switch the distribution root: generation bump. The material is NOT
  // touched — the device pushes committed materialx materials back through
  // the commit buffer itself, and the retranscode must preserve the bound
  // sampler (the clean param was consumed by routing on the first commit).
  anari::setParameter(d, d, "materialxSearchPaths", rootB.string());
  anari::commitParameters(d, d);

  char resolvedRoot[1024] = {};
  anariGetProperty(d, d, "materialx.distributionRoot", ANARI_STRING,
      resolvedRoot, sizeof(resolvedRoot), ANARI_WAIT);
  if (std::string(resolvedRoot) != rootB.string()) {
    std::printf("FAIL: distributionRoot '%s' != new root '%s'\n",
        resolvedRoot, rootB.string().c_str());
    ok = false;
  }

  ok = centerIsGreen(d, frame, size, "after root change") && ok;

  anari::release(d, green);
  anari::release(d, mat);
  anari::release(d, frame);
  anari::release(d, d);

  std::error_code ec;
  fs::remove_all(rootA, ec);
  fs::remove_all(rootB, ec);

  if (!ok) { std::printf("FAIL\n"); return 1; }
  std::printf("PASS\n");
  return 0;
}

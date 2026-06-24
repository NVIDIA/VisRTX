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

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <array>
#include <cstdio>
#include <string>

// std::array<>-based vec types; ext/std.h provides the ANARITypeFor mappings.
// (Do NOT use anari::math::* — that needs ext/linalg.h and its typedefs clash
// with ext/std.h.)
using vec3 = std::array<float, 3>;
using uvec3 = std::array<unsigned int, 3>;
using uvec2 = std::array<unsigned int, 2>;

// Surface MDL compile/distill errors (otherwise silent) to make the Step 5
// debug loop actionable — mirrors TestSpheres.cpp's status callback.
static void statusFunc(const void *, ANARIDevice, ANARIObject, ANARIDataType,
    ANARIStatusSeverity severity, ANARIStatusCode, const char *message)
{
  if (severity <= ANARI_SEVERITY_WARNING)
    std::fprintf(stderr, "[anari] %s\n", message);
}

// Render the current frame and report whether the center pixel is red
// (distinguishes a real MaterialX material from the grey ~0.8 diffuse fallback).
static bool centerIsRed(anari::Device d, anari::Frame frame, uvec2 size)
{
  anari::render(d, frame);
  anari::wait(d, frame);
  auto fb = anari::map<float>(d, frame, "channel.color");
  const float *px = fb.data + 4 * (size[0] * (size[1] / 2) + size[0] / 2);
  const bool red = px[0] > 0.3f && px[0] > px[2] + 0.1f;
  if (!red)
    std::printf("center pixel not red: (%.2f,%.2f,%.2f)\n", px[0], px[1], px[2]);
  anari::unmap(d, frame, "channel.color");
  return red;
}

int main()
{
  auto d = anari::Device(makeVisRTXDevice(statusFunc));
  // Committing the device sets MDL search paths (including the MaterialX MDL
  // library dir); without this the inline MDL cannot import materialx::*.
  anari::setParameter(d, d, "forceInit", true);
  anari::commitParameters(d, d);
  const std::string redPath =
      std::string(MATERIALX_TEST_DATA_DIR) + "/red_surface.mtlx";

  auto mat = anari::newObject<anari::Material>(d, "materialx");
  anari::setParameter(d, mat, "source", redPath);
  anari::commitParameters(d, mat);

  // Single quad facing +Z.
  std::array<vec3, 4> pos = {vec3{-1,-1,0}, {1,-1,0}, {1,1,0}, {-1,1,0}};
  std::array<uvec3, 2> idx = {uvec3{0,1,2}, uvec3{0,2,3}};
  auto geom = anari::newObject<anari::Geometry>(d, "triangle");
  anari::setParameterArray1D(d, geom, "vertex.position", pos.data(), pos.size());
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

  const bool firstRed = centerIsRed(d, frame, size);

  // Regression guard for source-param self-poisoning: re-set `source` to the
  // SAME .mtlx path (a real setParameter re-stages the material), forcing a
  // second commit where the live `source` param is the path again — not our
  // generated MDL. With the always-reapply handoff the material stays red;
  // a regression that skipped re-application would render grey here.
  anari::setParameter(d, mat, "source", redPath);
  anari::commitParameters(d, mat);
  const bool secondRed = centerIsRed(d, frame, size);

  anari::release(d, mat);
  anari::release(d, frame);
  anari::release(d, d);

  if (!firstRed || !secondRed) {
    std::printf("FAIL: firstRed=%d secondRed=%d\n", firstRed, secondRed);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}

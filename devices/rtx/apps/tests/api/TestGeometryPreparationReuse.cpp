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

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <cuda_runtime.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
static int errors = 0, preparations = 0, builds = 0;
static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType type,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++errors;
    fprintf(stderr, "%s\n", message);
  }
  // Diagnostic counts verify that metadata publication does not silently run
  // the expensive path. Images/bounds independently verify observable results.
  if (type == ANARI_GEOMETRY && std::strstr(message, "finalizing "))
    ++preparations;
  if (std::strstr(message, "visrtx::Group building ")
      || std::strstr(message, "visrtx::World building surface BVH"))
    ++builds;
}

static bool run(ANARIDevice d, const std::string &type, bool device)
{
  auto geometry = anari::newObject<anari::Geometry>(d, type.c_str());
  std::vector<vec3> positions;
  if (type == "triangle")
    positions = {{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
  else if (type == "quad")
    positions = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
  else if (type == "sphere")
    positions = {{0, 0, 0}};
  else
    positions = {{-0.5f, 0, 0}, {0.5f, 0, 0}};
  const bool sdf = type == "sdf";
  const size_t count = sdf ? 72 : positions.size();
  const auto elementType = sdf ? ANARI_UINT8 : ANARI_FLOAT32_VEC3;
  const size_t bytes = sdf ? count : count * sizeof(vec3);
  std::vector<unsigned char> inputs(2 * bytes);
  if (sdf) {
    const float radius = 0.5f;
    std::memcpy(inputs.data() + 44, &radius, sizeof(radius));
    std::memcpy(inputs.data() + 48, &radius, sizeof(radius));
    std::memcpy(inputs.data() + bytes, inputs.data(), bytes);
    const float shift = 2.f;
    std::memcpy(inputs.data() + bytes + 20, &shift, sizeof(shift));
  } else {
    std::memcpy(inputs.data(), positions.data(), bytes);
    for (auto &v : positions)
      v[0] += 2.f;
    std::memcpy(inputs.data() + bytes, positions.data(), bytes);
  }
  void *external = nullptr;
  anari::Array1D array;
  if (device) {
    if (cudaMalloc(&external, inputs.size()) != cudaSuccess
        || cudaMemcpy(
               external, inputs.data(), inputs.size(), cudaMemcpyHostToDevice)
            != cudaSuccess)
      return false;
    array =
        anariNewArray1D(d, external, nullptr, nullptr, elementType, 2 * count);
  } else {
    array = anari::newArray1D(d, elementType, 2 * count);
    std::memcpy(anariMapArray(d, array), inputs.data(), inputs.size());
    anariUnmapArray(d, array);
  }
  anari::setParameter(d, array, "end", uint64_t(count));
  anari::commitParameters(d, array);
  const char *inputName = sdf ? "primitive.sdf" : "vertex.position";
  anari::setParameter(d, geometry, inputName, array);
  anari::setParameter(d, geometry, "radius", 0.25f);
  if (type == "cone") {
    const std::array<float, 2> radii = {0.25f, 0.25f};
    anari::setParameterArray1D(d, geometry, "vertex.radius", radii.data(), 2);
  }
  anari::setParameter(d, geometry, "attribute0", vec4{0.2f, 0.4f, 0.8f, 1});
  anari::commitParameters(d, geometry);
  auto material = anari::newObject<anari::Material>(d, "physicallyBased");
  // Area-samplable types prepare their CDF as well as bounds on first use.
  anari::setParameter(d, material, "emissive", vec3{1, 1, 1});
  anari::commitParameters(d, material);
  auto surface = anari::newObject<anari::Surface>(d);
  anari::setParameter(d, surface, "geometry", geometry);
  anari::setAndReleaseParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, world, "surface", &surface, 1);
  anari::release(d, surface);
  anari::commitParameters(d, world);
  auto camera = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, camera, "position", vec3{0, 0, -3});
  anari::setParameter(d, camera, "direction", vec3{0, 0, 1});
  anari::commitParameters(d, camera);
  auto renderer = anari::newObject<anari::Renderer>(d, "debug");
  anari::setParameter(d, renderer, "method", "geometry.attribute0");
  anari::commitParameters(d, renderer);
  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", std::array<unsigned, 2>{32, 32});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(d, frame, "world", world);
  anari::setAndReleaseParameter(d, frame, "camera", camera);
  anari::setParameter(d, frame, "renderer", renderer);
  anari::commitParameters(d, frame);
  auto render = [&]() {
    anari::render(d, frame);
    anari::wait(d, frame);
    auto fb = anari::map<vec4>(d, frame, "channel.color");
    vec3 result{};
    if (!fb.data) {
      ++errors;
      return result;
    }
    const auto pixel = fb.data[16 * 32 + 16];
    for (int i = 0; i < 3; ++i)
      result[i] = pixel[i];
    anari::unmap(d, frame, "channel.color");
    return result;
  };
  auto bounds = [&]() {
    std::array<float, 6> result{};
    if (!anariGetProperty(d,
            world,
            "bounds",
            ANARI_FLOAT32_BOX3,
            result.data(),
            sizeof(result),
            ANARI_WAIT))
      ++errors;
    return result;
  };
  auto close = [](vec3 actual, vec3 expected) {
    for (int i = 0; i < 3; ++i)
      if (!std::isfinite(actual[i])
          || std::abs(actual[i] - expected[i]) > 1e-4f)
        return false;
    return true;
  };
  // Debug output includes geometric facing modulation. Obtain its reference
  // from initial full preparation, then require identical
  // preparation-preserving output.
  const auto reference = render();
  const auto originalBounds = bounds();
  bool ok = errors == 0 && reference[0] > 0.1f && reference[2] > 0.6f;
  anari::unsetParameter(d, geometry, "attribute0");
  anari::commitParameters(d, geometry);
  ok = close(render(), {0, 0, 0}) && ok;
  for (const char *attribute :
      {"attribute0", "attribute1", "attribute2", "attribute3", "color"}) {
    const std::string method = std::string("geometry.") + attribute;
    anari::setParameter(d, renderer, "method", method.c_str());
    anari::commitParameters(d, renderer);
    const int beforePrep = preparations, beforeBuild = builds;
    anari::setParameter(d, geometry, attribute, vec4{0.2f, 0.4f, 0.8f, 1});
    anari::commitParameters(d, geometry);
    // Staging a geometric edit after commit must not leak into its snapshot.
    anari::unsetParameter(d, geometry, inputName);
    const auto actual = render();
    const bool setOk = close(actual, reference) && bounds() == originalBounds
        && preparations == beforePrep && builds == beforeBuild;
    if (!setOk)
      fprintf(stderr,
          "%s %s set pixel=%g,%g,%g prep=%d build=%d\n",
          type.c_str(),
          attribute,
          actual[0],
          actual[1],
          actual[2],
          preparations - beforePrep,
          builds - beforeBuild);
    ok = setOk && ok;
    anari::setParameter(d, geometry, inputName, array);
    anari::unsetParameter(d, geometry, attribute);
    anari::commitParameters(d, geometry);
    const auto unset = render();
    const bool unsetOk = close(unset, {0, 0, 0}) && preparations == beforePrep
        && builds == beforeBuild;
    if (!unsetOk)
      fprintf(stderr,
          "%s %s unset pixel=%g,%g,%g prep=%d build=%d\n",
          type.c_str(),
          attribute,
          unset[0],
          unset[1],
          unset[2],
          preparations - beforePrep,
          builds - beforeBuild);
    ok = unsetOk && ok;
  }
  {
    // Scalar/string parameters are part of the preparation identity too.
    const int beforePrep = preparations, beforeBuild = builds;
    const bool round =
        type == "sphere" || type == "curve" || type == "cylinder";
    auto change = [&](bool changed) {
      if (round)
        anari::setParameter(d, geometry, "radius", changed ? 0.35f : 0.25f);
      else if (type == "cone")
        anari::setParameter(d, geometry, "caps", changed ? "both" : "none");
      else if (sdf)
        anari::setParameter(d, geometry, "epsilon", changed ? 1e-4f : 1e-5f);
      else
        anari::setParameter(d, geometry, "cullBackfaces", changed);
      anari::commitParameters(d, geometry);
    };
    change(true);
    const auto changedBounds = bounds();
    ok = preparations > beforePrep && builds > beforeBuild && ok;
    if (round)
      ok = changedBounds[0] < originalBounds[0] - 0.05f
          && changedBounds[3] > originalBounds[3] + 0.05f && ok;
    change(false);
    const auto restored = bounds();
    for (int i = 0; i < 6; ++i)
      ok = std::isfinite(restored[i])
          && std::abs(restored[i] - originalBounds[i]) < 1e-4f && ok;
  }
  if (!device) {
    // An in-place unmap changes values without changing the handle or GPU
    // address. Coalescing it with a geometry-wide attribute commit must still
    // prepare.
    const int beforePrep = preparations, beforeBuild = builds;
    auto *mapped = static_cast<unsigned char *>(anariMapArray(d, array));
    if (sdf) {
      const float shift = 0.25f;
      std::memcpy(mapped + 20, &shift, sizeof(shift));
    } else {
      for (size_t i = 0; i < count; ++i) {
        vec3 position;
        std::memcpy(position.data(), mapped + i * sizeof(vec3), sizeof(vec3));
        position[0] += 0.25f;
        std::memcpy(mapped + i * sizeof(vec3), position.data(), sizeof(vec3));
      }
    }
    anariUnmapArray(d, array);
    anari::setParameter(d, geometry, "attribute0", vec4{0.1f, 0.2f, 0.3f, 1});
    anari::commitParameters(d, geometry);
    const auto moved = bounds();
    ok = preparations > beforePrep && builds > beforeBuild && ok;
    for (int i = 0; i < 6; ++i)
      ok = std::isfinite(moved[i])
          && std::abs(moved[i] - originalBounds[i] - (i % 3 == 0 ? 0.25f : 0.f))
              < 1e-4f
          && ok;
    std::memcpy(anariMapArray(d, array), inputs.data(), inputs.size());
    anariUnmapArray(d, array);
    anari::setParameter(d, geometry, "attribute0", vec4{0, 0, 0, 1});
    anari::commitParameters(d, geometry);
    ok = bounds() == originalBounds && ok;
  }
  // Coalesced range and geometry-wide attribute commits must prepare the new
  // logical input.
  int beforePrep = preparations, beforeBuild = builds;
  anari::setParameter(d, array, "begin", uint64_t(count));
  anari::setParameter(d, array, "end", uint64_t(2 * count));
  anari::commitParameters(d, array);
  anari::setParameter(d, geometry, "attribute0", vec4{1, 0, 0, 1});
  anari::commitParameters(d, geometry);
  const auto shifted = bounds();
  ok = preparations > beforePrep && builds > beforeBuild && ok;
  for (int i = 0; i < 6; ++i)
    ok = std::isfinite(shifted[i])
        && std::abs(shifted[i] - originalBounds[i] - (i % 3 == 0 ? 2.f : 0.f))
            < 1e-4f
        && ok;
  // Last-public release changes backing pointers, but not values or ranges.
  // Coalesce it with a geometry-wide attribute commit to exercise pointer-only
  // invalidation.
  beforePrep = preparations;
  beforeBuild = builds;
  anari::release(d, array);
  if (external)
    cudaFree(external);
  anari::setParameter(d, geometry, "attribute0", vec4{0, 1, 0, 1});
  anari::commitParameters(d, geometry);
  ok = bounds() == shifted && ok;
  if (device)
    ok = preparations > beforePrep && builds > beforeBuild && ok;
  render();
  ok = errors == 0 && ok;
  printf("preparation-preserving updates %s device=%d %s\n",
      type.c_str(),
      device,
      ok ? "PASS" : "FAIL");
  anari::release(d, frame);
  anari::release(d, renderer);
  anari::release(d, world);
  anari::release(d, geometry);
  return ok;
}

int main()
{
  auto d = makeVisRTXDevice(status);
  bool ok = true;
  for (const char *type :
      {"triangle", "quad", "sphere", "curve", "cone", "cylinder", "sdf"})
    for (bool device : {false, true})
      ok = run(d, type, device) && ok;
  anari::release(d, d);
  return ok ? 0 : 1;
}

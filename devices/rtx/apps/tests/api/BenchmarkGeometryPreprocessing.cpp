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

// Manual evidence tool, deliberately not a wall-clock CTest assertion.
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include "DeviceTestArray.h"

using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
static int errors = 0;
static std::atomic<int> completions{0};
static void complete(const void *, ANARIDevice, ANARIFrame)
{
  ++completions;
}
static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++errors;
    fprintf(stderr, "%s\n", message);
  }
}
int main(int argc, char **argv)
{
  if (argc < 5 || argc > 10) {
    fprintf(stderr,
        "usage: %s triangle|sphere|cone|cylinder|quad|curve|sdf count device repeats [shared-surfaces] [distinct-geometries] [recommit|replace|unchanged] [frames:0=bounds] [emissive:0|1]\n",
        argv[0]);
    return 2;
  }
  const std::string type = argv[1];
  const size_t n = std::stoull(argv[2]);
  const bool device = std::atoi(argv[3]) != 0;
  const int repeats = std::atoi(argv[4]);
  const int sharedSurfaces = argc >= 6 ? std::atoi(argv[5]) : 1;
  const int distinctGeometries = argc >= 7 ? std::atoi(argv[6]) : 1;
  const std::string mode = argc >= 8 ? argv[7] : "recommit";
  const int numFrames = argc >= 9 ? std::atoi(argv[8]) : 0;
  const bool emissive = argc >= 10 ? std::atoi(argv[9]) != 0 : true;
  if (!n || repeats < 2 || sharedSurfaces < 1 || distinctGeometries < 1
      || numFrames < 0
      || (mode != "recommit" && mode != "replace" && mode != "unchanged")
      || (type != "triangle" && type != "sphere" && type != "cone"
          && type != "cylinder" && type != "quad" && type != "curve"
          && type != "sdf"))
    return 2;
  auto d = makeVisRTXDevice(status);
  const auto setupStart = std::chrono::steady_clock::now();
  std::vector<anari::Geometry> geometries;
  std::vector<vec3> positions;
  std::vector<float> radii;
  std::vector<unsigned> sphereIndices;
  std::vector<std::array<unsigned, 2>> tubeIndices;
  std::vector<std::array<unsigned, 3>> triIndices;
  std::vector<std::array<unsigned, 4>> quadIndices;
  std::vector<uint8_t> sdf;
  for (size_t i = 0; i < n; ++i) {
    const float x = float(i % 256) * 0.04f, y = float(i / 256) * 0.04f;
    const unsigned base = unsigned(positions.size());
    if (type == "triangle" || type == "quad") {
      positions.push_back({x, y, 0});
      positions.push_back({x + 0.02f, y, 0});
      positions.push_back({x + 0.02f, y + 0.02f, 0});
      if (type == "quad") {
        positions.push_back({x, y + 0.02f, 0});
        quadIndices.push_back({base, base + 1, base + 2, base + 3});
      } else
        triIndices.push_back({base, base + 1, base + 2});
    } else if (type == "sphere") {
      positions.push_back({x, y, 0});
      radii.push_back(0.01f);
      sphereIndices.push_back(base);
    } else if (type == "sdf") {
      const size_t offset = sdf.size();
      sdf.resize(offset + 72);
      const vec3 center = {x, y, 0};
      const float radius = 0.01f;
      std::memcpy(sdf.data() + offset + 20, center.data(), sizeof(center));
      std::memcpy(sdf.data() + offset + 44, &radius, sizeof(radius));
      std::memcpy(sdf.data() + offset + 48, &radius, sizeof(radius));
    } else {
      positions.push_back({x, y, 0});
      positions.push_back({x + 0.02f, y, 0});
      tubeIndices.push_back({base, base + 1});
      radii.push_back(0.01f);
      if (type != "cylinder")
        radii.push_back(0.005f);
    }
  }
  auto setInputs = [&](anari::Geometry geometry) {
    if (type == "sdf")
      setTestArray(d, geometry, "primitive.sdf", sdf, device, 0);
    else
      setTestArray(d, geometry, "vertex.position", positions, device, 0);
    if (type == "triangle")
      setTestArray(d, geometry, "primitive.index", triIndices, device, 0);
    if (type == "quad")
      setTestArray(d, geometry, "primitive.index", quadIndices, device, 0);
    if (type == "sphere")
      setTestArray(d, geometry, "primitive.index", sphereIndices, device, 0);
    if (type == "cone" || type == "cylinder")
      setTestArray(d, geometry, "primitive.index", tubeIndices, device, 0);
    if (type != "curve" && !radii.empty())
      setTestArray(d,
          geometry,
          type == "cylinder" ? "primitive.radius" : "vertex.radius",
          radii,
          device,
          0);
    if (type == "curve")
      anari::setParameter(d, geometry, "radius", 0.01f);
    anari::commitParameters(d, geometry);
  };
  for (int i = 0; i < distinctGeometries; ++i) {
    auto geometry = anari::newObject<anari::Geometry>(d, type.c_str());
    setInputs(geometry);
    geometries.push_back(geometry);
  }
  auto material = anari::newObject<anari::Material>(d, "physicallyBased");
  anari::setParameter(
      d, material, "emissive", emissive ? vec3{1, 1, 1} : vec3{0, 0, 0});
  anari::commitParameters(d, material);
  std::vector<anari::Surface> surfaces;
  for (auto geometry : geometries)
    for (int i = 0; i < sharedSurfaces; ++i) {
      auto surface = anari::newObject<anari::Surface>(d);
      anari::setParameter(d, surface, "geometry", geometry);
      anari::setParameter(d, surface, "material", material);
      anari::commitParameters(d, surface);
      surfaces.push_back(surface);
    }
  anari::release(d, material);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(
      d, world, "surface", surfaces.data(), surfaces.size());
  for (auto surface : surfaces)
    anari::release(d, surface);
  anari::commitParameters(d, world);
  std::vector<anari::Frame> frames;
  for (int i = 0; i < numFrames; ++i) {
    auto camera = anari::newObject<anari::Camera>(d, "perspective");
    anari::setParameter(d, camera, "position", vec3{0, 0, -3});
    anari::setParameter(d, camera, "direction", vec3{0, 0, 1});
    anari::commitParameters(d, camera);
    auto renderer = anari::newObject<anari::Renderer>(d, "interactive");
    anari::setParameter(d, renderer, "sampleLimit", 0);
    anari::setParameter(d, renderer, "pixelSamples", 1);
    anari::commitParameters(d, renderer);
    auto frame = anari::newObject<anari::Frame>(d);
    anari::setParameter(d, frame, "size", std::array<unsigned, 2>{64, 64});
    anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
    anari::setParameter(d, frame, "world", world);
    ANARIFrameCompletionCallback callback = complete;
    anari::setParameter(d,
        frame,
        "frameCompletionCallback",
        ANARI_FRAME_COMPLETION_CALLBACK,
        &callback);
    anari::setAndReleaseParameter(d, frame, "camera", camera);
    anari::setAndReleaseParameter(d, frame, "renderer", renderer);
    anari::commitParameters(d, frame);
    frames.push_back(frame);
  }
  // Bounds-only measures GPU-ready acceleration structures; render mode also
  // includes registry publication and completed tracing. No pre-query in render
  // mode: it would move update work outside the submission measurement.
  double warm = 0;
  for (int iteration = 0; iteration < repeats; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    if (iteration > 0) {
      for (auto geometry : geometries) {
        if (mode == "replace")
          setInputs(geometry); // Include allocation, staging, and H2D costs.
        else if (mode == "recommit") {
          anari::setParameter(
              d, geometry, "attribute0", vec4{float(iteration), 0, 0, 0});
          anari::commitParameters(d, geometry);
        }
      }
    }
    for (auto frame : frames)
      anari::render(d, frame);
    const auto submitted = std::chrono::steady_clock::now();
    for (auto frame : frames)
      anari::wait(d, frame);
    std::array<float, 6> bounds{};
    const bool ok = anariGetProperty(d,
        world,
        "bounds",
        ANARI_FLOAT32_BOX3,
        bounds.data(),
        sizeof(bounds),
        ANARI_WAIT);
    const auto completed = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(completed - start).count();
    if (!ok || errors || completions != (iteration + 1) * numFrames)
      return 1;
    // Validation is outside the timed interval. Never accept a skipped frame
    // or non-finite output as a faster completed render.
    for (auto frame : frames) {
      int samples = 0;
      if (!anari::getProperty(d, frame, "numSamples", samples) || samples < 1) {
        fprintf(
            stderr, "unexpected samples=%d iteration=%d\n", samples, iteration);
        return 1;
      }
      auto pixels = anari::map<vec4>(d, frame, "channel.color");
      if (!pixels.data)
        return 1;
      for (size_t i = 0; i < size_t(pixels.width) * pixels.height; ++i)
        for (float value : pixels.data[i])
          if (!std::isfinite(value))
            return 1;
      anari::unmap(d, frame, "channel.color");
    }
    for (float value : bounds)
      if (!std::isfinite(value))
        return 1;
    printf("sample iteration=%d ms=%.6f submit_ms=%.6f\n",
        iteration,
        ms,
        std::chrono::duration<double, std::milli>(submitted - start).count());
    if (iteration == 0) {
      printf("first_ms=%.6f setup_to_ready_ms=%.6f\n",
          ms,
          std::chrono::duration<double, std::milli>(completed - setupStart)
              .count());
    } else
      warm += ms;
  }
  printf("mode=%s geometries=%d shared_surfaces=%d frames=%d emissive=%d\n",
      mode.c_str(),
      distinctGeometries,
      sharedSurfaces,
      numFrames,
      emissive);
  printf("type=%s count=%zu device=%d warm_rebuild_ms=%.6f repeats=%d\n",
      type.c_str(),
      n,
      device,
      warm / (repeats - 1),
      repeats);
  for (auto frame : frames)
    anari::release(d, frame);
  anari::release(d, world);
  for (auto geometry : geometries)
    anari::release(d, geometry);
  anari::release(d, d);
  return 0;
}

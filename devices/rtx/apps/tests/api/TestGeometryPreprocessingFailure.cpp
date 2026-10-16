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
#include <anari/anari_cpp.hpp>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include "DeviceTestArray.h"

using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
static int errors = 0;
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
static std::atomic<int> completions{0};
static void complete(const void *, ANARIDevice, ANARIFrame)
{
  ++completions;
}

// Numeric failure is recoverable without poisoning CUDA. Test first preparation
// and failure after a good generation, via public samples/callback/bounds APIs.
int main(int argc, char **argv)
{
  const bool boundsFailure = argc > 1;
  auto d = makeVisRTXDevice(status);
  auto geometry = anari::newObject<anari::Geometry>(
      d, boundsFailure ? "cylinder" : "sphere");
  const std::array<vec3, 2> positions = {{{-0.25f, 0, 0}, {0.25f, 0, 0}}};
  setTestArray(d, geometry, "vertex.position", positions, true);
  auto material = anari::newObject<anari::Material>(d, "physicallyBased");
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
  auto renderer = anari::newObject<anari::Renderer>(d, "interactive");
  anari::commitParameters(d, renderer);
  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", std::array<unsigned, 2>{32, 32});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(d, frame, "camera", camera);
  anari::setAndReleaseParameter(d, frame, "renderer", renderer);
  anari::setParameter(d, frame, "world", world);
  ANARIFrameCompletionCallback callback = complete;
  anari::setParameter(d,
      frame,
      "frameCompletionCallback",
      ANARI_FRAME_COMPLETION_CALLBACK,
      &callback);
  anari::commitParameters(d, frame);
  bool ok = true;
  int previousSamples = 0;
  for (int step = 0; step < 4; ++step) {
    const bool invalid = step % 2 == 0;
    const std::array<float, 2> radii = {invalid
            ? (boundsFailure ? std::numeric_limits<float>::infinity() : 1e20f)
            : 0.25f,
        0.125f};
    setTestArray(d,
        geometry,
        boundsFailure ? "primitive.radius" : "vertex.radius",
        radii,
        true);
    anari::commitParameters(d, geometry);
    const int beforeErrors = errors;
    const int beforeCompletions = completions;
    anari::render(d, frame);
    anari::wait(d, frame);
    int samples = -1;
    ok = anari::getProperty(d, frame, "numSamples", samples) && ok;
    std::array<float, 6> bounds{};
    const bool haveBounds = anariGetProperty(d,
        world,
        "bounds",
        ANARI_FLOAT32_BOX3,
        bounds.data(),
        sizeof(bounds),
        ANARI_WAIT);
    if (invalid)
      ok = ok && errors > beforeErrors && samples == previousSamples
          && completions == beforeCompletions && !haveBounds;
    else {
      ok = ok && errors == beforeErrors && samples > 0
          && completions == beforeCompletions + 1 && haveBounds;
      previousSamples = samples;
    }
    printf("%s step=%d samples=%d callbacks=%d errors=%d bounds=%d %s\n",
        boundsFailure ? "bounds" : "area",
        step,
        samples,
        int(completions),
        errors,
        haveBounds,
        ok ? "PASS" : "FAIL");
  }
  anari::release(d, frame);
  anari::release(d, world);
  anari::release(d, geometry);
  anari::release(d, d);
  return ok ? 0 : 1;
}

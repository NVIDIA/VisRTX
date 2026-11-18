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

// Recoverable publication errors must not consume a checkerboard tile. ELF
// interposition requires the shared CUDA runtime, just like preparation tests.
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <anari/anari_cpp.hpp>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
bool failPublication = false;
int injections = 0;
int errors = 0;
int publicationErrors = 0;
std::atomic<int> completions{0};

void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++errors;
    if (std::strstr(message, "GPU publication failure"))
      ++publicationErrors;
    std::fprintf(stderr, "%s\n", message);
  }
}

void complete(const void *, ANARIDevice, ANARIFrame)
{
  ++completions;
}

bool run(bool checkerboard, bool reset)
{
  const int initialErrors = errors;
  auto d = makeVisRTXDevice(status);
  auto geometry = anari::newObject<anari::Geometry>(d, "quad");
  const std::array<vec3, 4> vertices = {
      vec3{-10, -10, 0}, vec3{10, -10, 0}, vec3{10, 10, 0}, vec3{-10, 10, 0}};
  anari::setParameterArray1D(
      d, geometry, "vertex.position", vertices.data(), vertices.size());
  anari::setParameter(d, geometry, "color", vec4{0.8f, 0.4f, 0.2f, 1.f});
  anari::commitParameters(d, geometry);
  auto material = anari::newObject<anari::Material>(d, "matte");
  anari::commitParameters(d, material);
  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geometry);
  anari::setAndReleaseParameter(d, surface, "material", material);
  anari::commitParameters(d, surface);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, world, "surface", &surface, 1);
  anari::release(d, surface);
  anari::commitParameters(d, world);
  auto camera = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, camera, "position", vec3{0, 0, -3});
  anari::setParameter(d, camera, "direction", vec3{0, 0, 1});
  anari::commitParameters(d, camera);
  // Debug is deterministic and has a finite sample limit of one. Orthographic
  // rays make facing modulation constant, so every tile must match exactly.
  auto renderer = anari::newObject<anari::Renderer>(d, "debug");
  anari::setParameter(d, renderer, "method", "geometry.color");
  anari::setParameter(d, renderer, "checkerboarding", checkerboard);
  anari::commitParameters(d, renderer);
  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", std::array<unsigned, 2>{8, 8});
  anari::setParameter(d, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setAndReleaseParameter(d, frame, "world", world);
  anari::setAndReleaseParameter(d, frame, "camera", camera);
  anari::setAndReleaseParameter(d, frame, "renderer", renderer);
  anari::setParameter(d, frame, "accumulationVersion", uint64_t(1));
  ANARIFrameCompletionCallback callback = complete;
  anari::setParameter(d,
      frame,
      "frameCompletionCallback",
      ANARI_FRAME_COMPLETION_CALLBACK,
      &callback);
  anari::commitParameters(d, frame);
  auto render = [&] {
    anari::render(d, frame);
    anari::wait(d, frame);
  };
  auto samples = [&] {
    int n = -1;
    anariGetProperty(
        d, frame, "numSamples", ANARI_INT32, &n, sizeof(n), ANARI_WAIT);
    return n;
  };
  auto pixels = [&] {
    auto fb = anari::map<vec4>(d, frame, "channel.color");
    std::vector<vec4> result;
    if (fb.data)
      result.assign(fb.data, fb.data + 64);
    anari::unmap(d, frame, "channel.color");
    return result;
  };
  const int tiles = checkerboard ? 4 : 1;
  // Resolve the frame's initial reset before the manual-version reset, then
  // obtain a complete reference cycle (extra calls stop at the sample limit).
  render();
  for (int i = 0; i < tiles; ++i)
    render();
  const auto reference = pixels();
  bool ok = samples() == 1 && reference.size() == 64 && errors == initialErrors;
  for (const auto &pixel : reference)
    ok = pixel[0] > 0.7f && pixel[1] > 0.3f && ok;
  if (!ok)
    std::fprintf(stderr,
        "reference failed: samples=%d pixels=%zu errors=%d\n",
        samples(),
        reference.size(),
        errors - initialErrors);

  anari::setParameter(d, frame, "accumulationVersion", uint64_t(2));
  anari::commitParameters(d, frame);
  if (!reset)
    render(); // Start a checkerboard cycle, then reject its second tile.
  const int before = completions;
  const int beforeInjections = injections;
  const int beforePublicationErrors = publicationErrors;
  failPublication = true;
  render();
  ok = injections == beforeInjections + 1
      && publicationErrors == beforePublicationErrors + 1
      && errors == initialErrors + 1 && completions == before && samples() == 0
      && ok;
  // No recommit: recovery must restart every tile, not skip the rejected one.
  for (int i = 0; i < tiles; ++i) {
    render();
    const bool stepOk =
        completions == before + i + 1 && samples() == (i + 1 == tiles ? 1 : 0);
    if (!stepOk)
      std::fprintf(stderr,
          "retry %d: completions=%d samples=%d\n",
          i,
          int(completions) - before,
          samples());
    ok = stepOk && ok;
  }
  const auto recovered = pixels();
  ok = recovered.size() == reference.size() && ok;
  if (recovered.size() == reference.size())
    for (size_t i = 0; i < reference.size(); ++i)
      for (int c = 0; c < 4; ++c)
        ok = std::isfinite(recovered[i][c])
            && std::abs(recovered[i][c] - reference[i][c]) < 1e-5f && ok;
  if (!ok)
    std::fprintf(stderr, "recovery failed before sample-limit check\n");
  render(); // The finite sample limit must suppress only this extra call.
  ok = completions == before + tiles && samples() == 1
      && errors == initialErrors + 1 && ok;
  anari::release(d, frame);
  anari::release(d, d);
  std::printf("publication recovery checkerboard=%d reset=%d %s\n",
      checkerboard,
      reset,
      ok ? "PASS" : "FAIL");
  return ok;
}
} // namespace

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
  auto real = reinterpret_cast<decltype(&cudaStreamSynchronize)>(
      dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
  if (!real)
    std::abort();
  const auto error = real(stream);
  if (!stream && failPublication) {
    failPublication = false;
    ++injections;
    return cudaErrorUnknown;
  }
  return error;
}

int main()
{
  bool ok = run(true, false);
  ok = run(true, true) && ok;
  ok = run(false, true) && ok;
  return ok ? 0 : 1;
}

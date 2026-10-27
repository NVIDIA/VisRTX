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
#include <cmath>
#include <cstdio>
#include <cstring>

static int errors = 0;
static int setups = 0;
static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (std::strstr(message, "layers created"))
    ++setups;
  if (severity == ANARI_SEVERITY_ERROR
      || severity == ANARI_SEVERITY_FATAL_ERROR) {
    ++errors;
    std::fprintf(stderr, "%s\n", message);
  }
}

int main()
{
  auto device = makeVisRTXDevice(status);
  auto world = anari::newObject<anari::World>(device);
  anari::commitParameters(device, world);
  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(
      device, camera, "position", std::array<float, 3>{0, 0, 2});
  anari::setParameter(
      device, camera, "direction", std::array<float, 3>{0, 0, -1});
  anari::setParameter(device, camera, "up", std::array<float, 3>{0, 1, 0});
  anari::commitParameters(device, camera);
  auto renderer = anari::newObject<anari::Renderer>(device, "interactive");
  anari::setParameter(device, renderer, "pixelSamples", 1);
  anari::setParameter(
      device, renderer, "background", std::array<float, 4>{.2f, .4f, .8f, 1});
  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  const char *modes[] = {"color",
      "colorAlbedo",
      "colorAlbedoNormal",
      "color",
      "colorAlbedoNormal"};
  for (unsigned configuration = 0; configuration < 5; ++configuration) {
    std::array<unsigned, 2> size = {
        37 + configuration * 13, 29 + configuration * 7};
    anari::setParameter(device, renderer, "denoise", configuration != 3);
    anari::setParameter(device, renderer, "denoiseMode", modes[configuration]);
    anari::commitParameters(device, renderer);
    anari::setParameter(device, frame, "size", size);
    int firstSetups = -1;
    for (unsigned step = 0; step < 8; ++step) {
      anari::setParameter(
          device, frame, "mdl.animationTime", float(step) * .1f);
      anari::commitParameters(device, frame);
      anari::render(device, frame);
      anari::wait(device, frame);
      auto image = anari::map<float>(device, frame, "channel.color");
      if (!image.data || image.width != size[0] || image.height != size[1]
          || image.pixelType != ANARI_FLOAT32_VEC4)
        ++errors;
      else
        for (unsigned c = 0; c < 4; ++c)
          if (!std::isfinite(image.data[c]))
            ++errors;
      anari::unmap(device, frame, "channel.color");
      if (step == 0)
        firstSetups = setups;
      else if (setups != firstSetups) {
        std::fprintf(
            stderr, "Repeated OptiX setup for unchanged configuration\n");
        ++errors;
      }
    }
  }
  anari::release(device, frame);
  anari::release(device, renderer);
  anari::release(device, camera);
  anari::release(device, world);
  anari::release(device, device);
  std::printf(
      "Denoiser commit/reuse, guide modes, disable/enable and resize: %d errors, %d layer messages\n",
      errors,
      setups);
  return errors ? 1 : 0;
}

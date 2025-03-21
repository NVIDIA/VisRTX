/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// anari_cpp
#include <anari/anari_cpp.hpp>
// visrtx
#include <anari/ext/visrtx/visrtx.h>
// std
#include <atomic>
#include <thread>
#include <vector>

static void statusFunc(const void * /*userData*/,
    ANARIDevice /*device*/,
    ANARIObject source,
    ANARIDataType /*sourceType*/,
    ANARIStatusSeverity severity,
    ANARIStatusCode /*code*/,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL ][%p] %s\n", source, message);
    std::exit(1);
  } else {
    fprintf(stderr, "[VISRTX][%p] %s\n", source, message);
  }
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);

  std::atomic_bool threadGate = false;
  std::vector<std::thread> threads;
  threads.reserve(4);
  for (int i = 0; i < 4; i++) {
    threads.emplace_back([&]() {
      while (!threadGate)
        ;

      std::vector<anari::Object> objects;
      for (int j = 0; j < 100; j++) {
        objects.push_back(
            anari::newObject<anari::Light>(device, "directional"));
        objects.push_back(
            anari::newObject<anari::Camera>(device, "perspective"));
        objects.push_back(
            anari::newObject<anari::Geometry>(device, "triangle"));
        objects.push_back(anari::newObject<anari::World>(device));
        objects.push_back(anari::newObject<anari::Frame>(device));
      }

      for (auto &o : objects)
        anari::release(device, o);
    });
  }

  threadGate = true;

  for (auto &t : threads) {
    if (t.joinable())
      t.join();
  }

  anari::release(device, device);
  return 0;
}

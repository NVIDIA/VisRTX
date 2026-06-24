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
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/ext/visrtx/visrtx_extensions.h>
#include <cstdio>

int main()
{
  auto device = makeVisRTXDevice();
  anari::setParameter(device, device, "forceInit", true);
  anari::commitParameters(device, device);
  auto mat = anari::newObject<anari::Material>(device, "materialx");
  if (!mat) {
    std::printf("FAIL: could not create materialx material\n");
    return 1;
  }
  auto ext = visrtx::getObjectExtensions(device, ANARI_MATERIAL, "materialx");
  if (!ext.VISRTX_MATERIAL_MATERIALX) {
    std::printf("FAIL: VISRTX_MATERIAL_MATERIALX not advertised\n");
    return 1;
  }
  anari::setParameter(device, mat, "source",
      std::string(MATERIALX_TEST_DATA_DIR) + "/two_materials.mtlx");
  anari::commitParameters(device, mat);
  const char *const *names = nullptr;
  anariGetProperty(device, mat, "materialNames", ANARI_STRING_LIST,
      &names, sizeof(names), ANARI_WAIT);
  int count = 0;
  if (names) for (auto p = names; *p; ++p) ++count;
  if (count != 2) {
    std::printf("FAIL: expected 2 material names, got %d\n", count);
    return 1;
  }
  anari::release(device, mat);
  anari::release(device, device);
  std::printf("PASS\n");
  return 0;
}

/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "anari/ext/visrtx/visrtx.h"
// std
#include <cstring>
#include <string_view>

namespace visrtx {

static void fillFeatureStruct(VisRTXFeatures *features, const char *const *list)
{
  std::memset(features, 0, sizeof(VisRTXFeatures));
  for (const auto *i = list; *i != NULL; ++i) {
    std::string_view feature = *i;
    if (feature == "ANARI_VISRTX_ARRAY1D_DYNAMIC_REGION")
      features->VISRTX_ARRAY1D_DYNAMIC_REGION = 1;
    else if (feature == "ANARI_VISRTX_CUDA_OUTPUT_BUFFERS")
      features->VISRTX_CUDA_OUTPUT_BUFFERS = 1;
    else if (feature == "ANARI_VISRTX_SAMPLER_COLOR_MAP")
      features->VISRTX_SAMPLER_COLOR_MAP = 1;
    else if (feature == "ANARI_VISRTX_TRIANGLE_ATTRIBUTE_INDEXING")
      features->VISRTX_TRIANGLE_ATTRIBUTE_INDEXING = 1;
  }
}

extern "C" VISRTX_DEVICE_INTERFACE int visrtxGetObjectFeatures(
    VisRTXFeatures *features,
    ANARILibrary library,
    const char *deviceName,
    const char *objectName,
    ANARIDataType objectType)
{
  const char *const *list = (const char *const *)anariGetObjectInfo(library,
      deviceName,
      objectName,
      objectType,
      "feature",
      ANARI_STRING_LIST);
  if (list) {
    fillFeatureStruct(features, list);
    return 1;
  } else {
    return 0;
  }
}

extern "C" VISRTX_DEVICE_INTERFACE int visrtxGetInstanceFeatures(
    VisRTXFeatures *features, ANARIDevice device, ANARIObject object)
{
  const char *const *list = NULL;
  anariGetProperty(device,
      object,
      "feature",
      ANARI_STRING_LIST,
      &list,
      sizeof(list),
      ANARI_WAIT);
  if (list) {
    fillFeatureStruct(features, list);
    return 1;
  } else {
    return 0;
  }
}

VISRTX_DEVICE_INTERFACE Features getObjectFeatures(anari::Library l,
    const char *device,
    const char *objectSubtype,
    anari::DataType objectType)
{
  Features f;
  visrtxGetObjectFeatures(&f, l, device, objectSubtype, objectType);
  return f;
}

VISRTX_DEVICE_INTERFACE Features getInstanceFeatures(
    anari::Device d, anari::Object o)
{
  Features f;
  visrtxGetInstanceFeatures(&f, d, o);
  return f;
}

} // namespace visrtx

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

#pragma once

// anari
#include <anari/anari_cpp.hpp>
// std
#include <cstring>
#include <string_view>

// VisRTX extension feature testing ///////////////////////////////////////////

struct VisRTXExtensions
{
  int VISRTX_ARRAY_CUDA;
  int VISRTX_CUDA_OUTPUT_BUFFERS;
  int VISRTX_INSTANCE_ATTRIBUTES;
  int VISRTX_MATERIAL_MDL;
  int VISRTX_SPATIAL_FIELD_NANOVDB;
  int VISRTX_SPATIAL_FIELD_DATA_CENTERING;
  int VISRTX_SPATIAL_FIELD_REGION_OF_INTEREST;
  int VISRTX_SPATIAL_FIELD_STRUCTURED_RECTILINEAR;
  int VISRTX_SPATIAL_FIELD_NANOVDB_RECTILINEAR;
  int VISRTX_TRIANGLE_BACK_FACE_CULLING;
  int VISRTX_TRIANGLE_FACE_VARYING_ATTRIBUTES;
};

int visrtxGetObjectExtensions(VisRTXExtensions *extensions,
    ANARIDevice device,
    ANARIDataType objectType,
    const char *objectSubtype);

int visrtxGetInstanceExtensions(
    VisRTXExtensions *extensions, ANARIDevice device, ANARIObject object);

namespace visrtx {

void fillExtensionStruct(VisRTXExtensions *extensions, const char *const *list);

VisRTXExtensions getObjectExtensions(
    anari::Device d, anari::DataType objectType, const char *objectSubtype);

VisRTXExtensions getInstanceExtensions(anari::Device, anari::Object);

} // namespace visrtx

///////////////////////////////////////////////////////////////////////////////
// Definitions ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

inline int visrtxGetInstanceExtensions(
    VisRTXExtensions *extensions, ANARIDevice device, ANARIObject object)
{
  const char *const *list = NULL;
  anariGetProperty(device,
      object,
      "extension",
      ANARI_STRING_LIST,
      &list,
      sizeof(list),
      ANARI_WAIT);
  if (list) {
    visrtx::fillExtensionStruct(extensions, list);
    return 1;
  } else {
    return 0;
  }
}

namespace visrtx {

inline void fillExtensionStruct(
    VisRTXExtensions *extensions, const char *const *list)
{
  std::memset(extensions, 0, sizeof(VisRTXExtensions));
  for (const auto *i = list; *i != NULL; ++i) {
    std::string_view feature = *i;
    if (feature == "ANARI_VISRTX_ARRAY_CUDA")
      extensions->VISRTX_ARRAY_CUDA = 1;
    else if (feature == "ANARI_VISRTX_CUDA_OUTPUT_BUFFERS")
      extensions->VISRTX_CUDA_OUTPUT_BUFFERS = 1;
    else if (feature == "ANARI_VISRTX_INSTANCE_ATTRIBUTES")
      extensions->VISRTX_INSTANCE_ATTRIBUTES = 1;
    else if (feature == "ANARI_VISRTX_SPATIAL_FIELD_DATA_CENTERING")
      extensions->VISRTX_SPATIAL_FIELD_DATA_CENTERING = 1;
    else if (feature == "ANARI_VISRTX_SPATIAL_FIELD_REGION_OF_INTEREST")
      extensions->VISRTX_SPATIAL_FIELD_REGION_OF_INTEREST = 1;
    else if (feature == "ANARI_VISRTX_SPATIAL_FIELD_NANOVDB")
      extensions->VISRTX_SPATIAL_FIELD_NANOVDB = 1;
    else if (feature == "ANARI_VISRTX_SPATIAL_FIELD_STRUCTURED_RECTILINEAR")
      extensions->VISRTX_SPATIAL_FIELD_STRUCTURED_RECTILINEAR = 1;
    else if (feature == "ANARI_VISRTX_SPATIAL_FIELD_NANOVDB_RECTILINEAR")
      extensions->VISRTX_SPATIAL_FIELD_NANOVDB_RECTILINEAR = 1;
    else if (feature == "ANARI_VISRTX_TRIANGLE_BACK_FACE_CULLING")
      extensions->VISRTX_TRIANGLE_BACK_FACE_CULLING = 1;
    else if (feature == "ANARI_VISRTX_TRIANGLE_FACE_VARYING_ATTRIBUTES")
      extensions->VISRTX_TRIANGLE_FACE_VARYING_ATTRIBUTES = 1;
    else if (feature == "ANARI_VISRTX_MATERIAL_MDL")
#ifdef USE_MDL
      extensions->VISRTX_MATERIAL_MDL = 1;
#else
      extensions->VISRTX_MATERIAL_MDL = 0;
#endif // defined(USE_MDL)
  }
}

inline VisRTXExtensions getObjectExtensions(
    anari::Device d, anari::DataType objectType, const char *objectSubtype)
{
  VisRTXExtensions f;
  visrtxGetObjectExtensions(&f, d, objectType, objectSubtype);
  return f;
}

inline VisRTXExtensions getInstanceExtensions(anari::Device d, anari::Object o)
{
  VisRTXExtensions f;
  visrtxGetInstanceExtensions(&f, d, o);
  return f;
}

} // namespace visrtx

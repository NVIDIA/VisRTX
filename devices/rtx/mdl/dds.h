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

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cstdio>

#ifdef WIN32
#include <intrin.h>
#endif

namespace visrtx::dds {

// Following
//   https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguid
//   https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-reference
//   https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header
//   https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header-dxt10
//   https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-pixelformat

constexpr std::uint32_t operator""_cc(const char *code, std::size_t size)
{
  if (size != 4)
    throw "FourCC must be four chars long";

  return (std::uint32_t(code[0]) | (std::uint32_t(code[1]) << 8)
      | (std::uint32_t(code[2]) << 16) | (std::uint32_t(code[3]) << 24));
}

#pragma pack(push, 1)

enum DXGI_FORMAT : std::uint32_t
{
  DXGI_FORMAT_UNKNOWN = 0,
  DXGI_FORMAT_R32G32B32A32_TYPELESS = 1,
  DXGI_FORMAT_R32G32B32A32_FLOAT = 2,
  DXGI_FORMAT_R32G32B32A32_UINT = 3,
  DXGI_FORMAT_R32G32B32A32_SINT = 4,
  DXGI_FORMAT_R32G32B32_TYPELESS = 5,
  DXGI_FORMAT_R32G32B32_FLOAT = 6,
  DXGI_FORMAT_R32G32B32_UINT = 7,
  DXGI_FORMAT_R32G32B32_SINT = 8,
  DXGI_FORMAT_R16G16B16A16_TYPELESS = 9,
  DXGI_FORMAT_R16G16B16A16_FLOAT = 10,
  DXGI_FORMAT_R16G16B16A16_UNORM = 11,
  DXGI_FORMAT_R16G16B16A16_UINT = 12,
  DXGI_FORMAT_R16G16B16A16_SNORM = 13,
  DXGI_FORMAT_R16G16B16A16_SINT = 14,
  DXGI_FORMAT_R32G32_TYPELESS = 15,
  DXGI_FORMAT_R32G32_FLOAT = 16,
  DXGI_FORMAT_R32G32_UINT = 17,
  DXGI_FORMAT_R32G32_SINT = 18,
  DXGI_FORMAT_R32G8X24_TYPELESS = 19,
  DXGI_FORMAT_D32_FLOAT_S8X24_UINT = 20,
  DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS = 21,
  DXGI_FORMAT_X32_TYPELESS_G8X24_UINT = 22,
  DXGI_FORMAT_R10G10B10A2_TYPELESS = 23,
  DXGI_FORMAT_R10G10B10A2_UNORM = 24,
  DXGI_FORMAT_R10G10B10A2_UINT = 25,
  DXGI_FORMAT_R11G11B10_FLOAT = 26,
  DXGI_FORMAT_R8G8B8A8_TYPELESS = 27,
  DXGI_FORMAT_R8G8B8A8_UNORM = 28,
  DXGI_FORMAT_R8G8B8A8_UNORM_SRGB = 29,
  DXGI_FORMAT_R8G8B8A8_UINT = 30,
  DXGI_FORMAT_R8G8B8A8_SNORM = 31,
  DXGI_FORMAT_R8G8B8A8_SINT = 32,
  DXGI_FORMAT_R16G16_TYPELESS = 33,
  DXGI_FORMAT_R16G16_FLOAT = 34,
  DXGI_FORMAT_R16G16_UNORM = 35,
  DXGI_FORMAT_R16G16_UINT = 36,
  DXGI_FORMAT_R16G16_SNORM = 37,
  DXGI_FORMAT_R16G16_SINT = 38,
  DXGI_FORMAT_R32_TYPELESS = 39,
  DXGI_FORMAT_D32_FLOAT = 40,
  DXGI_FORMAT_R32_FLOAT = 41,
  DXGI_FORMAT_R32_UINT = 42,
  DXGI_FORMAT_R32_SINT = 43,
  DXGI_FORMAT_R24G8_TYPELESS = 44,
  DXGI_FORMAT_D24_UNORM_S8_UINT = 45,
  DXGI_FORMAT_R24_UNORM_X8_TYPELESS = 46,
  DXGI_FORMAT_X24_TYPELESS_G8_UINT = 47,
  DXGI_FORMAT_R8G8_TYPELESS = 48,
  DXGI_FORMAT_R8G8_UNORM = 49,
  DXGI_FORMAT_R8G8_UINT = 50,
  DXGI_FORMAT_R8G8_SNORM = 51,
  DXGI_FORMAT_R8G8_SINT = 52,
  DXGI_FORMAT_R16_TYPELESS = 53,
  DXGI_FORMAT_R16_FLOAT = 54,
  DXGI_FORMAT_D16_UNORM = 55,
  DXGI_FORMAT_R16_UNORM = 56,
  DXGI_FORMAT_R16_UINT = 57,
  DXGI_FORMAT_R16_SNORM = 58,
  DXGI_FORMAT_R16_SINT = 59,
  DXGI_FORMAT_R8_TYPELESS = 60,
  DXGI_FORMAT_R8_UNORM = 61,
  DXGI_FORMAT_R8_UINT = 62,
  DXGI_FORMAT_R8_SNORM = 63,
  DXGI_FORMAT_R8_SINT = 64,
  DXGI_FORMAT_A8_UNORM = 65,
  DXGI_FORMAT_R1_UNORM = 66,
  DXGI_FORMAT_R9G9B9E5_SHAREDEXP = 67,
  DXGI_FORMAT_R8G8_B8G8_UNORM = 68,
  DXGI_FORMAT_G8R8_G8B8_UNORM = 69,
  DXGI_FORMAT_BC1_TYPELESS = 70,
  DXGI_FORMAT_BC1_UNORM = 71,
  DXGI_FORMAT_BC1_UNORM_SRGB = 72,
  DXGI_FORMAT_BC2_TYPELESS = 73,
  DXGI_FORMAT_BC2_UNORM = 74,
  DXGI_FORMAT_BC2_UNORM_SRGB = 75,
  DXGI_FORMAT_BC3_TYPELESS = 76,
  DXGI_FORMAT_BC3_UNORM = 77,
  DXGI_FORMAT_BC3_UNORM_SRGB = 78,
  DXGI_FORMAT_BC4_TYPELESS = 79,
  DXGI_FORMAT_BC4_UNORM = 80,
  DXGI_FORMAT_BC4_SNORM = 81,
  DXGI_FORMAT_BC5_TYPELESS = 82,
  DXGI_FORMAT_BC5_UNORM = 83,
  DXGI_FORMAT_BC5_SNORM = 84,
  DXGI_FORMAT_B5G6R5_UNORM = 85,
  DXGI_FORMAT_B5G5R5A1_UNORM = 86,
  DXGI_FORMAT_B8G8R8A8_UNORM = 87,
  DXGI_FORMAT_B8G8R8X8_UNORM = 88,
  DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM = 89,
  DXGI_FORMAT_B8G8R8A8_TYPELESS = 90,
  DXGI_FORMAT_B8G8R8A8_UNORM_SRGB = 91,
  DXGI_FORMAT_B8G8R8X8_TYPELESS = 92,
  DXGI_FORMAT_B8G8R8X8_UNORM_SRGB = 93,
  DXGI_FORMAT_BC6H_TYPELESS = 94,
  DXGI_FORMAT_BC6H_UF16 = 95,
  DXGI_FORMAT_BC6H_SF16 = 96,
  DXGI_FORMAT_BC7_TYPELESS = 97,
  DXGI_FORMAT_BC7_UNORM = 98,
  DXGI_FORMAT_BC7_UNORM_SRGB = 99,
  DXGI_FORMAT_AYUV = 100,
  DXGI_FORMAT_Y410 = 101,
  DXGI_FORMAT_Y416 = 102,
  DXGI_FORMAT_NV12 = 103,
  DXGI_FORMAT_P010 = 104,
  DXGI_FORMAT_P016 = 105,
  DXGI_FORMAT_420_OPAQUE = 106,
  DXGI_FORMAT_YUY2 = 107,
  DXGI_FORMAT_Y210 = 108,
  DXGI_FORMAT_Y216 = 109,
  DXGI_FORMAT_NV11 = 110,
  DXGI_FORMAT_AI44 = 111,
  DXGI_FORMAT_IA44 = 112,
  DXGI_FORMAT_P8 = 113,
  DXGI_FORMAT_A8P8 = 114,
  DXGI_FORMAT_B4G4R4A4_UNORM = 115,
  DXGI_FORMAT_P208 = 130,
  DXGI_FORMAT_V208 = 131,
  DXGI_FORMAT_V408 = 132,
  DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE = 189,
  DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE = 190,
  DXGI_FORMAT_FORCE_UINT = 0xffffffff,
};



enum DDSCAPS : std::uint32_t
{
  DDSCAPS_COMPLEX = 0x8,
  DDSCAPS_MIPMAP = 0x400000,
  DDSCAPS_TEXTURE = 0x1000,
};

enum DDSCAPS2 : std::uint32_t
{
  DDSCAPS2_CUBEMAP = 0x200,
  DDSCAPS2_CUBEMAP_POSITIVEX = 0x400,
  DDSCAPS2_CUBEMAP_NEGATIVEX = 0x800,
  DDSCAPS2_CUBEMAP_POSITIVEY = 0x1000,
  DDSCAPS2_CUBEMAP_NEGATIVEY = 0x2000,
  DDSCAPS2_CUBEMAP_POSITIVEZ = 0x4000,
  DDSCAPS2_CUBEMAP_NEGATIVEZ = 0x8000,
  DDSCAPS2_VOLUME = 0x200000,
};

enum D3D10_RESOURCE_DIMENSION : std::uint32_t
{
  D3D10_RESOURCE_DIMENSION_TEXTURE1D = 2,
  D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3,
  D3D10_RESOURCE_DIMENSION_TEXTURE3D = 4,
};

enum D3D10_RESOURCE_MISC : std::uint32_t
{
  D3D10_RESOURCE_MISC_TEXTURE_CUBE = 4,
};

enum D3D10_RESOURCE_MISC2 : std::uint32_t
{
  D3D10_RESOURCE_MISC2_ALPHA_MODE_UNKNOWN = 0x0,
  D3D10_RESOURCE_MISC2_ALPHA_MODE_STRAIGHT = 0x1,
  D3D10_RESOURCE_MISC2_ALPHA_MODE_PREMULTIPLIED = 0x2,
  D3D10_RESOURCE_MISC2_ALPHA_MODE_OPAQUE = 0x3,
  D3D10_RESOURCE_MISC2_ALPHA_MODE_CUSTOM = 0x4,
};

enum DDPF : std::uint32_t
{
  DDPF_ALPHAPIXELS = 0x1,
  DDPF_ALPHA = 0x2,
  DDPF_FOURCC = 0x4,
  DDPF_RGB = 0x40,
  DDPF_YUV = 0x200,
  DDPF_LUMINANCE = 0x20000,
};

struct DdsPixelFormat
{
  std::uint32_t size;
  DDPF flags;
  std::uint32_t fourCC;
  std::uint32_t rgbBitCount;
  std::uint32_t rBitMask;
  std::uint32_t gBitMask;
  std::uint32_t bBitMask;
  std::uint32_t aBitMask;
};

enum DDSD_FLAGS : std::uint32_t
{
  DDSD_CAPS = 0x1,
  DDSD_HEIGHT = 0x2,
  DDSD_WIDTH = 0x4,
  DDSD_PITCH = 0x8,
  DDSD_PIXELFORMAT = 0x1000,
  DDSD_MIPMAPCOUNT = 0x20000,
  DDSD_LINEARSIZE = 0x80000,
  DDSD_DEPTH = 0x800000,
};

struct DdsHeader
{
  std::uint32_t size;
  DDSD_FLAGS flags;
  std::uint32_t height;
  std::uint32_t width;
  std::uint32_t pitchOrLinearSize;
  std::uint32_t depth;
  std::uint32_t mipMapCount;
  std::uint32_t reserved1[11];
  DdsPixelFormat pixelFormat;
  std::uint32_t caps;
  std::uint32_t caps2;
  std::uint32_t caps3;
  std::uint32_t caps4;
  std::uint32_t reserved2;
};
static_assert(sizeof(DdsHeader) == 124);

struct DdsHeaderDxt10
{
  DXGI_FORMAT dxgiFormat;
  D3D10_RESOURCE_DIMENSION resourceDimension;
  std::uint32_t miscFlag;
  std::uint32_t arraySize;
  std::uint32_t miscFlags2;
};
static_assert(sizeof(DdsHeaderDxt10) == 20);

using DdsFile = struct
{
  std::uint32_t magic; // 'DDS '

  DdsHeader header;
  DdsHeaderDxt10 header10;
};

#pragma pack(pop)

constexpr const std::uint32_t DDS_MAGIC = 0x20534444; // "DDS "

inline const void *getDataPointer(const DdsFile *dds)
{
  auto data =
      reinterpret_cast<const std::uint8_t *>(&dds->header) + sizeof(DdsHeader);
  if ((dds->header.flags & DDSD_PIXELFORMAT)
      && (dds->header.pixelFormat.fourCC == "DX10"_cc)) {
    data += sizeof(DdsHeaderDxt10);
  }

  return data;
}

inline const char* getDxgiFormatString(DXGI_FORMAT format) {
  switch (format) {
  case DXGI_FORMAT_UNKNOWN: return "DXGI_FORMAT_UNKNOWN";
  case DXGI_FORMAT_R32G32B32A32_TYPELESS: return "DXGI_FORMAT_R32G32B32A32_TYPELESS";
  case DXGI_FORMAT_R32G32B32A32_FLOAT: return "DXGI_FORMAT_R32G32B32A32_FLOAT";
  case DXGI_FORMAT_R32G32B32A32_UINT: return "DXGI_FORMAT_R32G32B32A32_UINT";
  case DXGI_FORMAT_R32G32B32A32_SINT: return "DXGI_FORMAT_R32G32B32A32_SINT";
  case DXGI_FORMAT_R32G32B32_TYPELESS: return "DXGI_FORMAT_R32G32B32_TYPELESS";
  case DXGI_FORMAT_R32G32B32_FLOAT: return "DXGI_FORMAT_R32G32B32_FLOAT";
  case DXGI_FORMAT_R32G32B32_UINT: return "DXGI_FORMAT_R32G32B32_UINT";
  case DXGI_FORMAT_R32G32B32_SINT: return "DXGI_FORMAT_R32G32B32_SINT";
  case DXGI_FORMAT_R16G16B16A16_TYPELESS: return "DXGI_FORMAT_R16G16B16A16_TYPELESS";
  case DXGI_FORMAT_R16G16B16A16_FLOAT: return "DXGI_FORMAT_R16G16B16A16_FLOAT";
  case DXGI_FORMAT_R16G16B16A16_UNORM: return "DXGI_FORMAT_R16G16B16A16_UNORM";
  case DXGI_FORMAT_R16G16B16A16_UINT: return "DXGI_FORMAT_R16G16B16A16_UINT";
  case DXGI_FORMAT_R16G16B16A16_SNORM: return "DXGI_FORMAT_R16G16B16A16_SNORM";
  case DXGI_FORMAT_R16G16B16A16_SINT: return "DXGI_FORMAT_R16G16B16A16_SINT";
  case DXGI_FORMAT_R32G32_TYPELESS: return "DXGI_FORMAT_R32G32_TYPELESS";
  case DXGI_FORMAT_R32G32_FLOAT: return "DXGI_FORMAT_R32G32_FLOAT";
  case DXGI_FORMAT_R32G32_UINT: return "DXGI_FORMAT_R32G32_UINT";
  case DXGI_FORMAT_R32G32_SINT: return "DXGI_FORMAT_R32G32_SINT";
  case DXGI_FORMAT_R32G8X24_TYPELESS: return "DXGI_FORMAT_R32G8X24_TYPELESS";
  case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return "DXGI_FORMAT_D32_FLOAT_S8X24_UINT";
  case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return "DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS";
  case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT: return "DXGI_FORMAT_X32_TYPELESS_G8X24_UINT";
  case DXGI_FORMAT_R10G10B10A2_TYPELESS: return "DXGI_FORMAT_R10G10B10A2_TYPELESS";
  case DXGI_FORMAT_R10G10B10A2_UNORM: return "DXGI_FORMAT_R10G10B10A2_UNORM";
  case DXGI_FORMAT_R10G10B10A2_UINT: return "DXGI_FORMAT_R10G10B10A2_UINT";
  case DXGI_FORMAT_R11G11B10_FLOAT: return "DXGI_FORMAT_R11G11B10_FLOAT";
  case DXGI_FORMAT_R8G8B8A8_TYPELESS: return "DXGI_FORMAT_R8G8B8A8_TYPELESS";
  case DXGI_FORMAT_R8G8B8A8_UNORM: return "DXGI_FORMAT_R8G8B8A8_UNORM";
  case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB";
  case DXGI_FORMAT_R8G8B8A8_UINT: return "DXGI_FORMAT_R8G8B8A8_UINT";
  case DXGI_FORMAT_R8G8B8A8_SNORM: return "DXGI_FORMAT_R8G8B8A8_SNORM";
  case DXGI_FORMAT_R8G8B8A8_SINT: return "DXGI_FORMAT_R8G8B8A8_SINT";
  case DXGI_FORMAT_R16G16_TYPELESS: return "DXGI_FORMAT_R16G16_TYPELESS";
  case DXGI_FORMAT_R16G16_FLOAT: return "DXGI_FORMAT_R16G16_FLOAT";
  case DXGI_FORMAT_R16G16_UNORM: return "DXGI_FORMAT_R16G16_UNORM";
  case DXGI_FORMAT_R16G16_UINT: return "DXGI_FORMAT_R16G16_UINT";
  case DXGI_FORMAT_R16G16_SNORM: return "DXGI_FORMAT_R16G16_SNORM";
  case DXGI_FORMAT_R16G16_SINT: return "DXGI_FORMAT_R16G16_SINT";
  case DXGI_FORMAT_R32_TYPELESS: return "DXGI_FORMAT_R32_TYPELESS";
  case DXGI_FORMAT_D32_FLOAT: return "DXGI_FORMAT_D32_FLOAT";
  case DXGI_FORMAT_R32_FLOAT: return "DXGI_FORMAT_R32_FLOAT";
  case DXGI_FORMAT_R32_UINT: return "DXGI_FORMAT_R32_UINT";
  case DXGI_FORMAT_R32_SINT: return "DXGI_FORMAT_R32_SINT";
  case DXGI_FORMAT_R24G8_TYPELESS: return "DXGI_FORMAT_R24G8_TYPELESS";
  case DXGI_FORMAT_D24_UNORM_S8_UINT: return "DXGI_FORMAT_D24_UNORM_S8_UINT";
  case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: return "DXGI_FORMAT_R24_UNORM_X8_TYPELESS";
  case DXGI_FORMAT_X24_TYPELESS_G8_UINT: return "DXGI_FORMAT_X24_TYPELESS_G8_UINT";
  case DXGI_FORMAT_R8G8_TYPELESS: return "DXGI_FORMAT_R8G8_TYPELESS";
  case DXGI_FORMAT_R8G8_UNORM: return "DXGI_FORMAT_R8G8_UNORM";
  case DXGI_FORMAT_R8G8_UINT: return "DXGI_FORMAT_R8G8_UINT";
  case DXGI_FORMAT_R8G8_SNORM: return "DXGI_FORMAT_R8G8_SNORM";
  case DXGI_FORMAT_R8G8_SINT: return "DXGI_FORMAT_R8G8_SINT";
  case DXGI_FORMAT_R16_TYPELESS: return "DXGI_FORMAT_R16_TYPELESS";
  case DXGI_FORMAT_R16_FLOAT: return "DXGI_FORMAT_R16_FLOAT";
  case DXGI_FORMAT_D16_UNORM: return "DXGI_FORMAT_D16_UNORM";
  case DXGI_FORMAT_R16_UNORM: return "DXGI_FORMAT_R16_UNORM";
  case DXGI_FORMAT_R16_UINT: return "DXGI_FORMAT_R16_UINT";
  case DXGI_FORMAT_R16_SNORM: return "DXGI_FORMAT_R16_SNORM";
  case DXGI_FORMAT_R16_SINT: return "DXGI_FORMAT_R16_SINT";
  case DXGI_FORMAT_R8_TYPELESS: return "DXGI_FORMAT_R8_TYPELESS";
  case DXGI_FORMAT_R8_UNORM: return "DXGI_FORMAT_R8_UNORM";
  case DXGI_FORMAT_R8_UINT: return "DXGI_FORMAT_R8_UINT";
  case DXGI_FORMAT_R8_SNORM: return "DXGI_FORMAT_R8_SNORM";
  case DXGI_FORMAT_R8_SINT: return "DXGI_FORMAT_R8_SINT";
  case DXGI_FORMAT_A8_UNORM: return "DXGI_FORMAT_A8_UNORM";
  case DXGI_FORMAT_R1_UNORM: return "DXGI_FORMAT_R1_UNORM";
  case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: return "DXGI_FORMAT_R9G9B9E5_SHAREDEXP";
  case DXGI_FORMAT_R8G8_B8G8_UNORM: return "DXGI_FORMAT_R8G8_B8G8_UNORM";
  case DXGI_FORMAT_G8R8_G8B8_UNORM: return "DXGI_FORMAT_G8R8_G8B8_UNORM";
  case DXGI_FORMAT_BC1_TYPELESS: return "DXGI_FORMAT_BC1_TYPELESS";
  case DXGI_FORMAT_BC1_UNORM: return "DXGI_FORMAT_BC1_UNORM";
  case DXGI_FORMAT_BC1_UNORM_SRGB: return "DXGI_FORMAT_BC1_UNORM_SRGB";
  case DXGI_FORMAT_BC2_TYPELESS: return "DXGI_FORMAT_BC2_TYPELESS";
  case DXGI_FORMAT_BC2_UNORM: return "DXGI_FORMAT_BC2_UNORM";
  case DXGI_FORMAT_BC2_UNORM_SRGB: return "DXGI_FORMAT_BC2_UNORM_SRGB";
  case DXGI_FORMAT_BC3_TYPELESS: return "DXGI_FORMAT_BC3_TYPELESS";
  case DXGI_FORMAT_BC3_UNORM: return "DXGI_FORMAT_BC3_UNORM";
  case DXGI_FORMAT_BC3_UNORM_SRGB: return "DXGI_FORMAT_BC3_UNORM_SRGB";
  case DXGI_FORMAT_BC4_TYPELESS: return "DXGI_FORMAT_BC4_TYPELESS";
  case DXGI_FORMAT_BC4_UNORM: return "DXGI_FORMAT_BC4_UNORM";
  case DXGI_FORMAT_BC4_SNORM: return "DXGI_FORMAT_BC4_SNORM";
  case DXGI_FORMAT_BC5_TYPELESS: return "DXGI_FORMAT_BC5_TYPELESS";
  case DXGI_FORMAT_BC5_UNORM: return "DXGI_FORMAT_BC5_UNORM";
  case DXGI_FORMAT_BC5_SNORM: return "DXGI_FORMAT_BC5_SNORM";
  case DXGI_FORMAT_B5G6R5_UNORM: return "DXGI_FORMAT_B5G6R5_UNORM";
  case DXGI_FORMAT_B5G5R5A1_UNORM: return "DXGI_FORMAT_B5G5R5A1_UNORM";
  case DXGI_FORMAT_B8G8R8A8_UNORM: return "DXGI_FORMAT_B8G8R8A8_UNORM";
  case DXGI_FORMAT_B8G8R8X8_UNORM: return "DXGI_FORMAT_B8G8R8X8_UNORM";
  case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM: return "DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM";
  case DXGI_FORMAT_B8G8R8A8_TYPELESS: return "DXGI_FORMAT_B8G8R8A8_TYPELESS";
  case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return "DXGI_FORMAT_B8G8R8A8_UNORM_SRGB";
  case DXGI_FORMAT_B8G8R8X8_TYPELESS: return "DXGI_FORMAT_B8G8R8X8_TYPELESS";
  case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return "DXGI_FORMAT_B8G8R8X8_UNORM_SRGB";
  case DXGI_FORMAT_BC6H_TYPELESS: return "DXGI_FORMAT_BC6H_TYPELESS";
  case DXGI_FORMAT_BC6H_UF16: return "DXGI_FORMAT_BC6H_UF16";
  case DXGI_FORMAT_BC6H_SF16: return "DXGI_FORMAT_BC6H_SF16";
  case DXGI_FORMAT_BC7_TYPELESS: return "DXGI_FORMAT_BC7_TYPELESS";
  case DXGI_FORMAT_BC7_UNORM: return "DXGI_FORMAT_BC7_UNORM";
  case DXGI_FORMAT_BC7_UNORM_SRGB: return "DXGI_FORMAT_BC7_UNORM_SRGB";
  case DXGI_FORMAT_AYUV: return "DXGI_FORMAT_AYUV";
  case DXGI_FORMAT_Y410: return "DXGI_FORMAT_Y410";
  case DXGI_FORMAT_Y416: return "DXGI_FORMAT_Y416";
  case DXGI_FORMAT_NV12: return "DXGI_FORMAT_NV12";
  case DXGI_FORMAT_P010: return "DXGI_FORMAT_P010";
  case DXGI_FORMAT_P016: return "DXGI_FORMAT_P016";
  case DXGI_FORMAT_420_OPAQUE: return "DXGI_FORMAT_420_OPAQUE";
  case DXGI_FORMAT_YUY2: return "DXGI_FORMAT_YUY2";
  case DXGI_FORMAT_Y210: return "DXGI_FORMAT_Y210";
  case DXGI_FORMAT_Y216: return "DXGI_FORMAT_Y216";
  case DXGI_FORMAT_NV11: return "DXGI_FORMAT_NV11";
  case DXGI_FORMAT_AI44: return "DXGI_FORMAT_AI44";
  case DXGI_FORMAT_IA44: return "DXGI_FORMAT_IA44";
  case DXGI_FORMAT_P8: return "DXGI_FORMAT_P8";
  case DXGI_FORMAT_A8P8: return "DXGI_FORMAT_A8P8";
  case DXGI_FORMAT_B4G4R4A4_UNORM: return "DXGI_FORMAT_B4G4R4A4_UNORM";
  case DXGI_FORMAT_P208: return "DXGI_FORMAT_P208";
  case DXGI_FORMAT_V208: return "DXGI_FORMAT_V208";
  case DXGI_FORMAT_V408: return "DXGI_FORMAT_V408";
  case DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE: return "DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE";
  case DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE: return "DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE";
  case DXGI_FORMAT_FORCE_UINT: return "DXGI_FORMAT_FORCE_UINT";
  default:
    return "DXGI_FORMAT_UNKNOWN";
  }
}

inline DXGI_FORMAT getDxgiFormat(const DdsFile *dds)
{
  if (dds->header.pixelFormat.flags & dds::DDPF_FOURCC) {
    switch (dds->header.pixelFormat.fourCC) {
    case "DXT1"_cc:
      return DXGI_FORMAT_BC1_UNORM;
    case "DXT2"_cc:
    case "DXT3"_cc:
      return DXGI_FORMAT_BC2_UNORM;
    case "DXT4"_cc:
    case "DXT5"_cc:
      return DXGI_FORMAT_BC3_UNORM;
    case "ATI1"_cc:
    case "BC4U"_cc:
      return DXGI_FORMAT_BC4_UNORM;
    case "BC4S"_cc:
      return DXGI_FORMAT_BC4_SNORM;
    case "ATI2"_cc:
    case "BC5U"_cc:
      return DXGI_FORMAT_BC5_UNORM;
    case "BC5S"_cc:
      return DXGI_FORMAT_BC5_SNORM;
    case "DX10"_cc: {
      return dds->header10.dxgiFormat;
    }
    default:
      break;
    }
  } else {
    // Try and guess a dxigi format based on header contents
    if (dds->header.pixelFormat.flags & dds::DDPF_RGB) {
      if (dds->header.pixelFormat.rgbBitCount == 32) {
        if (dds->header.pixelFormat.rBitMask == 0x00FF0000
            && dds->header.pixelFormat.gBitMask == 0x0000FF00
            && dds->header.pixelFormat.bBitMask == 0x000000FF
            && dds->header.pixelFormat.aBitMask == 0xFF000000) {
          return DXGI_FORMAT_B8G8R8A8_UNORM;
        }
      }
    }
    if (dds->header.pixelFormat.flags & dds::DDPF_ALPHAPIXELS) {
      if (dds->header.pixelFormat.rgbBitCount == 32) {
        if (dds->header.pixelFormat.rBitMask == 0x000000FF
            && dds->header.pixelFormat.gBitMask == 0x0000FF00
            && dds->header.pixelFormat.bBitMask == 0x00FF0000
            && dds->header.pixelFormat.aBitMask == 0xFF000000) {
          return DXGI_FORMAT_R8G8B8A8_UNORM;
        }
      }
    }
  }

  return DXGI_FORMAT_UNKNOWN;
}

inline std::uint32_t computeLinearSize(const DdsFile *dds)
{
  std::uint32_t blockSize = 0;
  auto format = getDxgiFormat(dds);

  switch (format) {
  case DXGI_FORMAT_BC1_UNORM:
  case DXGI_FORMAT_BC1_UNORM_SRGB:
  case DXGI_FORMAT_BC4_SNORM:
  case DXGI_FORMAT_BC4_UNORM: {
    blockSize = 8;
    break;
  }
  case DXGI_FORMAT_BC2_UNORM:
  case DXGI_FORMAT_BC2_UNORM_SRGB:
  case DXGI_FORMAT_BC3_UNORM:
  case DXGI_FORMAT_BC3_UNORM_SRGB:
  case DXGI_FORMAT_BC5_SNORM:
  case DXGI_FORMAT_BC5_UNORM:
  case DXGI_FORMAT_BC6H_SF16:
  case DXGI_FORMAT_BC6H_UF16:
  case DXGI_FORMAT_BC7_UNORM:
  case DXGI_FORMAT_BC7_UNORM_SRGB: {
    blockSize = 16;
    break;
  }
  default:
    break;
  }

  return (std::max)(1u, ((dds->header.width + 3) / 4)) * blockSize
      * ((dds->header.height + 3) / 4);
}

inline std::uint32_t computePitch(const DdsHeader &header)
{
  assert(header.flags & DDSD_PITCH);

  return (header.width * header.pixelFormat.rgbBitCount + 7) / 8;
}

inline bool vflipImage(const DdsFile *dds, std::byte *output)
{
  // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#compression-algorithms

  auto swapImage =
      [](auto *to, const auto *from, auto swapper, auto width, auto height) {
        using Block = decltype(*to);

        auto inBlocks = from;
        auto outBlocks = to;

        auto srcBlockLine = inBlocks;
        auto destBlockLine = outBlocks + (height - 1) * width;

        for (auto j = 0; j < height; ++j) {
          for (auto i = 0; i < width; ++i) {
            destBlockLine[i] = swapper(srcBlockLine[i]);
          }
          srcBlockLine += width;
          destBlockLine -= width;
        }
        // Special case for the center line in case of an odd block size image.
        if (height % 2) {
          auto srcBlockLine = inBlocks + height / 2 * width;
          auto destBlockLine = outBlocks + height / 2 * width;

          for (auto i = 0; i < width; ++i) {
            destBlockLine[i] = swapper(srcBlockLine[i]);
          }
        }
      };

  auto format = getDxgiFormat(dds);
  switch (format) {
  case DXGI_FORMAT_BC1_UNORM:
  case DXGI_FORMAT_BC1_UNORM_SRGB: {
    // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc1
    struct Block
    {
      std::uint16_t color0; // base colors
      std::uint16_t color1;
      std::uint8_t c0; // weights
      std::uint8_t c1;
      std::uint8_t c2;
      std::uint8_t c3;
    };
    static_assert(sizeof(Block) == 8);

    auto getBlockSwapped = [](Block block) -> Block {
      using std::swap;
      swap(block.c0, block.c3);
      swap(block.c1, block.c2);

      return block;
    };

    auto inBlocks = reinterpret_cast<const Block *>(getDataPointer(dds));
    auto outBlocks = reinterpret_cast<Block *>(output);

    auto width = (dds->header.width + 3) / 4;
    auto height = (dds->header.height + 3) / 4;

    swapImage(outBlocks, inBlocks, getBlockSwapped, width, height);
    break;
  }

  case DXGI_FORMAT_BC2_UNORM:
  case DXGI_FORMAT_BC2_UNORM_SRGB: {
    // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc2
    struct Block
    {
      std::uint16_t a0; // alphas
      std::uint16_t a1;
      std::uint16_t a2;
      std::uint16_t a3;
      std::uint16_t color0; // base colors
      std::uint16_t color1;
      std::uint8_t c0; // weigths
      std::uint8_t c1;
      std::uint8_t c2;
      std::uint8_t c3;
    };
    static_assert(sizeof(Block) == 16);

    auto getBlockSwapped = [](Block block) -> Block {
      using std::swap;
      swap(block.a0, block.a3);
      swap(block.a1, block.a2);
      swap(block.c0, block.c3);
      swap(block.c1, block.c2);

      return block;
    };

    auto inBlocks = reinterpret_cast<const Block *>(getDataPointer(dds));
    auto outBlocks = reinterpret_cast<Block *>(output);

    auto width = (dds->header.width + 3) / 4;
    auto height = (dds->header.height + 3) / 4;

    swapImage(outBlocks, inBlocks, getBlockSwapped, width, height);
    break;
  }
  case DXGI_FORMAT_BC3_UNORM:
  case DXGI_FORMAT_BC3_UNORM_SRGB: {
  // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc3
    struct Block
    {
      std::uint8_t alpha0; // base alphas
      std::uint8_t alpha1;
      std::uint8_t a0, a1, a2, a3, a4, a5; //  alpha weights
      std::uint16_t color0; // base colors
      std::uint16_t color1;
      std::uint8_t c0; // color weights
      std::uint8_t c1;
      std::uint8_t c2;
      std::uint8_t c3;
    };
    static_assert(sizeof(Block) == 16);

    auto getBlockSwapped = [](Block block) -> Block {
      using std::swap;
      swap(block.c0, block.c3);
      swap(block.c1, block.c2);

      std::uint8_t swapped[6];
      swapped[0] = (block.a4 >> 4) | (block.a5 << 4);
      swapped[1] = (block.a5 >> 4) | (block.a3 << 4);
      swapped[2] = (block.a3 >> 4) | (block.a4 << 4);
      swapped[3] = (block.a1 >> 4) | (block.a2 << 4);
      swapped[4] = (block.a2 >> 4) | (block.a0 << 4);
      swapped[5] = (block.a0 >> 4) | (block.a1 << 4);

      block.a0 = swapped[0];
      block.a1 = swapped[1];
      block.a2 = swapped[2];
      block.a3 = swapped[3];
      block.a4 = swapped[4];
      block.a5 = swapped[5];

      return block;
    };

    auto inBlocks = reinterpret_cast<const Block *>(getDataPointer(dds));
    auto outBlocks = reinterpret_cast<Block *>(output);

    auto width = (dds->header.width + 3) / 4;
    auto height = (dds->header.height + 3) / 4;

    swapImage(outBlocks, inBlocks, getBlockSwapped, width, height);
    break;
  }
  case DXGI_FORMAT_BC4_SNORM:
  case DXGI_FORMAT_BC4_UNORM: {
    // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc4
    struct Block
    {
      std::uint8_t r0; // base luminance
      std::uint8_t r1;
      std::uint8_t w0, w1, w2, w3, w4, w5; // weights
    };
    static_assert(sizeof(Block) == 8);

    auto getBlockSwapped = [](Block block) -> Block {
      using std::swap;

      std::uint8_t swapped[6];
      swapped[0] = (block.w4 >> 4) | (block.w5 << 4);
      swapped[1] = (block.w5 >> 4) | (block.w3 << 4);
      swapped[2] = (block.w3 >> 4) | (block.w4 << 4);
      swapped[3] = (block.w1 >> 4) | (block.w2 << 4);
      swapped[4] = (block.w2 >> 4) | (block.w0 << 4);
      swapped[5] = (block.w0 >> 4) | (block.w1 << 4);

      block.w0 = swapped[0];
      block.w1 = swapped[1];
      block.w2 = swapped[2];
      block.w3 = swapped[3];
      block.w4 = swapped[4];
      block.w5 = swapped[5];
      return block;
    };

    auto inBlocks = reinterpret_cast<const Block *>(getDataPointer(dds));
    auto outBlocks = reinterpret_cast<Block *>(output);

    auto width = (dds->header.width + 3) / 4;
    auto height = (dds->header.height + 3) / 4;

    swapImage(outBlocks, inBlocks, getBlockSwapped, width, height);
    break;
  }
  case DXGI_FORMAT_BC5_SNORM:
  case DXGI_FORMAT_BC5_UNORM: {
    // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc5
    struct Block
    {
      std::uint8_t r0; // base luminance
      std::uint8_t r1;
      std::uint8_t wr0, wr1, wr2, wr3, wr4, wr5; //  weights
      std::uint8_t g0; // base luminance
      std::uint8_t g1;
      std::uint8_t wg0, wg1, wg2, wg3, wg4, wg5; //  weights
    };
    static_assert(sizeof(Block) == 16);

    auto getBlockSwapped = [](Block block) -> Block {
      using std::swap;

      std::uint8_t swapped[6];
      swapped[0] = (block.wr4 >> 4) | (block.wr5 << 4);
      swapped[1] = (block.wr5 >> 4) | (block.wr3 << 4);
      swapped[2] = (block.wr3 >> 4) | (block.wr4 << 4);
      swapped[3] = (block.wr1 >> 4) | (block.wr2 << 4);
      swapped[4] = (block.wr2 >> 4) | (block.wr0 << 4);
      swapped[5] = (block.wr0 >> 4) | (block.wr1 << 4);

      block.wr0 = swapped[0];
      block.wr1 = swapped[1];
      block.wr2 = swapped[2];
      block.wr3 = swapped[3];
      block.wr4 = swapped[4];
      block.wr5 = swapped[5];

      swapped[0] = (block.wg4 >> 4) | (block.wg5 << 4);
      swapped[1] = (block.wg5 >> 4) | (block.wg3 << 4);
      swapped[2] = (block.wg3 >> 4) | (block.wg4 << 4);
      swapped[3] = (block.wg1 >> 4) | (block.wg2 << 4);
      swapped[4] = (block.wg2 >> 4) | (block.wg0 << 4);
      swapped[5] = (block.wg0 >> 4) | (block.wg1 << 4);

      block.wg0 = swapped[0];
      block.wg1 = swapped[1];
      block.wg2 = swapped[2];
      block.wg3 = swapped[3];
      block.wg4 = swapped[4];
      block.wg5 = swapped[5];

      return block;
    };

    auto inBlocks = reinterpret_cast<const Block *>(getDataPointer(dds));
    auto outBlocks = reinterpret_cast<Block *>(output);

    auto width = (dds->header.width + 3) / 4;
    auto height = (dds->header.height + 3) / 4;

    swapImage(outBlocks, inBlocks, getBlockSwapped, width, height);
    break;
  }
  case DXGI_FORMAT_BC6H_SF16:
  case DXGI_FORMAT_BC6H_UF16:
  case DXGI_FORMAT_BC7_UNORM:
  case DXGI_FORMAT_BC7_UNORM_SRGB:
    // https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc6h-format
    // https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format-mode-reference
    // We don't suppor those yet. Just copy the image.

  default: {
    // Don't know how to flip the image. Return the original one.
    std::memcpy(output, getDataPointer(dds), computeLinearSize(dds));
    return false;
  }
  }

  return true;
}

} // namespace visrtx::dds

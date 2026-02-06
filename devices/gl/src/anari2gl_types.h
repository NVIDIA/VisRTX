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

#include <anari/anari.h>
#include "ogl.h"
#include "VisGLString.h"

namespace visgl {

static inline GLenum gl_type(ANARIDataType t)
{
  switch (t) {
  case ANARI_UFIXED8: return GL_UNSIGNED_BYTE;
  case ANARI_UFIXED8_VEC2: return GL_UNSIGNED_BYTE;
  case ANARI_UFIXED8_VEC3: return GL_UNSIGNED_BYTE;
  case ANARI_UFIXED8_VEC4: return GL_UNSIGNED_BYTE;
  case ANARI_UFIXED8_R_SRGB: return GL_UNSIGNED_BYTE;
  case ANARI_UFIXED8_RA_SRGB: return GL_UNSIGNED_BYTE;
  case ANARI_UFIXED8_RGB_SRGB: return GL_UNSIGNED_BYTE;
  case ANARI_UFIXED8_RGBA_SRGB: return GL_UNSIGNED_BYTE;
  case ANARI_UFIXED16: return GL_UNSIGNED_SHORT;
  case ANARI_UFIXED16_VEC2: return GL_UNSIGNED_SHORT;
  case ANARI_UFIXED16_VEC3: return GL_UNSIGNED_SHORT;
  case ANARI_UFIXED16_VEC4: return GL_UNSIGNED_SHORT;
  case ANARI_UFIXED32: return GL_UNSIGNED_INT;
  case ANARI_UFIXED32_VEC2: return GL_UNSIGNED_INT;
  case ANARI_UFIXED32_VEC3: return GL_UNSIGNED_INT;
  case ANARI_UFIXED32_VEC4: return GL_UNSIGNED_INT;

  case ANARI_FIXED8: return GL_BYTE;
  case ANARI_FIXED8_VEC2: return GL_BYTE;
  case ANARI_FIXED8_VEC3: return GL_BYTE;
  case ANARI_FIXED8_VEC4: return GL_BYTE;
  case ANARI_FIXED16: return GL_SHORT;
  case ANARI_FIXED16_VEC2: return GL_SHORT;
  case ANARI_FIXED16_VEC3: return GL_SHORT;
  case ANARI_FIXED16_VEC4: return GL_SHORT;
  case ANARI_FIXED32: return GL_INT;
  case ANARI_FIXED32_VEC2: return GL_INT;
  case ANARI_FIXED32_VEC3: return GL_INT;
  case ANARI_FIXED32_VEC4: return GL_INT;

  case ANARI_UINT8: return GL_UNSIGNED_BYTE;
  case ANARI_UINT8_VEC2: return GL_UNSIGNED_BYTE;
  case ANARI_UINT8_VEC3: return GL_UNSIGNED_BYTE;
  case ANARI_UINT8_VEC4: return GL_UNSIGNED_BYTE;
  case ANARI_UINT16: return GL_UNSIGNED_SHORT;
  case ANARI_UINT16_VEC2: return GL_UNSIGNED_SHORT;
  case ANARI_UINT16_VEC3: return GL_UNSIGNED_SHORT;
  case ANARI_UINT16_VEC4: return GL_UNSIGNED_SHORT;
  case ANARI_UINT32: return GL_UNSIGNED_INT;
  case ANARI_UINT32_VEC2: return GL_UNSIGNED_INT;
  case ANARI_UINT32_VEC3: return GL_UNSIGNED_INT;
  case ANARI_UINT32_VEC4: return GL_UNSIGNED_INT;

  case ANARI_INT8: return GL_BYTE;
  case ANARI_INT8_VEC2: return GL_BYTE;
  case ANARI_INT8_VEC3: return GL_BYTE;
  case ANARI_INT8_VEC4: return GL_BYTE;
  case ANARI_INT16: return GL_SHORT;
  case ANARI_INT16_VEC2: return GL_SHORT;
  case ANARI_INT16_VEC3: return GL_SHORT;
  case ANARI_INT16_VEC4: return GL_SHORT;
  case ANARI_INT32: return GL_INT;
  case ANARI_INT32_VEC2: return GL_INT;
  case ANARI_INT32_VEC3: return GL_INT;
  case ANARI_INT32_VEC4: return GL_INT;

  case ANARI_FLOAT32: return GL_FLOAT;
  case ANARI_FLOAT32_VEC2: return GL_FLOAT;
  case ANARI_FLOAT32_VEC3: return GL_FLOAT;
  case ANARI_FLOAT32_VEC4: return GL_FLOAT;

  default: return GL_NONE;
  }
}

static inline GLenum gl_format(ANARIDataType t)
{
  switch (t) {
  case ANARI_UFIXED8: return GL_RED;
  case ANARI_UFIXED8_VEC2: return GL_RG;
  case ANARI_UFIXED8_VEC3: return GL_RGB;
  case ANARI_UFIXED8_VEC4: return GL_RGBA;
  case ANARI_UFIXED8_R_SRGB: return GL_RED;
  case ANARI_UFIXED8_RA_SRGB: return GL_RG;
  case ANARI_UFIXED8_RGB_SRGB: return GL_RGB;
  case ANARI_UFIXED8_RGBA_SRGB: return GL_RGBA;
  case ANARI_UFIXED16: return GL_RED;
  case ANARI_UFIXED16_VEC2: return GL_RG;
  case ANARI_UFIXED16_VEC3: return GL_RGB;
  case ANARI_UFIXED16_VEC4: return GL_RGBA;
  case ANARI_UFIXED32: return GL_RED;
  case ANARI_UFIXED32_VEC2: return GL_RG;
  case ANARI_UFIXED32_VEC3: return GL_RGB;
  case ANARI_UFIXED32_VEC4: return GL_RGBA;

  case ANARI_FIXED8: return GL_RED;
  case ANARI_FIXED8_VEC2: return GL_RG;
  case ANARI_FIXED8_VEC3: return GL_RGB;
  case ANARI_FIXED8_VEC4: return GL_RGBA;
  case ANARI_FIXED16: return GL_RED;
  case ANARI_FIXED16_VEC2: return GL_RG;
  case ANARI_FIXED16_VEC3: return GL_RGB;
  case ANARI_FIXED16_VEC4: return GL_RGBA;
  case ANARI_FIXED32: return GL_RED;
  case ANARI_FIXED32_VEC2: return GL_RG;
  case ANARI_FIXED32_VEC3: return GL_RGB;
  case ANARI_FIXED32_VEC4: return GL_RGBA;

  case ANARI_UINT8: return GL_RED;
  case ANARI_UINT8_VEC2: return GL_RG;
  case ANARI_UINT8_VEC3: return GL_RGB;
  case ANARI_UINT8_VEC4: return GL_RGBA;
  case ANARI_UINT16: return GL_RED;
  case ANARI_UINT16_VEC2: return GL_RG;
  case ANARI_UINT16_VEC3: return GL_RGB;
  case ANARI_UINT16_VEC4: return GL_RGBA;
  case ANARI_UINT32: return GL_RED;
  case ANARI_UINT32_VEC2: return GL_RG;
  case ANARI_UINT32_VEC3: return GL_RGB;
  case ANARI_UINT32_VEC4: return GL_RGBA;

  case ANARI_INT8: return GL_RED;
  case ANARI_INT8_VEC2: return GL_RG;
  case ANARI_INT8_VEC3: return GL_RGB;
  case ANARI_INT8_VEC4: return GL_RGBA;
  case ANARI_INT16: return GL_RED;
  case ANARI_INT16_VEC2: return GL_RG;
  case ANARI_INT16_VEC3: return GL_RGB;
  case ANARI_INT16_VEC4: return GL_RGBA;
  case ANARI_INT32: return GL_RED;
  case ANARI_INT32_VEC2: return GL_RG;
  case ANARI_INT32_VEC3: return GL_RGB;
  case ANARI_INT32_VEC4: return GL_RGBA;

  case ANARI_FLOAT32: return GL_RED;
  case ANARI_FLOAT32_VEC2: return GL_RG;
  case ANARI_FLOAT32_VEC3: return GL_RGB;
  case ANARI_FLOAT32_VEC4: return GL_RGBA;

  default: return GL_NONE;
  }
}

static inline GLenum gl_normalized(ANARIDataType t)
{
  switch (t) {
  case ANARI_UFIXED8:
  case ANARI_UFIXED8_VEC2:
  case ANARI_UFIXED8_VEC3:
  case ANARI_UFIXED8_VEC4:
  case ANARI_UFIXED8_R_SRGB:
  case ANARI_UFIXED8_RA_SRGB:
  case ANARI_UFIXED8_RGB_SRGB:
  case ANARI_UFIXED8_RGBA_SRGB:
  case ANARI_UFIXED16:
  case ANARI_UFIXED16_VEC2:
  case ANARI_UFIXED16_VEC3:
  case ANARI_UFIXED16_VEC4:
  case ANARI_UFIXED32:
  case ANARI_UFIXED32_VEC2:
  case ANARI_UFIXED32_VEC3:
  case ANARI_UFIXED32_VEC4:

  case ANARI_FIXED8:
  case ANARI_FIXED8_VEC2:
  case ANARI_FIXED8_VEC3:
  case ANARI_FIXED8_VEC4:
  case ANARI_FIXED16:
  case ANARI_FIXED16_VEC2:
  case ANARI_FIXED16_VEC3:
  case ANARI_FIXED16_VEC4:
  case ANARI_FIXED32:
  case ANARI_FIXED32_VEC2:
  case ANARI_FIXED32_VEC3:
  case ANARI_FIXED32_VEC4: return GL_TRUE;

  default: return GL_FALSE;
  }
}

// depending on GL version/API not all of these
// gl defines will exist so we guard them
static inline GLenum gl_internal_format(ANARIDataType t)
{
  switch (t) {
#ifdef GL_R8
  case ANARI_UFIXED8: return GL_R8;
#endif
#ifdef GL_RG8
  case ANARI_UFIXED8_VEC2: return GL_RG8;
#endif
#ifdef GL_RGB8
  case ANARI_UFIXED8_VEC3: return GL_RGB8;
#endif
#ifdef GL_RGBA8
  case ANARI_UFIXED8_VEC4: return GL_RGBA8;
#endif
#ifdef GL_R8
  case ANARI_UFIXED8_R_SRGB: return GL_R8;
#endif
#ifdef GL_RG8
  case ANARI_UFIXED8_RA_SRGB: return GL_RG8;
#endif
#ifdef GL_SRGB8
  case ANARI_UFIXED8_RGB_SRGB: return GL_SRGB8;
#endif
#ifdef GL_SRGB8_ALPHA8
  case ANARI_UFIXED8_RGBA_SRGB: return GL_SRGB8_ALPHA8;
#endif
#ifdef GL_R16
  case ANARI_UFIXED16: return GL_R16;
#endif
#ifdef GL_RG16
  case ANARI_UFIXED16_VEC2: return GL_RG16;
#endif
#ifdef GL_RGB16
  case ANARI_UFIXED16_VEC3: return GL_RGB16;
#endif
#ifdef GL_RGBA16
  case ANARI_UFIXED16_VEC4: return GL_RGBA16;
#endif

#ifdef GL_R32
  case ANARI_UFIXED32: return GL_R32;
#else
  case ANARI_UFIXED32: return GL_R32F;
#endif
#ifdef GL_RG32
  case ANARI_UFIXED32_VEC2: return GL_RG32;
#else
  case ANARI_UFIXED32_VEC2: return GL_RG32F;
#endif
#ifdef GL_RGB32
  case ANARI_UFIXED32_VEC3: return GL_RGB32;
#else
  case ANARI_UFIXED32_VEC3: return GL_RGB32F;
#endif
#ifdef GL_RGBA32
  case ANARI_UFIXED32_VEC4: return GL_RGBA32;
#else
  case ANARI_UFIXED32_VEC4: return GL_RGBA32F;
#endif

#ifdef GL_R8_SNORM
  case ANARI_FIXED8: return GL_R8_SNORM;
#endif
#ifdef GL_RG8_SNORM
  case ANARI_FIXED8_VEC2: return GL_RG8_SNORM;
#endif
#ifdef GL_RGB8_SNORM
  case ANARI_FIXED8_VEC3: return GL_RGB8_SNORM;
#endif
#ifdef GL_RGBA8_SNORM
  case ANARI_FIXED8_VEC4: return GL_RGBA8_SNORM;
#endif
#ifdef GL_R16_SNORM
  case ANARI_FIXED16: return GL_R16_SNORM;
#endif
#ifdef GL_RG16_SNORM
  case ANARI_FIXED16_VEC2: return GL_RG16_SNORM;
#endif
#ifdef GL_RGB16_SNORM
  case ANARI_FIXED16_VEC3: return GL_RGB16_SNORM;
#endif
#ifdef GL_RGBA16_SNORM
  case ANARI_FIXED16_VEC4: return GL_RGBA16_SNORM;
#endif
#ifdef GL_R32_SNORM
  case ANARI_FIXED32: return GL_R32_SNORM;
#endif
#ifdef GL_RG32_SNORM
  case ANARI_FIXED32_VEC2: return GL_RG32_SNORM;
#endif
#ifdef GL_RGB32_SNORM
  case ANARI_FIXED32_VEC3: return GL_RGB32_SNORM;
#endif
#ifdef GL_RGBA32_SNORM
  case ANARI_FIXED32_VEC4: return GL_RGBA32_SNORM;
#endif

#ifdef GL_R8UI
  case ANARI_UINT8: return GL_R8UI;
#endif
#ifdef GL_RG8UI
  case ANARI_UINT8_VEC2: return GL_RG8UI;
#endif
#ifdef GL_RGB8UI
  case ANARI_UINT8_VEC3: return GL_RGB8UI;
#endif
#ifdef GL_RGBA8UI
  case ANARI_UINT8_VEC4: return GL_RGBA8UI;
#endif
#ifdef GL_R16UI
  case ANARI_UINT16: return GL_R16UI;
#endif
#ifdef GL_RG16UI
  case ANARI_UINT16_VEC2: return GL_RG16UI;
#endif
#ifdef GL_RGB16UI
  case ANARI_UINT16_VEC3: return GL_RGB16UI;
#endif
#ifdef GL_RGBA16UI
  case ANARI_UINT16_VEC4: return GL_RGBA16UI;
#endif
#ifdef GL_R32UI
  case ANARI_UINT32: return GL_R32UI;
#endif
#ifdef GL_RG32UI
  case ANARI_UINT32_VEC2: return GL_RG32UI;
#endif
#ifdef GL_RGB32UI
  case ANARI_UINT32_VEC3: return GL_RGB32UI;
#endif
#ifdef GL_RGBA32UI
  case ANARI_UINT32_VEC4: return GL_RGBA32UI;
#endif

#ifdef GL_R8I
  case ANARI_INT8: return GL_R8I;
#endif
#ifdef GL_RG8I
  case ANARI_INT8_VEC2: return GL_RG8I;
#endif
#ifdef GL_RGB8I
  case ANARI_INT8_VEC3: return GL_RGB8I;
#endif
#ifdef GL_RGBA8I
  case ANARI_INT8_VEC4: return GL_RGBA8I;
#endif
#ifdef GL_R16I
  case ANARI_INT16: return GL_R16I;
#endif
#ifdef GL_RG16I
  case ANARI_INT16_VEC2: return GL_RG16I;
#endif
#ifdef GL_RGB16I
  case ANARI_INT16_VEC3: return GL_RGB16I;
#endif
#ifdef GL_RGBA16I
  case ANARI_INT16_VEC4: return GL_RGBA16I;
#endif
#ifdef GL_R32I
  case ANARI_INT32: return GL_R32I;
#endif
#ifdef GL_RG32I
  case ANARI_INT32_VEC2: return GL_RG32I;
#endif
#ifdef GL_RGB32I
  case ANARI_INT32_VEC3: return GL_RGB32I;
#endif
#ifdef GL_RGBA32I
  case ANARI_INT32_VEC4: return GL_RGBA32I;
#endif

#ifdef GL_R32F
  case ANARI_FLOAT32: return GL_R32F;
#endif
#ifdef GL_RG32F
  case ANARI_FLOAT32_VEC2: return GL_RG32F;
#endif
#ifdef GL_RGB32F
  case ANARI_FLOAT32_VEC3: return GL_RGB32F;
#endif
#ifdef GL_RGBA32F
  case ANARI_FLOAT32_VEC4: return GL_RGBA32F;
#endif

  default: return GL_NONE;
  }
}

static inline bool gl_mipmappable(GLenum t)
{
  switch (t) {
  case GL_R8:
  case GL_RG8:
  case GL_RGB8:
  case GL_RGBA8:
  case GL_RGB565:
  case GL_RGBA4:
  case GL_RGB5_A1:
  case GL_RGB10_A2:
  case GL_SRGB8_ALPHA8:
  case GL_R16F:
  case GL_RG16F:
  case GL_RGBA16F:
  case GL_R11F_G11F_B10F: return true;

  default: return false;
  }
}

static GLenum gl_min_filter(int value)
{
  switch (value) {
  case STRING_ENUM_linear: return GL_LINEAR;
  case STRING_ENUM_nearest: return GL_NEAREST;
  default: return GL_NEAREST;
  }
}

static GLenum gl_min_filter_mip(int value)
{
  switch (value) {
  case STRING_ENUM_linear: return GL_LINEAR_MIPMAP_LINEAR;
  case STRING_ENUM_nearest: return GL_NEAREST_MIPMAP_LINEAR;
  default: return GL_NEAREST;
  }
}

static GLenum gl_mag_filter(int value)
{
  switch (value) {
  case STRING_ENUM_linear: return GL_LINEAR;
  case STRING_ENUM_nearest: return GL_NEAREST;
  default: return GL_NEAREST;
  }
}

static GLenum gl_wrap(int value)
{
  switch (value) {
  case STRING_ENUM_repeat: return GL_REPEAT;
  case STRING_ENUM_mirrorRepeat: return GL_MIRRORED_REPEAT;
  case STRING_ENUM_clampToEdge: return GL_CLAMP_TO_EDGE;
  default: return GL_CLAMP_TO_EDGE;
  }
}


static GLenum gl_compressed_image(int value)
{
  switch (value) {
    case STRING_ENUM_BC1_RGB: return GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
    case STRING_ENUM_BC1_RGBA: return GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
    case STRING_ENUM_BC1_RGBA_SRGB: return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
    case STRING_ENUM_BC1_RGB_SRGB: return GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
    case STRING_ENUM_BC2: return GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
    case STRING_ENUM_BC2_SRGB: return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
    case STRING_ENUM_BC3: return GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
    case STRING_ENUM_BC3_SRGB: return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;

    case STRING_ENUM_BC4: return GL_COMPRESSED_RED_RGTC1;
    case STRING_ENUM_BC4_SNORM: return GL_COMPRESSED_SIGNED_RED_RGTC1;
    case STRING_ENUM_BC5: return GL_COMPRESSED_RG_RGTC2;
    case STRING_ENUM_BC5_SNORM: return GL_COMPRESSED_SIGNED_RG_RGTC2;
    case STRING_ENUM_BC6H_SFLOAT: return GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT;
    case STRING_ENUM_BC6H_UFLOAT: return GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT;
    case STRING_ENUM_BC7: return GL_COMPRESSED_RGBA_BPTC_UNORM;
    case STRING_ENUM_BC7_SRGB: return GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM;

    case STRING_ENUM_EAC_R11G11_SNORM: return GL_COMPRESSED_SIGNED_RG11_EAC;
    case STRING_ENUM_EAC_R11G11_UNORM: return GL_COMPRESSED_RG11_EAC;
    case STRING_ENUM_EAC_R11_SNORM: return GL_COMPRESSED_SIGNED_R11_EAC;
    case STRING_ENUM_EAC_R11_UNORM: return GL_COMPRESSED_R11_EAC;
    case STRING_ENUM_ETC2_R8G8B8: return GL_COMPRESSED_RGB8_ETC2;
    case STRING_ENUM_ETC2_R8G8B8A1: return GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2;
    case STRING_ENUM_ETC2_R8G8B8A1_SRGB: return GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2;
    case STRING_ENUM_ETC2_R8G8B8A8: return GL_COMPRESSED_RGBA8_ETC2_EAC;
    case STRING_ENUM_ETC2_R8G8B8A8_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC;
    case STRING_ENUM_ETC2_R8G8B8_SRGB: return GL_COMPRESSED_SRGB8_ETC2;

    case STRING_ENUM_ASTC_10x10: return GL_COMPRESSED_RGBA_ASTC_10x10;
    case STRING_ENUM_ASTC_10x10_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10;
    case STRING_ENUM_ASTC_10x5: return GL_COMPRESSED_RGBA_ASTC_10x5;
    case STRING_ENUM_ASTC_10x5_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5;
    case STRING_ENUM_ASTC_10x6: return GL_COMPRESSED_RGBA_ASTC_10x6;
    case STRING_ENUM_ASTC_10x6_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6;
    case STRING_ENUM_ASTC_10x8: return GL_COMPRESSED_RGBA_ASTC_10x8;
    case STRING_ENUM_ASTC_10x8_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8;
    case STRING_ENUM_ASTC_12x10: return GL_COMPRESSED_RGBA_ASTC_12x10;
    case STRING_ENUM_ASTC_12x10_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10;
    case STRING_ENUM_ASTC_12x12: return GL_COMPRESSED_RGBA_ASTC_12x12;
    case STRING_ENUM_ASTC_12x12_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12;
    case STRING_ENUM_ASTC_4x4: return GL_COMPRESSED_RGBA_ASTC_4x4;
    case STRING_ENUM_ASTC_4x4_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4;
    case STRING_ENUM_ASTC_5x4: return GL_COMPRESSED_RGBA_ASTC_5x4;
    case STRING_ENUM_ASTC_5x4_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4;
    case STRING_ENUM_ASTC_5x5: return GL_COMPRESSED_RGBA_ASTC_5x5;
    case STRING_ENUM_ASTC_5x5_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5;
    case STRING_ENUM_ASTC_6x5: return GL_COMPRESSED_RGBA_ASTC_6x5;
    case STRING_ENUM_ASTC_6x5_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5;
    case STRING_ENUM_ASTC_6x6: return GL_COMPRESSED_RGBA_ASTC_6x6;
    case STRING_ENUM_ASTC_6x6_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6;
    case STRING_ENUM_ASTC_8x5: return GL_COMPRESSED_RGBA_ASTC_8x5;
    case STRING_ENUM_ASTC_8x5_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5;
    case STRING_ENUM_ASTC_8x6: return GL_COMPRESSED_RGBA_ASTC_8x6;
    case STRING_ENUM_ASTC_8x6_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6;
    case STRING_ENUM_ASTC_8x8: return GL_COMPRESSED_RGBA_ASTC_8x8;
    case STRING_ENUM_ASTC_8x8_SRGB: return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8;
    default: return GL_NONE;
  }
}

} // namespace visgl
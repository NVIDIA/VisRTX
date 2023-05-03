#pragma once

#include <anari/anari.h>
#include "glad/gl.h"
#include "VisGLString.h"

namespace visgl {

static inline GLenum gl_type(ANARIDataType t) {
    switch(t) {
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

static inline GLenum gl_format(ANARIDataType t) {
    switch(t) {
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

static inline GLenum gl_normalized(ANARIDataType t) {
    switch(t) {
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
        case ANARI_FIXED32_VEC4:
            return GL_TRUE;

        default:
            return GL_FALSE;
    }
}

// depending on GL version/API not all of these
// gl defines will exist so we guard them
static inline GLenum gl_internal_format(ANARIDataType t) {
    switch(t) {
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

static inline bool gl_mipmappable(GLenum t) {
    switch(t) {
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
        case GL_R11F_G11F_B10F:
            return true;

        default:
            return false;
    }
}

static GLenum gl_min_filter(int value) {
  switch(value) {
    case STRING_ENUM_linear: return GL_LINEAR;
    case STRING_ENUM_nearest: return GL_NEAREST;
    default: return GL_NEAREST;
  }
}

static GLenum gl_min_filter_mip(int value) {
  switch(value) {
    case STRING_ENUM_linear: return GL_LINEAR_MIPMAP_LINEAR;
    case STRING_ENUM_nearest: return GL_NEAREST_MIPMAP_LINEAR;
    default: return GL_NEAREST;
  }
}

static GLenum gl_mag_filter(int value) {
  switch(value) {
    case STRING_ENUM_linear: return GL_LINEAR;
    case STRING_ENUM_nearest: return GL_NEAREST;
    default: return GL_NEAREST;
  }
}

static GLenum gl_wrap(int value) {
  switch(value) {
    case STRING_ENUM_repeat: return GL_REPEAT;
    case STRING_ENUM_mirrorRepeat: return GL_MIRRORED_REPEAT;
    case STRING_ENUM_clampToEdge: return GL_CLAMP_TO_EDGE;
    default: return GL_CLAMP_TO_EDGE;
  }
}

}
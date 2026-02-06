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

// clang-format off

#ifndef MATH_UTIL_H
#define MATH_UTIL_H

#include <float.h>
#include <math.h>

static const float PI = 3.14159265358979323846f;

static inline float dot3(const float *a, const float *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static inline float dist3(const float *a, const float *b) {
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    float dz = a[2] - b[2];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}
static inline float dot4(const float *a, const float *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
static inline float *cross(float *dst, const float *a, const float *b) {
    dst[0] = a[1] * b[2] - a[2] * b[1];
    dst[1] = a[2] * b[0] - a[0] * b[2];
    dst[2] = a[0] * b[1] - a[1] * b[0];
    return dst;
}
static inline float *cross_with_base(float *dst, const float *a, const float *b, const float *base) {
    dst[0] = (a[1] - base[1]) * (b[2] - base[2]) - (a[2] - base[2]) * (b[1] - base[1]);
    dst[1] = (a[2] - base[2]) * (b[0] - base[0]) - (a[0] - base[0]) * (b[2] - base[2]);
    dst[2] = (a[0] - base[0]) * (b[1] - base[1]) - (a[1] - base[1]) * (b[0] - base[0]);
    return dst;
}
static inline float triple_product(const float *a, const float *b, const float *c) {
    return (a[0] * b[1] * c[2] + a[1] * b[2] * c[0] + a[2] * b[0] * c[1]) -
           (a[0] * b[2] * c[1] + a[1] * b[0] * c[2] + a[2] * b[1] * c[0]);
}
// annoyingly fminf/fmaxf handle NaNs differently than sse and these
// ternary versions compile to single instructions
static inline float fast_minf(float a, float b) { return a < b ? a : b; }
static inline float fast_maxf(float a, float b) { return a > b ? a : b; }

static inline float clamp(float x, float low, float high) { return fast_minf(high, fast_maxf(low, x)); }

static inline float smoothstep(float x) {
    x = clamp(x, 0.0f, 1.0f);
    return (x * x) * (3.0f - 2.0f * x);
}
static inline void cpyTransform(float *A, const float *B) {
    for (int i = 0; i < 16; ++i) {
        A[i] = B[i];
    }
}
static inline void transpose(float *A, const float *B) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            A[i + 4 * j] = B[j + 4 * i];
        }
    }
}
static inline void normalize(float *x, float *y, float *z) {
    float il = 1.0 / sqrt(*x * *x + *y * *y + *z * *z);
    *x *= il;
    *y *= il;
    *z *= il;
}
static inline void normalize3(float *v) {
    float il = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] *= il;
    v[1] *= il;
    v[2] *= il;
}
static inline void mul3(float *A, float *B, float *C) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float dot = 0;
            for (int k = 0; k < 4; ++k) {
                dot += B[i + 4 * k] * C[k + 4 * j];
            }
            A[i + 4 * j] = dot;
        }
    }
}
static inline void mul2(float *A, float *C) {
    float tmp[16];
    cpyTransform(tmp, A);
    mul3(A, tmp, C);
}
static inline void transpose_mul3(float *A, float *B, float *C) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float dot = 0;
            for (int k = 0; k < 4; ++k) {
                dot += B[i + 4 * k] * C[j + 4 * k];
            }
            A[i + 4 * j] = dot;
        }
    }
}
static inline void transpose_mul2(float *A, float *C) {
    float tmp[16];
    cpyTransform(tmp, A);
    transpose_mul3(A, tmp, C);
}
static inline void setIdentity(float *A) {
    A[0] = 1;
    A[4] = 0;
    A[8] = 0;
    A[12] = 0;
    A[1] = 0;
    A[5] = 1;
    A[9] = 0;
    A[13] = 0;
    A[2] = 0;
    A[6] = 0;
    A[10] = 1;
    A[14] = 0;
    A[3] = 0;
    A[7] = 0;
    A[11] = 0;
    A[15] = 1;
}
static inline void setTranslate(float *A, float x, float y, float z) {
    A[0] = 1;
    A[4] = 0;
    A[8] = 0;
    A[12] = x;
    A[1] = 0;
    A[5] = 1;
    A[9] = 0;
    A[13] = y;
    A[2] = 0;
    A[6] = 0;
    A[10] = 1;
    A[14] = z;
    A[3] = 0;
    A[7] = 0;
    A[11] = 0;
    A[15] = 1;
}
static inline void mulTranslate(float *A, float x, float y, float z) {
    float tmp[16];
    setTranslate(tmp, x, y, z);
    mul2(A, tmp);
}
static inline void setScale(float *A, float x, float y, float z) {
    A[0] = x;
    A[4] = 0;
    A[8] = 0;
    A[12] = 0;
    A[1] = 0;
    A[5] = y;
    A[9] = 0;
    A[13] = 0;
    A[2] = 0;
    A[6] = 0;
    A[10] = z;
    A[14] = 0;
    A[3] = 0;
    A[7] = 0;
    A[11] = 0;
    A[15] = 1;
}
static inline void mulScale(float *A, float x, float y, float z) {
    float tmp[16];
    setScale(tmp, x, y, z);
    mul2(A, tmp);
}
static inline void setRotateFromVectors(float *A, const float *u, const float *v) {
    float lu2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    float lv2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if (lu2 * lv2 < 1.e-10) {
        setIdentity(A);
        return;
    }
    float scaler = 1.0f / sqrtf(lu2 * lv2);

    float c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) * scaler;

    float x = (v[1] * u[2] - v[2] * u[1]) * scaler;
    float y = (v[2] * u[0] - v[0] * u[2]) * scaler;
    float z = (v[0] * u[1] - v[1] * u[0]) * scaler;

    float s2 = x * x + y * y + z * z;
    if (s2 < 1.e-10) {
        setIdentity(A);
        return;
    }
    float cs2 = (1.0f - c) / s2;
    A[0] = c + x * x * cs2;
    A[4] = x * y * cs2 + z;
    A[8] = x * z * cs2 - y;
    A[12] = 0;
    A[1] = x * y * cs2 - z;
    A[5] = c + y * y * cs2;
    A[9] = y * z * cs2 + x;
    A[13] = 0;
    A[2] = x * z * cs2 + y;
    A[6] = y * z * cs2 - x;
    A[10] = c + z * z * cs2;
    A[14] = 0;
    A[3] = 0;
    A[7] = 0;
    A[11] = 0;
    A[15] = 1;
}
static inline void mulRotateFromVectors(float *A, const float *u, const float *v) {
    float tmp[16];
    setRotateFromVectors(tmp, u, v);
    mul2(A, tmp);
}
static inline void setRotate(float *A, float x, float y, float z, float angle) {
    float l = 1.0f / sqrtf(x * x + y * y + z * z);
    x *= l;
    y *= l;
    z *= l;
    float c = cosf(angle);
    float s = sinf(angle);
    A[0] = x * x * (1.0f - c) + c;
    A[4] = x * y * (1.0f - c) + z * s;
    A[8] = x * z * (1.0f - c) - y * s;
    A[12] = 0;
    A[1] = x * y * (1.0f - c) - z * s;
    A[5] = y * y * (1.0f - c) + c;
    A[9] = y * z * (1.0f - c) + x * s;
    A[13] = 0;
    A[2] = x * z * (1.0f - c) + y * s;
    A[6] = y * z * (1.0f - c) - x * s;
    A[10] = z * z * (1.0f - c) + c;
    A[14] = 0;
    A[3] = 0;
    A[7] = 0;
    A[11] = 0;
    A[15] = 1;
}
static inline void mulRotate(float *A, float x, float y, float z, float angle) {
    float tmp[16];
    setRotate(tmp, x, y, z, angle);
    mul2(A, tmp);
}
static inline void setFrustum(float *A, float left, float right, float top, float bottom, float near, float far) {
    A[0] = 2.0f * near / (right - left);
    A[4] = 0;
    A[8] = (right + left) / (right - left);
    A[12] = 0;
    A[1] = 0;
    A[5] = 2.0f * near / (top - bottom);
    A[9] = (top + bottom) / (top - bottom);
    A[13] = 0;
    A[2] = 0;
    A[6] = 0;
    A[10] = (far + near) / (near - far);
    A[14] = 2 * far * near / (near - far);
    A[3] = 0;
    A[7] = 0;
    A[11] = -1;
    A[15] = 0;
}
static inline void setInverseFrustum(float *A, float left, float right, float top, float bottom, float near,
                                     float far) {
    A[0] = (right - left) / (2.0f * near);
    A[4] = 0;
    A[8] = 0;
    A[12] = (right + left) / (2.0f * near);
    A[1] = 0;
    A[5] = (top - bottom) / (2.0f * near);
    A[9] = 0;
    A[13] = (top + bottom) / (2.0f * near);
    A[2] = 0;
    A[6] = 0;
    A[10] = 0;
    A[14] = -1;
    A[3] = 0;
    A[7] = 0;
    A[11] = (near - far) / (2.0f * near * far);
    A[15] = (near + far) / (2.0f * near * far);
}
static inline void mulInverseFrustum(float *A, float left, float right, float top, float bottom, float near,
                                     float far) {
    float tmp[16];
    setInverseFrustum(tmp, left, right, top, bottom, near, far);
    mul2(A, tmp);
}
static inline void setOrtho(float *A, float left, float right, float top, float bottom, float near, float far) {
    A[0] = 2.0f / (right - left);
    A[4] = 0;
    A[8] = 0;
    A[12] = -(left + right) / (right - left);
    A[1] = 0;
    A[5] = 2.0f / (top - bottom);
    A[9] = 0;
    A[13] = -(bottom + top) / (top - bottom);
    A[2] = 0;
    A[6] = 0;
    A[10] = 2.0f / (near - far);
    A[14] = -(near + far) / (far - near);
    A[3] = 0;
    A[7] = 0;
    A[11] = 0;
    A[15] = 1;
}
static inline void setInverseOrtho(float *A, float left, float right, float top, float bottom, float near, float far) {
    A[0] = 0.5 * (right - left);
    A[4] = 0;
    A[8] = 0;
    A[12] = -0.5 * (left + right);
    A[1] = 0;
    A[5] = 0.5 * (top - bottom);
    A[9] = 0;
    A[13] = -0.5 * (bottom + top);
    A[2] = 0;
    A[6] = 0;
    A[10] = 0.5 * (near - far);
    A[14] = -0.5 * (near + far);
    A[3] = 0;
    A[7] = 0;
    A[11] = 0;
    A[15] = 1;
}
static inline void mulInverseOrtho(float *A, float left, float right, float top, float bottom, float near, float far) {
    float tmp[16];
    setInverseOrtho(tmp, left, right, top, bottom, near, far);
    mul2(A, tmp);
}
static inline void transformVector(float *y, const float *A, const float *x) {
    float tmp[4];
    for (int i = 0; i < 4; ++i) {
        tmp[i] = A[0 + i] * x[0] + A[4 + i] * x[1] + A[8 + i] * x[2] + A[12 + i] * x[3];
    }
    for (int i = 0; i < 4; ++i) {
        y[i] = tmp[i];
    }
}
static inline void transformVector3(float *y, const float *A, const float *x) {
    float tmp[4];
    for (int i = 0; i < 4; ++i) {
        tmp[i] = A[0 + i] * x[0] + A[4 + i] * x[1] + A[8 + i] * x[2] + A[12 + i];
    }
    float w = 1.0f / tmp[3];
    for (int i = 0; i < 3; ++i) {
        y[i] = tmp[i] * w;
    }
}
static inline void affineTransformVector3(float *y, const float *A, const float *x) {
    float tmp[3];
    for (int i = 0; i < 3; ++i) {
        tmp[i] = A[0 + i] * x[0] + A[4 + i] * x[1] + A[8 + i] * x[2] + A[12 + i];
    }
    for (int i = 0; i < 3; ++i) {
        y[i] = tmp[i];
    }
}
static inline void transposeTransformVector3(float *y, const float *A, const float *x) {
    float tmp[4];
    for (int i = 0; i < 4; ++i) {
        tmp[i] = A[4 * i + 0] * x[0] + A[4 * i + 1] * x[1] + A[4 * i + 2] * x[2] + A[4 * i + 3];
    }
    float w = 1.0f / tmp[3];
    for (int i = 0; i < 3; ++i) {
        y[i] = tmp[i] * w;
    }
}
static inline void setRotateDerivative(float *A, float x, float y, float z, float angle) {
    float l = 1.0f / sqrtf(x * x + y * y + z * z);
    x *= l;
    y *= l;
    z *= l;
    float c = -sinf(angle);
    float s = cosf(angle);
    A[0] = c + x * x * (-c);
    A[4] = x * y * (-c) + z * s;
    A[8] = x * z * (-c) - y * s;
    A[12] = 0;
    A[1] = x * y * (-c) - z * s;
    A[5] = c + y * y * (-c);
    A[9] = y * z * (-c) + x * s;
    A[13] = 0;
    A[2] = x * z * (-c) + y * s;
    A[6] = y * z * (-c) - x * s;
    A[10] = c + z * z * (-c);
    A[14] = 0;
    A[3] = 0;
    A[7] = 0;
    A[11] = 0;
    A[15] = 0;
}
static inline void mulRotateDerivative(float *A, float x, float y, float z, float angle) {
    float tmp[16];
    setRotateDerivative(tmp, x, y, z, angle);
    mul2(A, tmp);
}

static inline void setLookDirection(float *A, const float *eye, const float *dir, const float *up) {
    float s[3];
    cross(s, dir, up);

    float u[3];
    cross(u, s, dir);

    float d[3] = {dir[0], dir[1], dir[2]};

    normalize3(s);
    normalize3(u);
    normalize3(d);

    A[0] = s[0];
    A[4] = s[1];
    A[8] = s[2];
    A[12] = 0;
    A[1] = u[0];
    A[5] = u[1];
    A[9] = u[2];
    A[13] = 0;
    A[2] = -d[0];
    A[6] = -d[1];
    A[10] = -d[2];
    A[14] = 0;
    A[3] = 0;
    A[7] = 0;
    A[11] = 0;
    A[15] = 1;
    mulTranslate(A, -eye[0], -eye[1], -eye[2]);
}
static inline void mulLookDirection(float *A, const float *eye, const float *dir, const float *up) {
    float tmp[16];
    setLookDirection(tmp, eye, dir, up);
    mul2(A, tmp);
}

static inline void setInverseLookDirection(float *A, const float *eye, const float *dir, const float *up) {
    float s[3];
    cross(s, dir, up);

    float u[3];
    cross(u, s, dir);

    float d[3] = {dir[0], dir[1], dir[2]};

    normalize3(s);
    normalize3(u);
    normalize3(d);

    setTranslate(A, eye[0], eye[1], eye[2]);
    float B[16];
    B[0] = s[0];
    B[4] = u[0];
    B[8] = -d[0];
    B[12] = 0;
    B[1] = s[1];
    B[5] = u[1];
    B[9] = -d[1];
    B[13] = 0;
    B[2] = s[2];
    B[6] = u[2];
    B[10] = -d[2];
    B[14] = 0;
    B[3] = 0;
    B[7] = 0;
    B[11] = 0;
    B[15] = 1;
    mul2(A, B);
}

// 0  1  2  3
// 4  5  6  7
// 8  9 10 11
//12 13 14 15

static inline void setNormalTransform(float *A, const float *B) {
    float det =
        B[0]*B[5]*B[10] + B[1]*B[6]*B[8] + B[2]*B[4]*B[9]
        - B[0]*B[6]*B[9]- B[1]*B[4]*B[10] - B[2]*B[5]*B[8];
    det = 1.0f/det;

    A[0] = det*(B[5]*B[10] - B[6]*B[9]);
    A[4] = det*(B[2]*B[9] - B[1]*B[10]);
    A[8] = det*(B[1]*B[6] - B[2]*B[5]);
    A[12] = 0.0f;

    A[1] = det*(B[6]*B[8] - B[4]*B[10]);
    A[5] = det*(B[0]*B[10] - B[2]*B[8]);
    A[9] = det*(B[2]*B[4] - B[0]*B[6]);
    A[13] = 0.0f;

    A[2] = det*(B[4]*B[10] - B[6]*B[8]);
    A[6] = det*(B[1]*B[8] - B[0]*B[9]);
    A[10] = det*(B[0]*B[5] - B[1]*B[4]);
    A[14] = 0.0f;

    A[3] = 0.0f;
    A[7] = 0.0f;
    A[11] = 0.0f;
    A[15] = 1.0f;
}


static inline void setInverse(float *A, const float *B) {
    A[0] =
        B[5]*B[10]*B[15]
        - B[5]*B[11]*B[14]
        - B[9]*B[6]*B[15]
        + B[9]*B[7]*B[14]
        + B[13]*B[6]*B[11]
        - B[13]*B[7]*B[10];

    A[4] =
        -B[4]*B[10]*B[15]
        + B[4]*B[11]*B[14]
        + B[8]*B[6]*B[15]
        - B[8]*B[7]*B[14]
        - B[12]*B[6]*B[11]
        + B[12]*B[7]*B[10];

    A[8] =
        B[4]*B[9]*B[15]
        - B[4]*B[11]*B[13]
        - B[8]*B[5]*B[15]
        + B[8]*B[7]*B[13]
        + B[12]*B[5]*B[11]
        - B[12]*B[7]*B[9];

    A[12] =
        -B[4]*B[9]*B[14]
        + B[4]*B[10]*B[13]
        + B[8]*B[5]*B[14]
        - B[8]*B[6]*B[13]
        - B[12]*B[5]*B[10]
        + B[12]*B[6]*B[9];

    A[1] =
        -B[1]*B[10]*B[15]
        + B[1]*B[11]*B[14]
        + B[9]*B[2]*B[15]
        - B[9]*B[3]*B[14]
        - B[13]*B[2]*B[11]
        + B[13]*B[3]*B[10];

    A[5] =
        B[0]*B[10]*B[15]
        - B[0]*B[11]*B[14]
        - B[8]*B[2]*B[15]
        + B[8]*B[3]*B[14]
        + B[12]*B[2]*B[11]
        - B[12]*B[3]*B[10];

    A[9] =
        -B[0]*B[9]*B[15]
        + B[0]*B[11]*B[13]
        + B[8]*B[1]*B[15]
        - B[8]*B[3]*B[13]
        - B[12]*B[1]*B[11]
        + B[12]*B[3]*B[9];

    A[13] =
        B[0]*B[9]*B[14]
        - B[0]*B[10]*B[13]
        - B[8]*B[1]*B[14]
        + B[8]*B[2]*B[13]
        + B[12]*B[1]*B[10]
        - B[12]*B[2]*B[9];

    A[2] =
        B[1]*B[6]*B[15]
        - B[1]*B[7]*B[14]
        - B[5]*B[2]*B[15]
        + B[5]*B[3]*B[14]
        + B[13]*B[2]*B[7]
        - B[13]*B[3]*B[6];

    A[6] =
        -B[0]*B[6]*B[15]
        + B[0]*B[7]*B[14]
        + B[4]*B[2]*B[15]
        - B[4]*B[3]*B[14]
        - B[12]*B[2]*B[7]
        + B[12]*B[3]*B[6];

    A[10] =
        B[0]*B[5]*B[15]
        - B[0]*B[7]*B[13]
        - B[4]*B[1]*B[15]
        + B[4]*B[3]*B[13]
        + B[12]*B[1]*B[7]
        - B[12]*B[3]*B[5];

    A[14] =
        -B[0]*B[5]*B[14]
        + B[0]*B[6]*B[13]
        + B[4]*B[1]*B[14]
        - B[4]*B[2]*B[13]
        - B[12]*B[1]*B[6]
        + B[12]*B[2]*B[5];

    A[3] =
        -B[1]*B[6]*B[11]
        + B[1]*B[7]*B[10]
        + B[5]*B[2]*B[11]
        - B[5]*B[3]*B[10]
        - B[9]*B[2]*B[7]
        + B[9]*B[3]*B[6];

    A[7] =
        B[0]*B[6]*B[11]
        - B[0]*B[7]*B[10]
        - B[4]*B[2]*B[11]
        + B[4]*B[3]*B[10]
        + B[8]*B[2]*B[7]
        - B[8]*B[3]*B[6];

    A[11] =
        -B[0]*B[5]*B[11]
        + B[0]*B[7]*B[9]
        + B[4]*B[1]*B[11]
        - B[4]*B[3]*B[9]
        - B[8]*B[1]*B[7]
        + B[8]*B[3]*B[5];

    A[15] =
        B[0]*B[5]*B[10]
        - B[0]*B[6]*B[9]
        - B[4]*B[1]*B[10]
        + B[4]*B[2]*B[9]
        + B[8]*B[1]*B[6]
        - B[8]*B[2]*B[5];

    float det = B[0]*A[0] + B[1]*A[4] + B[2]*A[8] + B[3]*A[12];

    if(det != 0) {
        det = 1.0f / det;
    }

    for(int i = 0;i<16;i++) {
        A[i] *= det;
    }
}

static void transformBoundingBox(float *out, const float *transform, const float *bounds) {
    float tmp[6];
    tmp[0] = FLT_MAX;
    tmp[1] = FLT_MAX;
    tmp[2] = FLT_MAX;
    tmp[3] = -FLT_MAX;
    tmp[4] = -FLT_MAX;
    tmp[5] = -FLT_MAX;
    for(int i=0;i<8;++i) {
        float x[3];
        float y[3];

        //enumerate the corners of the bounding box
        x[0] = (i&1) ? bounds[0] : bounds[3];
        x[1] = (i&2) ? bounds[1] : bounds[4];
        x[2] = (i&4) ? bounds[2] : bounds[5];

        //transform it
        affineTransformVector3(y, transform, x);

        // calculate the new bounds
        tmp[0] = fast_minf(tmp[0], y[0]);
        tmp[1] = fast_minf(tmp[1], y[1]);
        tmp[2] = fast_minf(tmp[2], y[2]);
        tmp[3] = fast_maxf(tmp[3], y[0]);
        tmp[4] = fast_maxf(tmp[4], y[1]);
        tmp[5] = fast_maxf(tmp[5], y[2]);
    }
    for(int i = 0;i<6;++i) {
        out[i] = tmp[i];
    }
}

static void foldBoundingBox(float *out, const float *bounds) {
    out[0] = fast_minf(out[0], bounds[0]);
    out[1] = fast_minf(out[1], bounds[1]);
    out[2] = fast_minf(out[2], bounds[2]);
    out[3] = fast_maxf(out[3], bounds[3]);
    out[4] = fast_maxf(out[4], bounds[4]);
    out[5] = fast_maxf(out[5], bounds[5]);
}


static void projectBoundingBox(float *out, const float *bounds, const float *zero, const float *axis) {
    float offset = dot3(zero, axis);
    out[0] = FLT_MAX;
    out[1] = -FLT_MAX;
    for(int i=0;i<8;++i) {
        float x[3];

        //enumerate the corners of the bounding box
        x[0] = (i&1) ? bounds[0] : bounds[3];
        x[1] = (i&2) ? bounds[1] : bounds[4];
        x[2] = (i&4) ? bounds[2] : bounds[5];

        float s = dot3(x, axis)-offset;
        out[0] = fast_minf(s, out[0]);
        out[1] = fast_maxf(s, out[1]);
    }
}


#endif

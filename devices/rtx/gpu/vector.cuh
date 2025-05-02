#pragma once

#include "gpu/gpu_decl.h"

#include <cuda.h>
#include <vector_types.h>

VISRTX_DEVICE ::float2 operator+(
    const ::float2 &a, const ::float2 &b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}
VISRTX_DEVICE ::float2 operator-(const ::float2 &a, float s)
{
  return make_float2(a.x - s, a.y - s);
}
VISRTX_DEVICE ::float2 operator*(const ::float2 &a, float s)
{
  return make_float2(a.x * s, a.y * s);
}
VISRTX_DEVICE ::float2 operator*(float s, const ::float2 &a)
{
  return make_float2(a.x * s, a.y * s);
}

VISRTX_DEVICE ::float3 operator+(
    const ::float3 &a, const ::float3 &b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
VISRTX_DEVICE ::float3 operator-(const ::float3 &a)
{
  return make_float3(-a.x, -a.y, -a.z);
}
VISRTX_DEVICE ::float3 operator-(
    const ::float3 &a, const ::float3 &b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
VISRTX_DEVICE ::float3 operator*(
    const ::float3 &a, const ::float3 &b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
VISRTX_DEVICE ::float3 operator*(const ::float3 &a, float s)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
VISRTX_DEVICE ::float3 operator*(float s, const ::float3 &a)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
VISRTX_DEVICE ::float3 operator/(const ::float3 &a, float s)
{
  float d = 1.0f / s;
  return make_float3(a.x * d, a.y * d, a.z * d);
}
VISRTX_DEVICE void operator*=(
    ::float3 &a, const ::float3 &b)
{
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}
VISRTX_DEVICE void operator*=(::float3 &a, float s)
{
  a.x *= s;
  a.y *= s;
  a.z *= s;
}
VISRTX_DEVICE void operator/=(::float3 &a, float s)
{
  float d = 1.0f / s;
  a.x *= d;
  a.y *= d;
  a.z *= d;
}

VISRTX_DEVICE ::float4 operator+(
    const ::float4 &a, const ::float4 &b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
VISRTX_DEVICE ::float4 operator*(const ::float4 &a, float s)
{
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

VISRTX_DEVICE float length(const ::float3 &d)
{
  return sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
}

VISRTX_DEVICE ::float3 normalize(const ::float3 &d)
{
  const float inv_len = 1.0f / length(d);
  return make_float3(d.x * inv_len, d.y * inv_len, d.z * inv_len);
}

VISRTX_DEVICE float dot(
    const ::float3 &u, const ::float3 &v)
{
  return u.x * v.x + u.y * v.y + u.z * v.z;
}

VISRTX_DEVICE ::float3 cross(
    const ::float3 &a, const ::float3 &b)
{
  return make_float3(
      a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

VISRTX_DEVICE ::float3 lerp(
    const ::float3 &a, const ::float3 &b, float t)
{
  return a + (b - a) * t;
}

VISRTX_DEVICE ::float3 make_float3(float s)
{
  return make_float3(s, s, s);
}

VISRTX_DEVICE ::float4 make_float4(const ::int4 &v0)
{
  return make_float4(float(v0.x), float(v0.y), float(v0.z), float(v0.w));
}

VISRTX_DEVICE ::int4 make_int4(const ::float4 &v0)
{
  return make_int4(int(v0.x), int(v0.y), int(v0.z), int(v0.w));
}

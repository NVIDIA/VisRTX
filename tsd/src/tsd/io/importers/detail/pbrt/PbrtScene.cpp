// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "PbrtScene.hpp"
#include <cmath>

namespace pbrt {

// ParamList ///////////////////////////////////////////////////////////////////

static const std::vector<float> s_emptyFloats;
static const std::vector<int> s_emptyInts;

bool ParamList::has(const std::string &name) const
{
  return values.count(name) > 0;
}

float ParamList::getFloat(const std::string &name, float def) const
{
  auto it = values.find(name);
  if (it == values.end())
    return def;
  auto *v = std::get_if<std::vector<float>>(&it->second);
  if (!v || v->empty())
    return def;
  return (*v)[0];
}

int ParamList::getInt(const std::string &name, int def) const
{
  auto it = values.find(name);
  if (it == values.end())
    return def;
  auto *v = std::get_if<std::vector<int>>(&it->second);
  if (!v || v->empty())
    return def;
  return (*v)[0];
}

std::string ParamList::getString(
    const std::string &name, const std::string &def) const
{
  auto it = values.find(name);
  if (it == values.end())
    return def;
  auto *v = std::get_if<std::vector<std::string>>(&it->second);
  if (!v || v->empty())
    return def;
  return (*v)[0];
}

bool ParamList::getBool(const std::string &name, bool def) const
{
  auto it = values.find(name);
  if (it == values.end())
    return def;
  auto *v = std::get_if<std::vector<bool>>(&it->second);
  if (!v || v->empty())
    return def;
  return (*v)[0];
}

const std::vector<float> &ParamList::getFloats(const std::string &name) const
{
  auto it = values.find(name);
  if (it == values.end())
    return s_emptyFloats;
  auto *v = std::get_if<std::vector<float>>(&it->second);
  if (!v)
    return s_emptyFloats;
  return *v;
}

std::string ParamList::getType(const std::string &name) const
{
  auto it = types.find(name);
  return it == types.end() ? std::string{} : it->second;
}

const std::vector<int> &ParamList::getInts(const std::string &name) const
{
  auto it = values.find(name);
  if (it == values.end())
    return s_emptyInts;
  auto *v = std::get_if<std::vector<int>>(&it->second);
  if (!v)
    return s_emptyInts;
  return *v;
}

// Transform ///////////////////////////////////////////////////////////////////

Transform Transform::identity()
{
  return {};
}

Transform Transform::translate(float x, float y, float z)
{
  Transform t;
  t.m[12] = x;
  t.m[13] = y;
  t.m[14] = z;
  return t;
}

Transform Transform::scale(float x, float y, float z)
{
  Transform t;
  t.m[0] = x;
  t.m[5] = y;
  t.m[10] = z;
  return t;
}

Transform Transform::rotate(float angleDeg, float ax, float ay, float az)
{
  constexpr float DEG_TO_RAD = 3.14159265358979323846f / 180.f;
  float rad = angleDeg * DEG_TO_RAD;
  float c = std::cos(rad);
  float s = std::sin(rad);

  // Normalize axis
  float len = std::sqrt(ax * ax + ay * ay + az * az);
  if (len < 1e-12f)
    return identity();
  ax /= len;
  ay /= len;
  az /= len;

  // Rodrigues' rotation formula
  float ic = 1.f - c;

  Transform t;
  t.m[0] = c + ax * ax * ic;
  t.m[1] = ay * ax * ic + az * s;
  t.m[2] = az * ax * ic - ay * s;
  t.m[3] = 0.f;

  t.m[4] = ax * ay * ic - az * s;
  t.m[5] = c + ay * ay * ic;
  t.m[6] = az * ay * ic + ax * s;
  t.m[7] = 0.f;

  t.m[8] = ax * az * ic + ay * s;
  t.m[9] = ay * az * ic - ax * s;
  t.m[10] = c + az * az * ic;
  t.m[11] = 0.f;

  t.m[12] = 0.f;
  t.m[13] = 0.f;
  t.m[14] = 0.f;
  t.m[15] = 1.f;
  return t;
}

Transform Transform::lookAt(float ex,
    float ey,
    float ez,
    float lx,
    float ly,
    float lz,
    float ux,
    float uy,
    float uz)
{
  // Forward direction (eye -> look-at).
  float dx = lx - ex;
  float dy = ly - ey;
  float dz = lz - ez;
  float dlen = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (dlen < 1e-12f)
    return identity();
  dx /= dlen;
  dy /= dlen;
  dz /= dlen;

  // Right = up x forward (PBRT v4 convention: column 0 of camera-to-world).
  float rx = uy * dz - uz * dy;
  float ry = uz * dx - ux * dz;
  float rz = ux * dy - uy * dx;
  float rlen = std::sqrt(rx * rx + ry * ry + rz * rz);
  if (rlen < 1e-12f)
    return identity();
  rx /= rlen;
  ry /= rlen;
  rz /= rlen;

  // Recompute up = forward x right (orthonormal frame).
  float nux = dy * rz - dz * ry;
  float nuy = dz * rx - dx * rz;
  float nuz = dx * ry - dy * rx;

  // Camera-to-world: columns are right, up, forward, eye.
  Transform t;
  t.m[0] = rx;
  t.m[1] = ry;
  t.m[2] = rz;
  t.m[3] = 0.f;

  t.m[4] = nux;
  t.m[5] = nuy;
  t.m[6] = nuz;
  t.m[7] = 0.f;

  t.m[8] = dx;
  t.m[9] = dy;
  t.m[10] = dz;
  t.m[11] = 0.f;

  t.m[12] = ex;
  t.m[13] = ey;
  t.m[14] = ez;
  t.m[15] = 1.f;
  return t;
}

Transform Transform::operator*(const Transform &rhs) const
{
  Transform result;
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      float sum = 0.f;
      for (int k = 0; k < 4; ++k)
        sum += m[k * 4 + row] * rhs.m[col * 4 + k];
      result.m[col * 4 + row] = sum;
    }
  }
  return result;
}

} // namespace pbrt

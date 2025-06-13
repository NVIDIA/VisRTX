// Copyright 2023-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TSDMath.hpp"
// std
#include <string>

namespace tsd::manipulators {

using UpdateToken = size_t;

struct CameraPose
{
  std::string name;
  tsd::float3 lookat{0.f};
  tsd::float3 azeldist{0.f};
  float fixedDist{tsd::math::inf};
  int upAxis{0};
};

enum class OrbitAxis
{
  POS_X,
  POS_Y,
  POS_Z,
  NEG_X,
  NEG_Y,
  NEG_Z
};

class Orbit
{
 public:
  Orbit() = default;

  void setConfig(const CameraPose &p);
  void setConfig(anari::math::float3 center,
      float dist,
      anari::math::float2 azel = anari::math::float2(0.f));
  void setCenter(anari::math::float3 center);
  void setDistance(float dist);
  void setFixedDistance(float dist);
  void setAzel(anari::math::float2 azel);

  void startNewRotation();

  bool hasChanged(UpdateToken &t) const;

  void rotate(anari::math::float2 delta);
  void zoom(float delta);
  void pan(anari::math::float2 delta);

  void setAxis(OrbitAxis axis);
  OrbitAxis axis() const;

  anari::math::float2 azel() const;

  anari::math::float3 eye() const;
  anari::math::float3 at() const;
  anari::math::float3 dir() const;
  anari::math::float3 up() const;

  float distance() const;
  float fixedDistance() const;

  anari::math::float3 eye_FixedDistance() const;

 protected:
  void update();

  OrbitAxis negateAxis(OrbitAxis current) const;
  anari::math::float3 azelToDirection(float az, float el, OrbitAxis axis) const;

  // Data //

  UpdateToken m_token{1};

  // NOTE: degrees
  anari::math::float2 m_azel{0.f};

  float m_distance{1.f};
  float m_fixedDistance{tsd::math::inf};
  float m_speed{0.25f};

  bool m_invertRotation{false};

  anari::math::float3 m_eye;
  anari::math::float3 m_eyeFixedDistance;
  anari::math::float3 m_at;
  anari::math::float3 m_up;
  anari::math::float3 m_right;

  OrbitAxis m_axis{OrbitAxis::POS_Y};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline void Orbit::setConfig(const CameraPose &p)
{
  setConfig(p.lookat, p.azeldist.z, {p.azeldist.x, p.azeldist.y});
  setAxis(static_cast<tsd::manipulators::OrbitAxis>(p.upAxis));
}

inline void Orbit::setConfig(
    anari::math::float3 center, float dist, anari::math::float2 azel)
{
  m_at = center;
  m_distance = dist;
  m_azel = azel;
  m_speed = dist;
  if (m_fixedDistance == tsd::math::inf)
    m_fixedDistance = dist;
  update();
}

inline void Orbit::setCenter(anari::math::float3 center)
{
  setConfig(center, m_distance, m_azel);
}

inline void Orbit::setDistance(float dist)
{
  setConfig(m_at, dist, m_azel);
}

inline void Orbit::setFixedDistance(float dist)
{
  m_fixedDistance = dist;
}

inline void Orbit::setAzel(anari::math::float2 azel)
{
  setConfig(m_at, m_distance, azel);
}

inline void Orbit::startNewRotation()
{
  m_invertRotation = m_azel.y > 90.f && m_azel.y < 270.f;
}

inline bool Orbit::hasChanged(UpdateToken &t) const
{
  if (t < m_token) {
    t = m_token;
    return true;
  } else
    return false;
}

inline void Orbit::rotate(anari::math::float2 delta)
{
  delta *= 100;
  if (m_axis == OrbitAxis::POS_Z || m_axis == OrbitAxis::NEG_X
      || m_axis == OrbitAxis::NEG_Y)
    delta.x = -delta.x;
  delta.x = m_invertRotation ? -delta.x : delta.x;
  delta.y = m_distance < 0.f ? -delta.y : delta.y;
  m_azel += delta;

  auto maintainUnitCircle = [](float inDegrees) -> float {
    while (inDegrees > 360.f)
      inDegrees -= 360.f;
    while (inDegrees < 0.f)
      inDegrees += 360.f;
    return inDegrees;
  };

  m_azel.x = maintainUnitCircle(m_azel.x);
  m_azel.y = maintainUnitCircle(m_azel.y);
  update();
}

inline void Orbit::zoom(float delta)
{
  m_distance -= m_speed * delta;
  update();
}

inline void Orbit::pan(anari::math::float2 delta)
{
  delta *= m_speed;

  const anari::math::float3 amount = -delta.x * m_right + delta.y * m_up;

  m_eye += amount;
  m_at += amount;

  update();
}

inline void Orbit::setAxis(OrbitAxis axis)
{
  m_axis = axis;
  update();
}

inline OrbitAxis Orbit::axis() const
{
  return m_axis;
}

inline anari::math::float2 Orbit::azel() const
{
  return m_azel;
}

inline anari::math::float3 Orbit::eye() const
{
  return m_eye;
}

inline anari::math::float3 Orbit::at() const
{
  return m_at;
}

inline anari::math::float3 Orbit::dir() const
{
  return linalg::normalize(at() - eye());
}

inline anari::math::float3 Orbit::up() const
{
  return m_up;
}

inline float Orbit::distance() const
{
  return m_distance;
}

inline float Orbit::fixedDistance() const
{
  return m_fixedDistance;
}

inline anari::math::float3 Orbit::eye_FixedDistance() const
{
  return m_eyeFixedDistance;
}

inline void Orbit::update()
{
  const float distance = std::abs(m_distance);

  const OrbitAxis axis = m_distance < 0.f ? negateAxis(m_axis) : m_axis;

  const float azimuth = tsd::math::radians(-m_azel.x);
  const float elevation = tsd::math::radians(-m_azel.y);

  const anari::math::float3 toLocalOrbit =
      azelToDirection(azimuth, elevation, axis);

  const anari::math::float3 localOrbitPos = toLocalOrbit * distance;
  const anari::math::float3 fromLocalOrbit = -localOrbitPos;

  const anari::math::float3 alteredElevation =
      azelToDirection(azimuth, elevation + 3, m_axis);

  const anari::math::float3 cameraRight =
      linalg::cross(toLocalOrbit, alteredElevation);
  const anari::math::float3 cameraUp =
      linalg::cross(cameraRight, fromLocalOrbit);

  m_eye = localOrbitPos + m_at;
  m_up = linalg::normalize(cameraUp);
  m_right = linalg::normalize(cameraRight);

  m_eyeFixedDistance = (toLocalOrbit * m_fixedDistance) + m_at;

  m_token++;
}

inline OrbitAxis Orbit::negateAxis(OrbitAxis current) const
{
  switch (current) {
  case OrbitAxis::POS_X:
    return OrbitAxis::NEG_X;
  case OrbitAxis::POS_Y:
    return OrbitAxis::NEG_Y;
  case OrbitAxis::POS_Z:
    return OrbitAxis::NEG_Z;
  case OrbitAxis::NEG_X:
    return OrbitAxis::POS_X;
  case OrbitAxis::NEG_Y:
    return OrbitAxis::POS_Y;
  case OrbitAxis::NEG_Z:
    return OrbitAxis::POS_Z;
  }
  return {};
}

inline anari::math::float3 Orbit::azelToDirection(
    float az, float el, OrbitAxis axis) const
{
  const float x = std::sin(az) * std::cos(el);
  const float y = std::cos(az) * std::cos(el);
  const float z = std::sin(el);
  switch (axis) {
  case OrbitAxis::POS_X:
    return -normalize(anari::math::float3(z, y, x));
  case OrbitAxis::POS_Y:
    return -normalize(anari::math::float3(x, z, y));
  case OrbitAxis::POS_Z:
    return -normalize(anari::math::float3(x, y, z));
  case OrbitAxis::NEG_X:
    return normalize(anari::math::float3(z, y, x));
  case OrbitAxis::NEG_Y:
    return normalize(anari::math::float3(x, z, y));
  case OrbitAxis::NEG_Z:
    return normalize(anari::math::float3(x, y, z));
  }
  return {};
}

} // namespace tsd::manipulators

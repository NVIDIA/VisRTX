// Copyright 2023-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TSDMath.hpp"
// std
#include <string>

namespace tsd::rendering {

using UpdateToken = size_t;

struct CameraPose
{
  std::string name;
  tsd::math::float3 lookat{0.f};
  tsd::math::float3 azeldist{0.f};
  float fixedDist{tsd::math::inf};
  int upAxis{0};
};

enum class UpAxis
{
  POS_X,
  POS_Y,
  POS_Z,
  NEG_X,
  NEG_Y,
  NEG_Z
};

class Manipulator
{
 public:
  Manipulator() = default;

  void setConfig(const CameraPose &p);
  void setConfig(anari::math::float3 center,
      float dist,
      anari::math::float2 azel = anari::math::float2(0.f));
  void setCenter(anari::math::float3 center);
  void setDistance(float dist);
  void setFixedDistance(float dist);
  void setAzel(anari::math::float2 azel);

  void setZoomSpeed(float speed);
  float zoomSpeed() const;

  void startNewRotation();

  bool hasChanged(UpdateToken &t) const;

  void rotate(anari::math::float2 delta);
  void zoom(float delta);
  void pan(anari::math::float2 delta);

  void setAxis(UpAxis axis);
  UpAxis axis() const;

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

  UpAxis negateAxis(UpAxis current) const;
  anari::math::float3 azelToDirection(float az, float el, UpAxis axis) const;

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

  UpAxis m_axis{UpAxis::POS_Y};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline void Manipulator::setConfig(const CameraPose &p)
{
  setConfig(p.lookat, p.azeldist.z, {p.azeldist.x, p.azeldist.y});
  setAxis(static_cast<UpAxis>(p.upAxis));
}

inline void Manipulator::setConfig(
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

inline void Manipulator::setCenter(anari::math::float3 center)
{
  setConfig(center, m_distance, m_azel);
}

inline void Manipulator::setDistance(float dist)
{
  setConfig(m_at, dist, m_azel);
}

inline void Manipulator::setFixedDistance(float dist)
{
  m_fixedDistance = dist;
}

inline void Manipulator::setAzel(anari::math::float2 azel)
{
  setConfig(m_at, m_distance, azel);
}

inline void Manipulator::setZoomSpeed(float speed)
{
  m_speed = speed;
}

inline float Manipulator::zoomSpeed() const
{
  return m_speed;
}

inline void Manipulator::startNewRotation()
{
  m_invertRotation = m_azel.y > 90.f && m_azel.y < 270.f;
}

inline bool Manipulator::hasChanged(UpdateToken &t) const
{
  if (t < m_token) {
    t = m_token;
    return true;
  } else
    return false;
}

inline void Manipulator::rotate(anari::math::float2 delta)
{
  delta *= 100;
  if (m_axis == UpAxis::POS_Z || m_axis == UpAxis::NEG_X
      || m_axis == UpAxis::NEG_Y)
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

inline void Manipulator::zoom(float delta)
{
  m_distance -= m_speed * delta;
  update();
}

inline void Manipulator::pan(anari::math::float2 delta)
{
  delta *= m_speed;

  const anari::math::float3 amount = -delta.x * m_right + delta.y * m_up;

  m_eye += amount;
  m_at += amount;

  update();
}

inline void Manipulator::setAxis(UpAxis axis)
{
  m_axis = axis;
  update();
}

inline UpAxis Manipulator::axis() const
{
  return m_axis;
}

inline anari::math::float2 Manipulator::azel() const
{
  return m_azel;
}

inline anari::math::float3 Manipulator::eye() const
{
  return m_eye;
}

inline anari::math::float3 Manipulator::at() const
{
  return m_at;
}

inline anari::math::float3 Manipulator::dir() const
{
  return linalg::normalize(at() - eye());
}

inline anari::math::float3 Manipulator::up() const
{
  return m_up;
}

inline float Manipulator::distance() const
{
  return m_distance;
}

inline float Manipulator::fixedDistance() const
{
  return m_fixedDistance;
}

inline anari::math::float3 Manipulator::eye_FixedDistance() const
{
  return m_eyeFixedDistance;
}

inline void Manipulator::update()
{
  const float distance = std::abs(m_distance);

  const UpAxis axis = m_distance < 0.f ? negateAxis(m_axis) : m_axis;

  const float azimuth = tsd::math::radians(-m_azel.x);
  const float elevation = tsd::math::radians(-m_azel.y);

  const anari::math::float3 toLocalManipulator =
      azelToDirection(azimuth, elevation, axis);

  const anari::math::float3 localManipulatorPos = toLocalManipulator * distance;
  const anari::math::float3 fromLocalManipulator = -localManipulatorPos;

  const anari::math::float3 alteredElevation =
      azelToDirection(azimuth, elevation + 3, m_axis);

  const anari::math::float3 cameraRight =
      linalg::cross(toLocalManipulator, alteredElevation);
  const anari::math::float3 cameraUp =
      linalg::cross(cameraRight, fromLocalManipulator);

  m_eye = localManipulatorPos + m_at;
  m_up = linalg::normalize(cameraUp);
  m_right = linalg::normalize(cameraRight);

  m_eyeFixedDistance = (toLocalManipulator * m_fixedDistance) + m_at;

  m_token++;
}

inline UpAxis Manipulator::negateAxis(UpAxis current) const
{
  switch (current) {
  case UpAxis::POS_X:
    return UpAxis::NEG_X;
  case UpAxis::POS_Y:
    return UpAxis::NEG_Y;
  case UpAxis::POS_Z:
    return UpAxis::NEG_Z;
  case UpAxis::NEG_X:
    return UpAxis::POS_X;
  case UpAxis::NEG_Y:
    return UpAxis::POS_Y;
  case UpAxis::NEG_Z:
    return UpAxis::POS_Z;
  }
  return {};
}

inline anari::math::float3 Manipulator::azelToDirection(
    float az, float el, UpAxis axis) const
{
  const float x = std::sin(az) * std::cos(el);
  const float y = std::cos(az) * std::cos(el);
  const float z = std::sin(el);
  switch (axis) {
  case UpAxis::POS_X:
    return -normalize(anari::math::float3(z, y, x));
  case UpAxis::POS_Y:
    return -normalize(anari::math::float3(x, z, y));
  case UpAxis::POS_Z:
    return -normalize(anari::math::float3(x, y, z));
  case UpAxis::NEG_X:
    return normalize(anari::math::float3(z, y, x));
  case UpAxis::NEG_Y:
    return normalize(anari::math::float3(x, z, y));
  case UpAxis::NEG_Z:
    return normalize(anari::math::float3(x, y, z));
  }
  return {};
}

} // namespace tsd::rendering

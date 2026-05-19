// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Manipulator.hpp"

#include <cmath>

namespace tsd::rendering {

void Manipulator::setConfig(const CameraPose &p)
{
  m_mode = static_cast<ManipulatorMode>(p.mode);
  m_axis = static_cast<UpAxis>(p.upAxis);
  setConfig(p.lookat, p.azeldist.z, {p.azeldist.x, p.azeldist.y});
}

void Manipulator::setConfig(
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

void Manipulator::setCenter(anari::math::float3 center)
{
  if (m_mode == ManipulatorMode::Look) {
    const auto eye = m_eye;
    m_at = center;
    const auto eyeToAt = eye - m_at;
    const auto d = linalg::length(eyeToAt);
    if (d > 0.f) {
      m_distance = d;
      m_speed = d;
      m_azel = directionToAzel(linalg::normalize(eyeToAt), m_axis);
    }
    m_eye = eye;
    update();
    return;
  }
  setConfig(center, m_distance, m_azel);
}

void Manipulator::setDistance(float dist)
{
  m_distance = dist;
  update();
}

void Manipulator::setFixedDistance(float dist)
{
  m_fixedDistance = dist;
  update();
}

void Manipulator::setAzel(anari::math::float2 azel)
{
  m_azel = azel;
  update();
}

void Manipulator::setMode(ManipulatorMode mode)
{
  if (m_mode == mode)
    return;

  m_mode = mode;
  m_token++;
}

void Manipulator::setZoomSpeed(float speed)
{
  m_speed = speed;
}

float Manipulator::zoomSpeed() const
{
  return m_speed;
}

void Manipulator::startNewRotation()
{
  m_invertRotation = m_azel.y > 90.f && m_azel.y < 270.f;
}

bool Manipulator::hasChanged(UpdateToken &t) const
{
  return tsd::core::versionChanged(t, m_token);
}

void Manipulator::rotate(anari::math::float2 delta)
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

void Manipulator::zoom(float delta)
{
  m_distance -= m_speed * delta;
  update();
}

void Manipulator::pan(anari::math::float2 delta)
{
  delta *= m_speed;

  const anari::math::float3 amount = -delta.x * m_right + delta.y * m_up;

  m_eye += amount;
  m_at += amount;

  update();
}

void Manipulator::setAxis(UpAxis axis)
{
  m_axis = axis;
  update();
}

UpAxis Manipulator::axis() const
{
  return m_axis;
}

ManipulatorMode Manipulator::mode() const
{
  return m_mode;
}

anari::math::float2 Manipulator::azel() const
{
  return m_azel;
}

anari::math::float3 Manipulator::eye() const
{
  return m_eye;
}

anari::math::float3 Manipulator::at() const
{
  return m_at;
}

anari::math::float3 Manipulator::dir() const
{
  return linalg::normalize(at() - eye());
}

anari::math::float3 Manipulator::up() const
{
  return m_up;
}

float Manipulator::distance() const
{
  return m_distance;
}

float Manipulator::fixedDistance() const
{
  return m_fixedDistance;
}

anari::math::float3 Manipulator::eye_FixedDistance() const
{
  return m_eyeFixedDistance;
}

void Manipulator::update()
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

  if (m_mode == ManipulatorMode::Look)
    m_at = m_eye - localManipulatorPos;
  else
    m_eye = localManipulatorPos + m_at;
  m_up = linalg::normalize(cameraUp);
  m_right = linalg::normalize(cameraRight);

  m_eyeFixedDistance = (toLocalManipulator * m_fixedDistance) + m_at;

  m_token++;
}

UpAxis Manipulator::negateAxis(UpAxis current) const
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

anari::math::float3 Manipulator::azelToDirection(
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

anari::math::float2 Manipulator::directionToAzel(
    anari::math::float3 direction, UpAxis axis) const
{
  float az = 0.f;
  float el = 0.f;

  switch (axis) {
  case UpAxis::POS_Y: {
    const anari::math::float3 d = -direction;
    el = std::asin(d.y);
    az = std::atan2(d.x, d.z);
    break;
  }
  case UpAxis::NEG_Y: {
    const anari::math::float3 d = direction;
    el = std::asin(d.y);
    az = std::atan2(d.x, d.z);
    break;
  }
  case UpAxis::POS_Z: {
    const anari::math::float3 d = -direction;
    el = std::asin(d.z);
    az = std::atan2(d.x, d.y);
    break;
  }
  case UpAxis::NEG_Z: {
    const anari::math::float3 d = direction;
    el = std::asin(d.z);
    az = std::atan2(d.x, d.y);
    break;
  }
  case UpAxis::POS_X: {
    const anari::math::float3 d = -direction;
    el = std::asin(d.x);
    az = std::atan2(d.z, d.y);
    break;
  }
  case UpAxis::NEG_X: {
    const anari::math::float3 d = direction;
    el = std::asin(d.x);
    az = std::atan2(d.z, d.y);
    break;
  }
  }

  return {-tsd::math::degrees(az), -tsd::math::degrees(el)};
}

} // namespace tsd::rendering

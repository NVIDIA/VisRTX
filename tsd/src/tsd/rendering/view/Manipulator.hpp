// Copyright 2023-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ObjectVersion.hpp"
#include "tsd/core/TSDMath.hpp"
// std
#include <string>

namespace tsd::rendering {

using UpdateToken = tsd::core::ObjectVersion;

enum class UpAxis
{
  POS_X,
  POS_Y,
  POS_Z,
  NEG_X,
  NEG_Y,
  NEG_Z
};

enum class ManipulatorMode
{
  Orbit,
  Look
};

/*
 * Value type that captures a named camera viewpoint as az/el/distance
 * spherical coordinates relative to a look-at center, plus an up-axis.
 *
 * Example:
 *   CameraPose p{"front", {0,0,0}, {0, 0, 5.f}};
 *   manipulator.setConfig(p);
 */
struct CameraPose
{
  std::string name;
  tsd::math::float3 lookat{0.f};
  tsd::math::float3 azeldist{0.f};
  float fixedDist{tsd::math::inf};
  int upAxis{static_cast<int>(UpAxis::POS_Y)};
  int mode{static_cast<int>(ManipulatorMode::Orbit)};
};

/*
 * Orbit-style camera controller that tracks az/el/distance to a center point
 * and exposes rotate/zoom/pan operations for interactive or scripted
 * navigation.
 *
 * Example:
 *   Manipulator m;
 *   m.setConfig({0,0,0}, 5.f);
 *   m.rotate({dx, dy});
 *   m.zoom(scrollDelta);
 *   anari::setParameter(d, cam, "position", m.eye());
 */
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
  void setMode(ManipulatorMode mode);

  void setZoomSpeed(float speed);
  float zoomSpeed() const;

  void startNewRotation();

  bool hasChanged(UpdateToken &t) const;

  void rotate(anari::math::float2 delta);
  void zoom(float delta);
  void pan(anari::math::float2 delta);

  void setAxis(UpAxis axis);
  UpAxis axis() const;
  ManipulatorMode mode() const;

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
  anari::math::float2 directionToAzel(
      anari::math::float3 direction, UpAxis axis) const;

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
  ManipulatorMode m_mode{ManipulatorMode::Orbit};
};

} // namespace tsd::rendering

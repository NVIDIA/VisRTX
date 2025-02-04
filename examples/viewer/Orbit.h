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

// glm
#include <glm/glm.hpp>

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
  Orbit(glm::vec3 at = glm::vec3(0.f),
      float dist = 1.f,
      glm::vec2 azel = glm::vec2(0.f));

  void startNewRotation();

  void setAzel(glm::vec2 azel);

  void rotate(glm::vec2 delta);
  void zoom(float delta);
  void pan(glm::vec2 delta);

  void setAxis(OrbitAxis axis);

  glm::vec2 azel() const;

  glm::vec3 eye() const;
  glm::vec3 dir() const;
  glm::vec3 up() const;

  float distance() const;

  glm::vec3 eye_FixedDistance() const; // using original distance

 protected:
  void update();

  // Data //

  // NOTE: degrees
  glm::vec2 m_azel{0.f};

  float m_distance{1.f};
  float m_originalDistance{1.f};
  float m_speed{0.25f};

  bool m_invertRotation{false};

  glm::vec3 m_eye;
  glm::vec3 m_eyeFixedDistance;
  glm::vec3 m_at;
  glm::vec3 m_up;
  glm::vec3 m_right;

  OrbitAxis m_axis{OrbitAxis::POS_Y};
};

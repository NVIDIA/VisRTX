// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CameraUpdateDelegate.hpp"

// std
#include <cassert>
#include <utility>

namespace tsd::core {

CameraUpdateDelegate::CameraUpdateDelegate(Camera *camera) : m_camera(camera)
{
  m_token = 1;
  if (m_camera)
    m_camera->setUpdateDelegate(this);
}

CameraUpdateDelegate::~CameraUpdateDelegate()
{
  if (m_camera)
    m_camera->setUpdateDelegate(nullptr);
}

void CameraUpdateDelegate::signalParameterUpdated(
    const Object *o, const Parameter *p)
{
  m_token++;
}

bool CameraUpdateDelegate::hasChanged(UpdateToken &t) const
{
  if (t < m_token) {
    t = m_token;
    return true;
  }
  return false;
}

void CameraUpdateDelegate::detach()
{
  if (m_camera)
    m_camera->setUpdateDelegate(nullptr);
  m_camera = nullptr;
}

} // namespace tsd::core

// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CameraUpdateDelegate.hpp"

// std
#include <cassert>
#include <utility>


namespace tsd::core {

CameraUpdateDelegate::CameraUpdateDelegate(Camera *camera)
  : m_camera(camera) {
  m_token = 1;
  if (m_camera)
    m_camera->setUpdateDelegate(this);
}

CameraUpdateDelegate::~CameraUpdateDelegate() {
  if (m_camera)
    m_camera->setUpdateDelegate(nullptr);
}

void CameraUpdateDelegate::signalParameterUpdated(const Object *o, const Parameter *p) {
  m_token += (m_ignoreChangesScope == 0) ? 1 : 0;
}

bool CameraUpdateDelegate::hasChanged(UpdateToken &t) const {
  if (t < m_token) {
    t = m_token;
    return true;
  }
  return false;
}

void CameraUpdateDelegate::detach() {
  if (m_camera)
    m_camera->setUpdateDelegate(nullptr);
  m_camera = nullptr;
}

void CameraUpdateDelegate::pushIgnoreChangeScope() {
  m_ignoreChangesScope++;
}

void CameraUpdateDelegate::popIgnoreChangeScope() {
  assert(m_ignoreChangesScope > 0 && "Mismatched popIgnoreChangeScope call");
  m_ignoreChangesScope--;
}

} // namespace tsd::core

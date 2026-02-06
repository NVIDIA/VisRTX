// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <utility>
#include "tsd/core/scene/objects/Camera.hpp"

namespace tsd::core {

class CameraUpdateDelegate : public EmptyUpdateDelegate
{
 public:
  using UpdateToken = size_t;
  using DeletedCallback = std::function<void()>;

  CameraUpdateDelegate(Camera *camera);
  ~CameraUpdateDelegate();

  void signalParameterUpdated(const Object *o, const Parameter *p) override;

  bool hasChanged(UpdateToken &t) const;
  void detach();

 private:
  Camera *m_camera{nullptr};
  UpdateToken m_token{1};
};

} // namespace tsd::core

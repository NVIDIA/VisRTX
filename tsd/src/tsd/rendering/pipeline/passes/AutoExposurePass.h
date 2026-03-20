// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"

namespace tsd::rendering {

struct AutoExposurePass : public ImagePass
{
  AutoExposurePass();
  ~AutoExposurePass() override;
  const char *name() const override { return "Auto Exposure"; }

  void setHDREnabled(bool enabled);
  float currentExposure() const;

 private:
  void render(ImageBuffers &b, int stageId) override;

  bool m_hdrEnabled{false};
  bool m_hasExposure{false};
  float m_currentExposure{0.f};
  float m_response{0.15f};
};

} // namespace tsd::rendering

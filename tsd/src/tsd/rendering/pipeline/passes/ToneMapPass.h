// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"

namespace tsd::rendering {

enum class ToneMapOperator
{
  NONE,
  REINHARD,
  ACES,
  HABLE,
  KHRONOS_PBR_NEUTRAL,
  AGX
};

struct ToneMapPass : public ImagePass
{
  ToneMapPass();
  ~ToneMapPass() override;
  const char *name() const override
  {
    return "Tone Map";
  }

  void setOperator(ToneMapOperator op);
  void setAutoExposureEnabled(bool enabled);
  void setExposure(float exposure);
  void setHDREnabled(bool enabled);

 protected:
  void render(ImageBuffers &b, int stageId) override;

 private:
  ToneMapOperator m_operator{ToneMapOperator::ACES};
  bool m_autoExposureEnabled{false};
  float m_exposure{0.f};
  bool m_hdrEnabled{false};
};

} // namespace tsd::rendering

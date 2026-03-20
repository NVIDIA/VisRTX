// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImagePass.h"
// anari
#include <anari/frontend/anari_enums.h>

namespace tsd::rendering {

struct OutputTransformPass : public ImagePass
{
  OutputTransformPass();
  ~OutputTransformPass() override;
  const char *name() const override
  {
    return "Output Transform";
  }

  void setColorFormat(anari::DataType format);
  void setGamma(float gamma);

 protected:
  void render(ImageBuffers &b, int stageId) override;

 private:
  anari::DataType m_colorFormat{ANARI_UFIXED8_RGBA_SRGB};
  float m_gamma{2.2f};
};

} // namespace tsd::rendering

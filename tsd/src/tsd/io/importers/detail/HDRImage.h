// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <string>
#include <vector>

namespace tsd::io {

struct HDRImage
{
  bool import(std::string fileName);

  unsigned width;
  unsigned height;
  unsigned numComponents;
  std::vector<float> pixel;
};

} // namespace tsd

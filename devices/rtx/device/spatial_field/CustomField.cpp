/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "CustomField.h"

namespace visrtx {

void CustomField::finalize()
{
  // gpuData() virtual-dispatches to the concrete custom field, which has
  // already uploaded its device data and called m_uniformGrid.init(...).
  m_uniformGrid.computeValueRanges(gpuData());

  // Upload gpuData() into the device registry. Must run after the grid is
  // populated so the snapshot includes the macrocell pointers. Built-in fields
  // do the same at the end of their finalize(); omitting it leaves
  // registry.fields[index] uninitialized → garbage samplerCallableIndex/grid
  // on the GPU → illegal access when the volume is sampled.
  upload();
}

} // namespace visrtx

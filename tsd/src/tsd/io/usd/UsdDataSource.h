// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// usd
#include <pxr/base/vt/array.h>
#include <pxr/imaging/hd/dataSourceTypeDefs.h>

namespace tsd::io::usd {

// Read an int-array data source, or an empty array when it is absent. Every
// Hydra schema hands topology out this way, and every reader of one wants the
// same "absent is empty" answer.
inline pxr::VtIntArray intArrayOf(const pxr::HdIntArrayDataSourceHandle &source)
{
  return source ? source->GetTypedValue(0) : pxr::VtIntArray();
}

} // namespace tsd::io::usd

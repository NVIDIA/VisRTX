// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <mi/base/uuid.h>

namespace visrtx::libmdl {
using Uuid = mi::base::Uuid;

struct UuidHasher
{
  std::size_t operator()(const mi::base::Uuid &uuid) const noexcept
  {
    return mi::base::uuid_hash32(uuid);
  }
};

} // namespace visrtx::libmdl

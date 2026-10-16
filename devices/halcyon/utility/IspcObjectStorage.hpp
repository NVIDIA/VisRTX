// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace halcyon {

template <int SIZE = 256>
struct alignas(16) IspcObjectStorage
{
  char data[SIZE];

  operator void *() const
  {
    return (void *)data;
  }

  constexpr int size() const
  {
    return SIZE;
  }
};

} // namespace halcyon

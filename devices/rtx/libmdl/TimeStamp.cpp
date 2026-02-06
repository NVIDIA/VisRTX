// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "TimeStamp.h"

namespace visrtx::libmdl {

TimeStamp newTimeStamp()
{
  static TimeStamp ts = 0;

  return ++ts;
}

} // namespace visrtx::libmdl
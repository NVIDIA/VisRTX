// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tsd::io {

enum class ArchiveObjectPolicy
{
  All,
  LightsOnly
};

enum class FileBindingArchivePolicy
{
  Include,
  Omit
};

} // namespace tsd::io

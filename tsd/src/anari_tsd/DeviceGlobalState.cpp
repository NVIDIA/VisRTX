// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DeviceGlobalState.h"

namespace tsd_device {

DeviceGlobalState::DeviceGlobalState(anari::Device d)
    : helium::BaseGlobalDeviceState(d)
{}

bool DeviceGlobalState::usingExternalScene() const
{
  return scene != &localScene;
}

} // namespace tsd_device
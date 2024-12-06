// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DeviceGlobalState.h"

namespace tsd_device {

DeviceGlobalState::DeviceGlobalState(ANARIDevice d)
    : helium::BaseGlobalDeviceState(d)
{}

} // namespace tsd_device
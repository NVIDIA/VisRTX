// Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Core.h"

#include <string>
#include <string_view>
#include <tuple>

using namespace std::string_literals;

namespace visrtx::libmdl {

std::tuple<std::string, std::string> parseMaterialSourceName(
    std::string_view argument, Core *logger);

} // namespace visrtx::libmdl

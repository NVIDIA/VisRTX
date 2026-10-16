// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// anari
#ifdef __cplusplus
#include <anari/anari_cpp.hpp>
#else
#include <anari/anari.h>
#endif
// halcyon
#include "anari_library_halcyon_export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Create a halcyon device directly, for applications that link the device
// library instead of loading it with anariLoadLibrary()
HALCYON_EXPORT ANARIDevice anariNewHalcyonDevice(
    ANARIStatusCallback defaultCallback ANARI_DEFAULT_VAL(0),
    const void *userPtr ANARI_DEFAULT_VAL(0));

#ifdef __cplusplus
} // extern "C"
#endif

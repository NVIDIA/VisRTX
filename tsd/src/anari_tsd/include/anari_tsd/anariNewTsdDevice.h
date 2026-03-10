// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ANARI-SDK
#ifdef __cplusplus
#include <anari/anari_cpp.hpp>
#else
#include <anari/anari.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

ANARIDevice anariNewTsdDevice(
    ANARIStatusCallback defaultCallback ANARI_DEFAULT_VAL(0),
    const void *userPtr ANARI_DEFAULT_VAL(0));

#ifdef __cplusplus
} // extern "C"
#endif

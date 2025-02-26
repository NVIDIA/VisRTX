// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "glad/gl.h"

// undef problematic defines pulled in from windows headers
#ifdef far
#undef far
#endif

#ifdef near
#undef near
#endif

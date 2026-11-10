# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Format the complete production header after reference columns agree."""


def table_header(notice, columns):
    text = notice + "#pragma once\n\n"
    text += "// Included by owenSobol.h inside namespace visrtx after uint32_t is available.\n"
    text += "// Storage is immutable host data or CUDA constant memory, per compilation pass.\n"
    text += "// Derived: pbrt-v4 b0bdf461ba5dd6504b70ebf0932acc0fe7aa744f\n"
    text += "// Rows 0..15, columns 0..31 (source stride 52). See apps/tests/unit/sobol.\n"
    text += "// clang-format off\n#ifdef __CUDA_ARCH__\n__device__ __constant__\n#endif\n"
    text += "static const uint32_t kSobolDirection[16][32] = {\n"
    for row in columns:
        text += "    {\n"
        for i in range(0, 32, 4):
            text += "        " + ", ".join(f"0x{x:08x}u" for x in row[i:i+4]) + ",\n"
        text += "    },\n"
    text += "};\n// clang-format on\n"
    return text

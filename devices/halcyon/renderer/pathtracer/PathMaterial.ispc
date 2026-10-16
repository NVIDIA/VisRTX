// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "PathMaterial.isph"

// Exported so that ispc writes the records and flags into PathMaterial_ispc.h,
// which the C++ side (PathMaterial.h) reads them from
export void PathMaterial_layouts(uniform PathShadeFlag,
    uniform PathMaterialFlag,
    uniform PathSampleFlag,
    uniform PathSoADerivative,
    uniform PathContextGroup,
    uniform MaterialKind,
    uniform Attribute,
    uniform PathMaterialContext *uniform,
    uniform PathShadeInSoA *uniform,
    uniform PathShadeOutSoA *uniform,
    uniform PathEvalInSoA *uniform,
    uniform PathEvalOutSoA *uniform,
    uniform PathSubsurfaceInSoA *uniform,
    uniform PathSubsurfaceOutSoA *uniform,
    uniform PathHandleSoA *uniform,
    uniform PathMediumSoA *uniform,
    uniform PathShadowOutSoA *uniform)
{}


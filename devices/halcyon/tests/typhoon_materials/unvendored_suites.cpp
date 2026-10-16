// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Typhoon's test runner (testMaterialXCppMain.cpp) registers every suite,
// including those that need OpenUSD, OpenImageIO or the rest of Typhoon's
// renderer, and the BSDF suite, which includes its integrator (see
// external/typhoon/PROVENANCE.md). They register no tests here,
// so the runner stays unmodified.

void Test_RegisterBsdfTests() {}
void Test_RegisterAdapterTests() {}
void Test_RegisterClosureClassificationTests() {}
void Test_RegisterTextureSystemTests() {}

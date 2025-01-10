// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/device_ptr.h>

namespace tsd {

// Use existing GPU grids
void jacobi3D(
    int nx, int ny, int nz, float *d_grid, float *d_old_grid, int iterations);

// Use host grid
void jacobi3D(int nx, int ny, int nz, float *h_grid, int iterations);

} // namespace tsd

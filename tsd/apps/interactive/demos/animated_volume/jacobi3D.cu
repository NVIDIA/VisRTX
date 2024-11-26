// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// thrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
// std
#include <cmath>

#define GPU_FCN __host__ __device__

namespace tsd {

struct GridIndexer
{
  int nx, ny, nz;
  GPU_FCN GridIndexer(int nx, int ny, int nz) : nx(nx), ny(ny), nz(nz) {}
  GPU_FCN int operator()(int x, int y, int z) const
  {
    return x + nx * (y + ny * z);
  }
  GPU_FCN bool isBoundary(int x, int y, int z) const
  {
    return x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0
        || z == nz - 1;
  }
};

struct JacobiStep
{
  thrust::device_ptr<float> grid, old_grid;
  const int nx, ny, nz;
  const GridIndexer index;

  GPU_FCN JacobiStep(thrust::device_ptr<float> grid,
      thrust::device_ptr<float> old_grid,
      int nx,
      int ny,
      int nz)
      : grid(grid),
        old_grid(old_grid),
        nx(nx),
        ny(ny),
        nz(nz),
        index(nx, ny, nz)
  {}

  GPU_FCN void operator()(size_t idx) const
  {
    const size_t z = idx / (nx * ny);
    const size_t y = (idx % (nx * ny)) / nx;
    const size_t x = idx % nx;

    if (index.isBoundary(x, y, z)) {
      grid[idx] = old_grid[idx];
      return;
    }

    grid[idx] = 0.1666f
        * (old_grid[index(x + 1, y, z)] + old_grid[index(x - 1, y, z)]
            + old_grid[index(x, y + 1, z)] + old_grid[index(x, y - 1, z)]
            + old_grid[index(x, y, z + 1)] + old_grid[index(x, y, z - 1)]);
  }
};

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void jacobi3D(int nx, int ny, int nz, float *h_grid, int iterations)
{
  const size_t grid_size = nx * ny * nz;

  thrust::device_vector<float> d_grid(grid_size);
  thrust::device_vector<float> d_old_grid(grid_size);

  thrust::copy(h_grid, h_grid + grid_size, d_grid.begin());
  d_old_grid = d_grid;

  for (int iter = 0; iter < iterations; ++iter) {
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator(size_t(0u)),
        thrust::make_counting_iterator(grid_size),
        JacobiStep(d_grid.data(), d_old_grid.data(), nx, ny, nz));

    thrust::swap(d_grid, d_old_grid);
  }

  if (iterations % 2)
    thrust::swap(d_grid, d_old_grid);

  thrust::copy(d_grid.begin(), d_grid.end(), h_grid);
}

} // namespace tsd

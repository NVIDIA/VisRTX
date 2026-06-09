/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "../utility/CudaImageTexture.h"

// cuda
#include <cuda_runtime.h>

// std
#include <vector>

namespace visrtx {

/**
 * Utility to build a backmapping LUT for rectilinear coordinates.
 *
 * Given a forward axis array (monotonically increasing object-space
 * coordinates), this builds an 8192-texel 1D CUDA texture that maps normalized
 * object-space coordinates [0,1] to normalized index-space coordinates [0,1].
 *
 * Uses linear interpolation to invert the piecewise-linear forward mapping.
 */
class RectilinearLUT
{
 public:
  // Both builders divide by per-segment and total coordinate gaps; a duplicate
  // or decreasing coordinate would produce NaN/inf LUT texels. Enforce the
  // strictly-increasing contract up front so callers fail fast (their finalize
  // paths warn on a false return) instead of silently building a poisoned LUT.
  static bool isStrictlyIncreasing(const float *axisCoords, size_t numCoords)
  {
    for (size_t i = 1; i < numCoords; ++i)
      if (!(axisCoords[i] > axisCoords[i - 1]))
        return false;
    return true;
  }

  /**
   * Build a rectilinear LUT for a single axis.
   *
   * @param axisCoords Host pointer to monotonically increasing axis coordinates
   * @param numCoords Number of coordinates in the axis (must be >= 2)
   * @param outLutTexture Output CUDA texture object (null if invalid)
   * @param outCudaArray Output CUDA array handle (for cleanup)
   *
   * Bounds are deduced from axisCoords[0] and axisCoords[numCoords-1].
   * Requires strictly monotonic increasing input. Returns false and empty
   * texture object if validation fails.
   * @return true if successful, false otherwise
   */
  static bool buildAxisLUT(const float *axisCoords,
      size_t numCoords,
      cudaTextureObject_t &outLutTexture,
      cudaArray_t &outCudaArray)
  {
    outLutTexture = {};
    outCudaArray = {};

    if (!axisCoords || numCoords < 2 || !isStrictlyIncreasing(axisCoords, numCoords)) {
      return false;
    }

    constexpr size_t lutSize = 8192;
    std::vector<float> lut;
    lut.reserve(lutSize);
    // Make sure bounds interpolate exactly
    lut.push_back(0.0f);

    float step = (axisCoords[numCoords - 1] - axisCoords[0]) / (lutSize - 1);
    float currentValue = axisCoords[0] + step;
    float invDim = 1.0f / (numCoords - 1);

    size_t baseIndex = 0;

    for (size_t i = 1; i < lutSize - 1; ++i) {
      while (
          currentValue > axisCoords[baseIndex + 1] && baseIndex < numCoords - 2)
        ++baseIndex;

      float pct = (currentValue - axisCoords[baseIndex])
          / (axisCoords[baseIndex + 1] - axisCoords[baseIndex]);

      lut.push_back((baseIndex + pct) * invDim);
      currentValue += step;
    }

    // Make sure bounds interpolate exactly
    lut.push_back(1.0f);

    makeCudaArrayFloat(outCudaArray, 1, lut.data(), uvec3(lutSize, 1, 1));

    if (!outCudaArray) {
      return false;
    }

    outLutTexture =
        makeCudaTextureObject1D(outCudaArray, false, "linear", "clampToEdge");

    if (!outLutTexture) {
      cudaFreeArray(outCudaArray);
      return false;
    }

    return true;
  }

  /**
   * Build the inverse rectilinear LUT for a single axis: maps normalized
   * index-space coordinates [0,1] back to normalized object-space coordinates
   * [0,1]. This is the companion of buildAxisLUT (which maps object->index) and
   * lets the isosurface voxel-DDA recover the object-space position of an
   * integer voxel boundary directly (no per-hit root-find): for boundary index
   * k, objectPos = boundsMin + tex(invLUT, k/(numCoords-1)) * (boundsMax-boundsMin).
   *
   * Same contract as buildAxisLUT (strictly monotonic input, >= 2 coords).
   */
  static bool buildInverseAxisLUT(const float *axisCoords,
      size_t numCoords,
      cudaTextureObject_t &outLutTexture,
      cudaArray_t &outCudaArray)
  {
    outLutTexture = {};
    outCudaArray = {};

    if (!axisCoords || numCoords < 2 || !isStrictlyIncreasing(axisCoords, numCoords))
      return false;

    constexpr size_t lutSize = 8192;
    const float c0 = axisCoords[0];
    const float cN = axisCoords[numCoords - 1];
    const float invExtent = (cN != c0) ? 1.0f / (cN - c0) : 0.0f;

    std::vector<float> lut;
    lut.reserve(lutSize);
    for (size_t i = 0; i < lutSize; ++i) {
      const float ni = float(i) / float(lutSize - 1); // normalized index [0,1]
      const float fidx = ni * float(numCoords - 1); // fractional grid index
      size_t k = size_t(fidx);
      if (k > numCoords - 2)
        k = numCoords - 2;
      const float frac = fidx - float(k);
      const float obj =
          axisCoords[k] + frac * (axisCoords[k + 1] - axisCoords[k]);
      lut.push_back((obj - c0) * invExtent); // normalized object [0,1]
    }

    makeCudaArrayFloat(outCudaArray, 1, lut.data(), uvec3(lutSize, 1, 1));
    if (!outCudaArray)
      return false;

    outLutTexture =
        makeCudaTextureObject1D(outCudaArray, false, "linear", "clampToEdge");
    if (!outLutTexture) {
      cudaFreeArray(outCudaArray);
      return false;
    }
    return true;
  }
};

} // namespace visrtx

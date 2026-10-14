/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Array.h"

namespace visrtx {

/* Numeric arrays can mirror their elements into a CUDA array; object arrays
 * only expose linear handles. The CUDA array lives as long as an internal
 * reference to this array: holders keep it alive through ChangeObserverPtr,
 * are re-finalized on every data or extent change before the next frame, and
 * must drop any CUDA object built on it before releasing their reference.
 */
struct NumericArray : public Array
{
  NumericArray(ANARIDataType arrayType,
      DeviceGlobalState *state,
      const ArrayMemoryDescriptor &d,
      size_t totalNumElements);
  ~NumericArray() override;
  NumericArray(const NumericArray &) = delete;
  NumericArray &operator=(const NumericArray &) = delete;
  NumericArray(NumericArray &&) = delete;
  NumericArray &operator=(NumericArray &&) = delete;

  // Allocate on first use, reallocate on extent change, and copy elements when
  // stale. Returns null on failure; the error is reported once per version.
  cudaArray_t cudaArray();

 protected:
  virtual cudaError_t makeCUDAArray(cudaArray_t &array) const = 0;

 private:
  void on_NoInternalReferences() override;
  void freeCUDAArray();

  cudaArray_t m_cuArray{};
  size_t m_cuArraySize{0};
  helium::TimeStamp m_cuArrayUploaded{0};
  helium::TimeStamp m_cuArrayFailure{0};
};

} // namespace visrtx

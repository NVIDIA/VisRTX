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

#include "NumericArray.h"

#include <stdexcept>

namespace visrtx {

NumericArray::NumericArray(ANARIDataType arrayType,
    DeviceGlobalState *state,
    const ArrayMemoryDescriptor &d,
    size_t totalNumElements)
    : Array(arrayType, state, d, totalNumElements)
{
  if (anari::isObject(d.elementType))
    throw std::runtime_error(
        "cannot create numeric array storage from an array of objects");
}

NumericArray::~NumericArray()
{
  freeCUDAArray();
}

cudaArray_t NumericArray::cudaArray()
{
  // Only Array1D can change its extent (begin/end). Every holder is notified of
  // that change and rebuilds what it built on the old one before the next
  // frame.
  if (m_cuArray && m_cuArraySize != totalSize())
    freeCUDAArray();

  const auto modified = lastDataModified();
  if (m_cuArray && m_cuArrayUploaded >= modified)
    return m_cuArray;

  const auto error = makeCUDAArray(m_cuArray);
  if (error != cudaSuccess) {
    // This failure is handled here; do not leave it for a later CUDA/Thrust
    // operation to misattribute to its own work.
    cudaGetLastError();
    // Every holder finalization retries; report each source version once.
    if (m_cuArrayFailure != modified) {
      m_cuArrayFailure = modified;
      reportMessage(ANARI_SEVERITY_ERROR,
          "CUDA array upload failed: %s",
          cudaGetErrorString(error));
    }
    return {};
  }
  m_cuArraySize = totalSize();
  m_cuArrayUploaded = modified;
  return m_cuArray;
}

void NumericArray::on_NoInternalReferences()
{
  // No holder remains, hence nothing built on the CUDA array is still alive.
  freeCUDAArray();
  Array::on_NoInternalReferences();
}

void NumericArray::freeCUDAArray()
{
  if (m_cuArray)
    cudaFreeArray(m_cuArray);
  m_cuArray = {};
  m_cuArraySize = 0;
  m_cuArrayUploaded = {};
}

} // namespace visrtx

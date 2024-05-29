/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "array/Array2D.h"
#include "utility/CudaImageTexture.h"

namespace visrtx {

Array2D::Array2D(DeviceGlobalState *state, const Array2DMemoryDescriptor &d)
    : helium::Array2D(state, d)
{}

const void *Array2D::dataGPU() const
{
  const_cast<Array2D *>(this)->markDataIsOffloaded(true);
  uploadArrayData();
  return m_deviceData.buffer.ptr();
}

cudaArray_t Array2D::acquireCUDAArrayFloat()
{
  if (!m_cuArrayFloat)
    makeCudaArrayFloat(m_cuArrayFloat, *this, uvec2(size().x, size().y));
  m_arrayRefCountFloat++;
  return m_cuArrayFloat;
}

void Array2D::releaseCUDAArrayFloat()
{
  m_arrayRefCountFloat--;
  if (m_arrayRefCountFloat == 0) {
    cudaFreeArray(m_cuArrayFloat);
    m_cuArrayFloat = {};
  }
}

cudaArray_t Array2D::acquireCUDAArrayUint8()
{
  if (!m_cuArrayUint8)
    makeCudaArrayUint8(m_cuArrayUint8, *this, uvec2(size().x, size().y));
  m_arrayRefCountUint8++;
  return m_cuArrayUint8;
}

void Array2D::releaseCUDAArrayUint8()
{
  m_arrayRefCountUint8--;
  if (m_arrayRefCountUint8 == 0) {
    cudaFreeArray(m_cuArrayUint8);
    m_cuArrayUint8 = {};
  }
}

void Array2D::uploadArrayData() const
{
  helium::Array2D::uploadArrayData();
  m_deviceData.buffer.upload(
      (uint8_t *)data(), anari::sizeOf(elementType()) * totalSize());
  if (m_cuArrayFloat)
    makeCudaArrayFloat(m_cuArrayFloat, *this, uvec2(size().x, size().y));
  if (m_cuArrayUint8)
    makeCudaArrayUint8(m_cuArrayUint8, *this, uvec2(size().x, size().y));
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Array2D *);

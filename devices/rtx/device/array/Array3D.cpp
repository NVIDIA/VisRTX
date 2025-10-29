/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "array/Array3D.h"
#include "utility/CudaImageTexture.h"

namespace visrtx {

Array3D::Array3D(DeviceGlobalState *state, const Array3DMemoryDescriptor &d)
    : Array(ANARI_ARRAY3D, state, d, d.numItems1 * d.numItems2 * d.numItems3)
{
  m_size[0] = d.numItems1;
  m_size[1] = d.numItems2;
  m_size[2] = d.numItems3;
}

size_t Array3D::size(int dim) const
{
  return m_size[dim];
}

anari::math::uint3 Array3D::size() const
{
  return anari::math::uint3(
      uint32_t(size(0)), uint32_t(size(1)), uint32_t(size(2)));
}

cudaArray_t Array3D::acquireCUDAArrayFloat()
{
  if (!m_cuArrayFloat)
    makeCudaArrayFloat(
        m_cuArrayFloat, *this, uvec3(size().x, size().y, size().z));
  m_arrayRefCountFloat++;
  return m_cuArrayFloat;
}

void Array3D::releaseCUDAArrayFloat()
{
  m_arrayRefCountFloat--;
  if (m_arrayRefCountFloat == 0) {
    cudaFreeArray(m_cuArrayFloat);
    m_cuArrayFloat = {};
  }
}

cudaArray_t Array3D::acquireCUDAArrayUint8()
{
  if (!m_cuArrayUint8)
    makeCudaArrayUint8(
        m_cuArrayUint8, *this, uvec3(size().x, size().y, size().z));
  m_arrayRefCountUint8++;
  return m_cuArrayUint8;
}

void Array3D::releaseCUDAArrayUint8()
{
  m_arrayRefCountUint8--;
  if (m_arrayRefCountUint8 == 0) {
    cudaFreeArray(m_cuArrayUint8);
    m_cuArrayUint8 = {};
  }
}

void Array3D::uploadArrayData() const
{
  Array::uploadArrayData();
  if (m_cuArrayFloat) {
    makeCudaArrayFloat(
        m_cuArrayFloat, *this, uvec3(size().x, size().y, size().z));
  }
  if (m_cuArrayUint8) {
    makeCudaArrayUint8(
        m_cuArrayUint8, *this, uvec3(size().x, size().y, size().z));
  }
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Array3D *);

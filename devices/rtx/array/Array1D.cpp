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

#include "array/Array1D.h"
#include "utility/CudaImageTexture.h"
// std
#include <algorithm>

namespace visrtx {

Array1D::Array1D(DeviceGlobalState *state, const Array1DMemoryDescriptor &d)
    : Array(ANARI_ARRAY1D, state, d, d.numItems), m_end(d.numItems)
{}

void Array1D::commitParameters()
{
  const auto capacity = totalCapacity();
  m_begin = getParam<size_t>("begin", 0);
  m_begin = std::clamp(m_begin, size_t(0), capacity - 1);
  m_end = getParam<size_t>("end", capacity);
  m_end = std::clamp(m_end, size_t(1), capacity);
}

void Array1D::finalize()
{
  if (size() == 0) {
    reportMessage(ANARI_SEVERITY_ERROR, "array size must be greater than zero");
    return;
  }

  if (m_begin > m_end) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "array 'begin' is not less than 'end', swapping values");
    std::swap(m_begin, m_end);
  }

  markDataModified();
  notifyChangeObservers();
}

size_t Array1D::totalSize() const
{
  return size();
}

size_t Array1D::size() const
{
  return m_end - m_begin;
}

const void *Array1D::begin(AddressSpace as) const
{
  auto *p = (unsigned char *)data(as);
  auto s = anari::sizeOf(elementType());
  return p + (s * m_begin);
}

const void *Array1D::end(AddressSpace as) const
{
  auto *p = (unsigned char *)data(as);
  auto s = anari::sizeOf(elementType());
  return p + (s * m_end);
}

cudaArray_t Array1D::acquireCUDAArrayFloat()
{
  if (!m_cuArrayFloat)
    makeCudaArrayFloat(m_cuArrayFloat, *this, {totalSize(), 1});
  m_arrayRefCountFloat++;
  return m_cuArrayFloat;
}

void Array1D::releaseCUDAArrayFloat()
{
  m_arrayRefCountFloat--;
  if (m_arrayRefCountFloat == 0) {
    cudaFreeArray(m_cuArrayFloat);
    m_cuArrayFloat = {};
  }
}

cudaArray_t Array1D::acquireCUDAArrayUint8()
{
  if (!m_cuArrayUint8)
    makeCudaArrayUint8(m_cuArrayUint8, *this, {totalSize(), 1});
  m_arrayRefCountUint8++;
  return m_cuArrayUint8;
}

void Array1D::releaseCUDAArrayUint8()
{
  m_arrayRefCountUint8--;
  if (m_arrayRefCountUint8 == 0) {
    cudaFreeArray(m_cuArrayUint8);
    m_cuArrayUint8 = {};
  }
}

void Array1D::uploadArrayData() const
{
  Array::uploadArrayData();
  if (m_cuArrayFloat)
    makeCudaArrayFloat(m_cuArrayFloat, *this, {totalSize(), 1});
  if (m_cuArrayUint8)
    makeCudaArrayUint8(m_cuArrayUint8, *this, {totalSize(), 1});
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Array1D *);

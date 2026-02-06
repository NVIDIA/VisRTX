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
// helium
#include <helium/array/Array1D.h>
// std
#include <cassert>

namespace visrtx {

using Array1DMemoryDescriptor = helium::Array1DMemoryDescriptor;

struct Array1D : public Array
{
  Array1D(DeviceGlobalState *state, const Array1DMemoryDescriptor &d);

  void commitParameters() override;
  void finalize() override;

  size_t totalSize() const override;
  size_t size() const;

  const void *begin(AddressSpace as = AddressSpace::HOST) const;
  const void *end(AddressSpace as = AddressSpace::HOST) const;

  template <typename T>
  const T *beginAs(AddressSpace as = AddressSpace::HOST) const;
  template <typename T>
  const T *endAs(AddressSpace as = AddressSpace::HOST) const;

  template <typename T>
  const T *valueAt(size_t i) const;

  cudaArray_t acquireCUDAArrayFloat();
  void releaseCUDAArrayFloat();

  cudaArray_t acquireCUDAArrayUint8();
  void releaseCUDAArrayUint8();

  void uploadArrayData() const override;

 private:
  size_t m_begin{0};
  size_t m_end{0};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline const T *Array1D::beginAs(AddressSpace as) const
{
  assert(anari::ANARITypeFor<T>::value == elementType());

  return dataAs<T>(as) + m_begin;
}

template <typename T>
inline const T *Array1D::endAs(AddressSpace as) const
{
  assert(anari::ANARITypeFor<T>::value == elementType());

  return dataAs<T>(as) + m_end;
}

template <typename T>
inline const T *Array1D::valueAt(size_t i) const
{
  return &beginAs<T>()[i];
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Array1D *, ANARI_ARRAY1D);

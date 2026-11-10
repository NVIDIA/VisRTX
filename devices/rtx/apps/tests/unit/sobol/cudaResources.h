// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda_runtime.h>

#include <memory>

// Standalone tests need explicit CUDA error returns (including allocation
// failures), unlike DeviceBuffer's unchecked copies. Keep ownership local
// without linking the device library or its allocation/logging infrastructure.
struct CudaFree
{
  void operator()(void *pointer) const;
};

template <typename T>
struct CudaBuffer
{
  CudaBuffer() = default;
  CudaBuffer(const CudaBuffer &) = delete;
  CudaBuffer &operator=(const CudaBuffer &) = delete;
  CudaBuffer(CudaBuffer &&) = default;
  CudaBuffer &operator=(CudaBuffer &&) = default;

  cudaError_t allocate(size_t count);
  T *get() const;
  void abandon();

 private:
  std::unique_ptr<T, CudaFree> m_pointer;
};

struct CudaTexture
{
  ~CudaTexture();
  CudaTexture() = default;
  CudaTexture(const CudaTexture &) = delete;
  CudaTexture &operator=(const CudaTexture &) = delete;
  CudaTexture(CudaTexture &&) = delete;
  CudaTexture &operator=(CudaTexture &&) = delete;

  cudaArray_t array = nullptr;
  cudaTextureObject_t texture = 0;
};

// Inlined definitions ////////////////////////////////////////////////////////

inline void CudaFree::operator()(void *pointer) const
{
  cudaFree(pointer);
}

template <typename T>
inline cudaError_t CudaBuffer<T>::allocate(size_t count)
{
  T *pointer = nullptr;
  const auto error = cudaMalloc(&pointer, count * sizeof(T));
  if (error == cudaSuccess)
    m_pointer.reset(pointer);
  return error;
}

template <typename T>
inline T *CudaBuffer<T>::get() const
{
  return m_pointer.get();
}

template <typename T>
inline void CudaBuffer<T>::abandon()
{
  // A negative-test child must not call CUDA again after trapping. Its process
  // owns the disposable context; process exit reclaims the allocation.
  (void)m_pointer.release();
}

inline CudaTexture::~CudaTexture()
{
  if (texture)
    cudaDestroyTextureObject(texture);
  if (array)
    cudaFreeArray(array);
}

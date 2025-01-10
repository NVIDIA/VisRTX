// Copyright 2023-2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// mpi
#include <mpi.h>
// std
#include <cstdint>

namespace wormhole {

template <typename T>
struct RMAWindow
{
  RMAWindow(MPI_Comm c = MPI_COMM_WORLD);
  ~RMAWindow();

  void resize(size_t size);

  void put(int rank, const T *data, size_t size, size_t dstOffset = 0);
  void get(int rank, T *data, size_t size, size_t srcOffset = 0);
  void fence();

  T *ptr();
  const T *data() const;
  size_t size() const;

 private:
  void detach();

  MPI_Comm m_comm{};
  MPI_Win m_win{};
  size_t m_size{0};
  void *m_ptr{nullptr};
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline RMAWindow<T>::RMAWindow(MPI_Comm c) : m_comm(c)
{}

template <typename T>
inline RMAWindow<T>::~RMAWindow()
{
  detach();
}

template <typename T>
inline void RMAWindow<T>::resize(size_t newSize)
{
  detach();
  m_size = newSize;
  MPI_Win_allocate(
      newSize * sizeof(T), 1, MPI_INFO_NULL, m_comm, &m_ptr, &m_win);
  fence();
}

template <typename T>
inline void RMAWindow<T>::put(
    int rank, const T *data, size_t size, size_t dstOffset)
{
  MPI_Put(data,
      size * sizeof(T),
      MPI_INT8_T,
      rank,
      dstOffset * sizeof(T),
      size * sizeof(T),
      MPI_INT8_T,
      m_win);
}

template <typename T>
inline void RMAWindow<T>::get(
    int rank, T *data, size_t size, size_t dstOffset)
{
  MPI_Get(data,
      size * sizeof(T),
      MPI_INT8_T,
      rank,
      dstOffset * sizeof(T),
      size * sizeof(T),
      MPI_INT8_T,
      m_win);
}

template <typename T>
inline void RMAWindow<T>::fence()
{
  MPI_Win_fence(0, m_win);
}

template <typename T>
inline T *RMAWindow<T>::ptr()
{
  return (T *)m_ptr;
}

template <typename T>
inline const T *RMAWindow<T>::data() const
{
  return (const T *)m_ptr;
}

template <typename T>
inline size_t RMAWindow<T>::size() const
{
  return m_size;
}

template <typename T>
inline void RMAWindow<T>::detach()
{
  if (m_win) {
    MPI_Win_free(&m_win);
    m_ptr = nullptr;
    m_size = 0;
  }
}

} // namespace wormhole

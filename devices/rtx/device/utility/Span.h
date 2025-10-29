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

#pragma once

namespace visrtx {

template <typename T>
struct Span
{
  Span() = default;
  Span(const T *, size_t size);

  size_t size() const;
  size_t size_bytes() const;

  const T *data() const;
  const T &operator[](size_t i) const;

  operator bool() const;

  const T *begin() const;
  const T *end() const;

  void reset();

 private:
  const T *m_data{nullptr};
  size_t m_size{0};
};

template <typename T>
Span<T> make_Span(const T *ptr, size_t size);

// Inlined definitions ////////////////////////////////////////////////////////

template <typename T>
inline Span<T>::Span(const T *d, size_t s) : m_data(d), m_size(s)
{}

template <typename T>
inline size_t Span<T>::size() const
{
  return m_size;
}

template <typename T>
inline size_t Span<T>::size_bytes() const
{
  return m_size * sizeof(T);
}

template <typename T>
inline const T *Span<T>::data() const
{
  return m_data;
}

template <typename T>
inline const T &Span<T>::operator[](size_t i) const
{
  return m_data[i];
}

template <typename T>
inline Span<T>::operator bool() const
{
  return m_data != nullptr;
}

template <typename T>
inline const T *Span<T>::begin() const
{
  return m_data;
}

template <typename T>
inline const T *Span<T>::end() const
{
  return begin() + size();
}

template <typename T>
inline void Span<T>::reset()
{
  *this = Span<T>();
}

template <typename T>
inline Span<T> make_Span(const T *ptr, size_t size)
{
  return Span<T>(ptr, size);
}

} // namespace visrtx
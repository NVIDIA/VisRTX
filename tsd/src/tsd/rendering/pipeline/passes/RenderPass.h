// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TSDMath.hpp"

namespace tsd::rendering {

struct RenderPass
{
  struct Buffers
  {
    uint32_t *color{nullptr};
    float *depth{nullptr};
    uint32_t *objectId{nullptr};
  };

  RenderPass();
  virtual ~RenderPass();

  void setEnabled(bool enabled);
  bool isEnabled() const;

  tsd::math::uint2 getDimensions() const;

 protected:
  virtual void render(Buffers &b, int stageId) = 0;
  virtual void updateSize();

 private:
  void setDimensions(uint32_t width, uint32_t height);

  tsd::math::uint2 m_size{0, 0};
  bool m_enabled{true};

  friend struct RenderPipeline;
};

// Utility functions //////////////////////////////////////////////////////////

namespace detail {

void *allocate_(size_t numBytes);
void free_(void *ptr);
void memcpy_(void *dst, const void *src, size_t numBytes);

template <typename T>
inline void copy(T *dst, const T *src, size_t numElements)
{
  detail::memcpy_(dst, src, sizeof(T) * numElements);
}

template <typename T>
inline T *allocate(size_t numElements)
{
  return (T *)detail::allocate_(numElements * sizeof(T));
}

template <typename T>
inline void free(T *ptr)
{
  detail::free_(ptr);
}

} // namespace detail

} // namespace tsd::rendering

// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/TSDMath.hpp"
// tsd_rendering
#include "tsd/rendering/pipeline/passes/detail/ComputeStream.h"

namespace tsd::rendering {

/*
 * POD struct holding pointers to all per-pixel output buffers (color, depth,
 * object/primitive/instance IDs, albedo, normals) shared across pipeline
 * passes.
 *
 * Example:
 *   ImageBuffers b;
 *   b.color = myColorBuffer;
 *   pass.render(b, 0);
 */
struct ImageBuffers
{
  uint32_t *color{nullptr};
  float *depth{nullptr};
  uint32_t *objectId{nullptr};
  uint32_t *primitiveId{nullptr};
  uint32_t *instanceId{nullptr};
  tsd::math::float3 *albedo{nullptr};
  tsd::math::float3 *normal{nullptr};
  detail::ComputeStream stream{};
};

/*
 * Abstract single-stage unit of work in an ImagePipeline; receives a shared
 * ImageBuffers, can be independently enabled/disabled, and is resized by the
 * pipeline when the output dimensions change.
 *
 * Example:
 *   struct MyPass : ImagePass {
 *     void render(ImageBuffers &b, int stageId) override { ... }
 *   };
 *   pipeline.emplace_back<MyPass>();
 */
struct ImagePass
{
  ImagePass();
  virtual ~ImagePass();

  void setEnabled(bool enabled);
  bool isEnabled() const;
  virtual const char *name() const;

  tsd::math::uint2 getDimensions() const;

 protected:
  virtual void render(ImageBuffers &b, int stageId) = 0;
  virtual void updateSize();

 private:
  void setDimensions(uint32_t width, uint32_t height);

  tsd::math::uint2 m_size{0, 0};
  bool m_enabled{true};

  friend struct ImagePipeline;
};

// Utility functions //////////////////////////////////////////////////////////

namespace detail {

void *allocate_(size_t numBytes);
void free_(void *ptr);
void memcpy_(void *dst, const void *src, size_t numBytes);
void convertFloatColorBuffer_(
    ComputeStream stream, const float *v, uint8_t *out, size_t totalSize);

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

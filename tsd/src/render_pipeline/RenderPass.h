// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd
#include "tsd/core/TSDMath.hpp"
// std
#include <vector>
// anari
#include <anari/anari_cpp.hpp>
// OpenGL
#if __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

namespace tsd {

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

  tsd::uint2 getDimensions() const;

 protected:
  virtual void render(Buffers &b, int stageId) = 0;
  virtual void updateSize();

 private:
  void setDimensions(uint32_t width, uint32_t height);

  tsd::uint2 m_size{0, 0};

  friend struct RenderPipeline;
};

// RenderPass subtypes ////////////////////////////////////////////////////////

struct AnariRenderPass : public RenderPass
{
  AnariRenderPass(anari::Device d);
  ~AnariRenderPass() override;

  void setCamera(anari::Camera c);
  void setRenderer(anari::Renderer r);
  void setWorld(anari::World w);

  anari::Frame getFrame() const;

 private:
  void updateSize() override;
  void render(Buffers &b, int stageId) override;
  void copyFrameData();
  void composite(Buffers &b, int stageId);
  void cleanup();

  Buffers m_buffers;

  bool m_firstFrame{true};
  bool m_deviceSupportsCUDAFrames{false};

  anari::Device m_device{nullptr};
  anari::Frame m_frame{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};
  anari::World m_world{nullptr};
};

struct OutlineRenderPass : public RenderPass
{
  OutlineRenderPass();
  ~OutlineRenderPass() override;

  void setOutlineId(uint32_t id);

 private:
  void render(Buffers &b, int stageId) override;

  uint32_t m_outlineId{~0u};
};

struct CopyToGLImagePass : public RenderPass
{
  CopyToGLImagePass();
  ~CopyToGLImagePass() override;

  GLuint getGLTexture() const;

 private:
  void render(Buffers &b, int stageId) override;
  void updateSize() override;

  struct CopyToGLImagePassImpl;
  CopyToGLImagePassImpl *m_impl{nullptr};
};

// Utility functions //////////////////////////////////////////////////////////

namespace detail {

void *allocate_(size_t numBytes);
void free_(void *ptr);
void memcpy_(void *dst, const void *src, size_t numBytes);

template <typename T>
inline void copy(T *dst, const T *src, size_t numElements)
{
  tsd::detail::memcpy_(dst, src, sizeof(T) * numElements);
}

template <typename T>
inline T *allocate(size_t numElements)
{
  return (T *)tsd::detail::allocate_(numElements * sizeof(T));
}

template <typename T>
inline void free(T *ptr)
{
  tsd::detail::free_(ptr);
}

} // namespace detail

} // namespace tsd

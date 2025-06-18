// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd
#include "tsd/core/TSDMath.hpp"
// std
#include <vector>
// anari
#include <anari/anari_cpp.hpp>

#ifdef ENABLE_SDL
#include <SDL3/SDL.h>
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

  void setEnabled(bool enabled);
  bool isEnabled() const;

  tsd::uint2 getDimensions() const;

 protected:
  virtual void render(Buffers &b, int stageId) = 0;
  virtual void updateSize();

 private:
  void setDimensions(uint32_t width, uint32_t height);

  tsd::uint2 m_size{0, 0};
  bool m_enabled{true};

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
  void setColorFormat(anari::DataType t);
  void setEnableIDs(bool on);

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
  bool m_enableIDs{false};

  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};
  anari::World m_world{nullptr};
  anari::Frame m_frame{nullptr};
};

struct PickPass : public RenderPass
{
  using PickOpFunc = std::function<void(RenderPass::Buffers &b)>;

  PickPass();
  ~PickPass() override;

  void setPickOperation(PickOpFunc &&f);

 private:
  void render(Buffers &b, int stageId) override;

  PickOpFunc m_op;
};

struct VisualizeDepthPass : public RenderPass
{
  VisualizeDepthPass();
  ~VisualizeDepthPass() override;

  void setMaxDepth(float d);

 private:
  void render(Buffers &b, int stageId) override;

  float m_maxDepth{1.f};
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

#ifdef ENABLE_SDL
struct CopyToSDLTexturePass : public RenderPass
{
  CopyToSDLTexturePass(SDL_Renderer *renderer);
  ~CopyToSDLTexturePass() override;

  SDL_Texture *getTexture() const;

 private:
  bool checkGLInterop() const;
  void render(Buffers &b, int stageId) override;
  void updateSize() override;

  struct CopyToSDLTexturePassImpl;
  CopyToSDLTexturePassImpl *m_impl{nullptr};
};
#endif

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

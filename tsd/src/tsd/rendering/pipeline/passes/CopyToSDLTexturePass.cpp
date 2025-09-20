// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#if ENABLE_SDL

#include "CopyToSDLTexturePass.h"
#include "tsd/core/Logging.hpp"

#ifdef ENABLE_CUDA
// cuda
#include <SDL3/SDL_opengl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#endif

namespace tsd::rendering {

struct CopyToSDLTexturePass::CopyToSDLTexturePassImpl
{
  SDL_Renderer *renderer{nullptr};
  SDL_Texture *texture{nullptr};
  bool glInteropAvailable{false};
#ifdef ENABLE_CUDA
  cudaGraphicsResource_t graphicsResource{nullptr};
#endif
};

CopyToSDLTexturePass::CopyToSDLTexturePass(SDL_Renderer *renderer)
{
  m_impl = new CopyToSDLTexturePassImpl;
  m_impl->renderer = renderer;
  m_impl->glInteropAvailable = checkGLInterop();
}

CopyToSDLTexturePass::~CopyToSDLTexturePass()
{
#ifdef ENABLE_CUDA
  if (m_impl->graphicsResource)
    cudaGraphicsUnregisterResource(m_impl->graphicsResource);
#endif
  SDL_DestroyTexture(m_impl->texture);
  delete m_impl;
  m_impl = nullptr;
}

SDL_Texture *CopyToSDLTexturePass::getTexture() const
{
  return m_impl->texture;
}

bool CopyToSDLTexturePass::checkGLInterop() const
{
#ifdef ENABLE_CUDA
  unsigned int numDevices = 0;
  int cudaDevices[8]; // Assuming max 8 devices for simplicity

  cudaError_t err =
      cudaGLGetDevices(&numDevices, cudaDevices, 8, cudaGLDeviceListAll);
  if (err != cudaSuccess) {
    tsd::core::logWarning("[RenderPipeline] failed to get CUDA GL devices");
    cudaGetLastError(); // Clear the error so it is not captured by subsequent calls.
    return false;
  }

  if (numDevices > 0) {
    int currentDevice = 0;
    cudaGetDevice(&currentDevice);
    for (unsigned int i = 0; i < numDevices; ++i) {
      if (currentDevice == cudaDevices[i]) {
        tsd::core::logStatus(
            "[RenderPipeline] using CUDA-GL interop via SDL3");
        return true;
      }
    }
  }
#endif

  tsd::core::logWarning(
      "[RenderPipeline] unable to use CUDA-GL interop via SDL3");
  return false;
}

void CopyToSDLTexturePass::render(RenderBuffers &b, int /*stageId*/)
{
  const auto size = getDimensions();

#ifdef ENABLE_CUDA
  if (m_impl->glInteropAvailable && m_impl->graphicsResource) {
    cudaGraphicsMapResources(1, &m_impl->graphicsResource);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(
        &array, m_impl->graphicsResource, 0, 0);
    cudaMemcpy2DToArray(array,
        0,
        0,
        b.color,
        size.x * sizeof(b.color[0]),
        size.x * sizeof(b.color[0]),
        size.y,
        cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &m_impl->graphicsResource);
  } else {
#endif
    SDL_UpdateTexture(m_impl->texture,
        nullptr,
        b.color,
        getDimensions().x * sizeof(b.color[0]));
#ifdef ENABLE_CUDA
  }
#endif
}

void CopyToSDLTexturePass::updateSize()
{
#ifdef ENABLE_CUDA
  if (m_impl->graphicsResource) {
    cudaGraphicsUnregisterResource(m_impl->graphicsResource);
    m_impl->graphicsResource = nullptr;
  }
#endif

  if (m_impl->texture)
    SDL_DestroyTexture(m_impl->texture);
  auto newSize = getDimensions();
  m_impl->texture = SDL_CreateTexture(m_impl->renderer,
      SDL_PIXELFORMAT_RGBA32,
      SDL_TEXTUREACCESS_STREAMING,
      newSize.x,
      newSize.y);

#ifdef ENABLE_CUDA
  if (m_impl->glInteropAvailable) {
    SDL_PropertiesID propID = SDL_GetTextureProperties(m_impl->texture);
    Sint64 texID =
        SDL_GetNumberProperty(propID, SDL_PROP_TEXTURE_OPENGL_TEXTURE_NUMBER, -1);

    if (texID > 0) {
      cudaGraphicsGLRegisterImage(&m_impl->graphicsResource,
          static_cast<GLuint>(texID),
          GL_TEXTURE_2D,
          cudaGraphicsRegisterFlagsWriteDiscard);
    } else {
      tsd::core::logWarning(
          "[RenderPipeline] could not get SDL texture number!");
    }
  }
#endif
}

} // namespace tsd::rendering

#endif

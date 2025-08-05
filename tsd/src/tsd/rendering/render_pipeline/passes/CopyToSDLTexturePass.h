// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef ENABLE_SDL
#define ENABLE_SDL 1
#endif

#if ENABLE_SDL
#include "RenderPass.h"
// SDL3
#include <SDL3/SDL.h>

namespace tsd::rendering {

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

} // namespace tsd::rendering
#endif

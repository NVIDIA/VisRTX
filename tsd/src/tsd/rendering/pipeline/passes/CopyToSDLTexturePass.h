// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef ENABLE_SDL
#define ENABLE_SDL 1
#endif

#if ENABLE_SDL
#include "ImagePass.h"
// SDL3
#include <SDL3/SDL.h>

namespace tsd::rendering {

struct CopyToSDLTexturePass : public ImagePass
{
  CopyToSDLTexturePass(SDL_Renderer *renderer);
  ~CopyToSDLTexturePass() override;
  const char *name() const override { return "Copy To SDL"; }

  SDL_Texture *getTexture() const;

 private:
  bool checkGLInterop() const;
  void render(ImageBuffers &b, int stageId) override;
  void updateSize() override;

  struct CopyToSDLTexturePassImpl;
  CopyToSDLTexturePassImpl *m_impl{nullptr};
};

} // namespace tsd::rendering
#endif

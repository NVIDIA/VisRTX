// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_scene
#include "tsd/scene/objects/Array.hpp"

struct SDL_Renderer;

namespace tsd::ui {

// Must be called once at application startup before any preview helper runs.
// Hands the cache the renderer to create SDL_Textures against.
void setupArrayPreview(SDL_Renderer *r);
void teardownArrayPreview();

// Inline preview widget. Renders an ImGui::Image when the array is
// previewable, otherwise emits a small "no preview" placeholder. Returns
// true when a real thumbnail was drawn.
bool buildUI_array_preview(const tsd::scene::Array &a, float maxDim = 128.f);

} // namespace tsd::ui

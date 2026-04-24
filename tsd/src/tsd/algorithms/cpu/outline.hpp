// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tsd::algorithms::cpu {

// Highlight edges of a selected object by blending an orange tint
// on pixels where the 3x3 neighborhood straddles the object boundary.
void outlineObject(const uint32_t *objectId,
    uint32_t *color,
    uint32_t outlineId,
    uint32_t width,
    uint32_t height);

// Highlight edges between neighboring primitive identities, where identity is
// defined by the pair (objectId, primitiveId). Background transitions are
// treated as silhouettes.
void outlinePrimitives(const uint32_t *objectId,
    const uint32_t *primitiveId,
    uint32_t *color,
    uint32_t outlineColor,
    uint32_t thickness,
    uint32_t width,
    uint32_t height);

} // namespace tsd::algorithms::cpu

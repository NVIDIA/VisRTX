// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "shading/halcyon_objects.isph"
#include "shading/halcyon_util.isph"

// Exported interface /////////////////////////////////////////////////////////

export uniform ulong sizeof_Group()
{
  return sizeof(uniform Group);
}

export void Group_setSurfaces(
    void *uniform _self, void *uniform surfaces, uniform ulong numSurfaces)
{
  uniform Group *uniform self = object_cast<Group>(_self);
  self->surfaces = object_array_cast<Surface>(surfaces);
  self->numSurfaces = numSurfaces;
}

// The group's valid, visible volumes
export void Group_setVolumes(
    void *uniform _self, void *uniform volumes, uniform ulong numVolumes)
{
  uniform Group *uniform self = object_cast<Group>(_self);
  self->volumes = object_array_cast<Volume>(volumes);
  self->numVolumes = numVolumes;
}

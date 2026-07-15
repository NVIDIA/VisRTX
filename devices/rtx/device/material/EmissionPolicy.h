// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// The renderer-side registration policy (ADR 0007): the single source of truth
// for which described emission slots become next-event-sampled Geometry Lights.
// A slot registers iff it is non-null AND faithfully NEE-evaluable on the
// synthetic hit.
//
// `kFaithfulSet` is the EDF-kind half of "faithful". It is in lockstep with two
// GPU-side sites that encode the same diffuse-only assumption:
//   - gpu/lightPickPower.h  (double-sided Lambertian flux for GEOMETRY lights)
//   - gpu/sampleLight.h     (double-sided normal orientation at the synthetic
//                            hit, geometric normal reused as shading normal)
// Growing `kFaithfulSet` beyond {Diffuse} REQUIRES updating both — see the
// synthetic-hit-fidelity follow-up in ADR 0007.

#include "libmdl/EmissionDescriptor.h"

namespace visrtx {

// EDF kinds the renderer can evaluate faithfully at the fidelity-limited
// next-event synthetic hit (geometric normal, synthesized tangent, object id 0,
// forced front). Today: diffuse only.
constexpr libmdl::EdfKind kFaithfulSet = libmdl::EdfKind::Diffuse;

// Whether a described emission slot should register as a Geometry Light.
inline bool isRegisterable(const libmdl::SlotDescriptor &slot)
{
  return slot.verdict != libmdl::EmissionVerdict::ProvablyNull
      && libmdl::isSubsetOf(slot.edfKinds, kFaithfulSet)
      && slot.mode == libmdl::IntensityMode::RadiantExitance
      && !slot.dependsOnGeometricState
      && slot.sign == libmdl::EmissionSign::ProvablyNonnegative;
}

} // namespace visrtx

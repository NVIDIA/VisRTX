// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Owned, immutable intermediate representation of an MDL material's emission
// sub-DAGs (surface/backface × {emission EDF, intensity, mode}, plus
// thin_walled), extracted once from a compiled material while it is alive in
// the registry. The IR retains ZERO MDL-SDK expression pointers — the device
// does not keep the compiled material, so any retained IExpression* would
// dangle. The descriptor fold (EmissionFold) walks this IR plus a value source;
// it never touches the SDK. See ADR 0007.

#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/itransaction.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace visrtx::libmdl {

// Raw MDL semantic of a modeled call node. Stored as the SDK enum (a plain
// integer, not a pointer) so the fold keys on semantics, never DB names — no
// user module can masquerade as an intrinsic. DS_UNKNOWN marks an unmodeled
// call, which the fold treats as lattice-top (Unknown).
using Semantic = mi::neuraylib::IFunction_definition::Semantics;

enum class EmissionNodeKind : std::uint8_t
{
  Constant, // a folded literal (color/float/bool/int/enum)
  Parameter, // a class-compilation argument slot (symbolic under class compile)
  Texture, // a tex::lookup_* whose texture is a parameter or bound resource
  Call, // a modeled call (df::, operator, math::, constructor) over operands
  Opaque, // an unmodeled node — the fold joins it to Unknown
};

enum class ConstantKind : std::uint8_t
{
  None,
  Color, // rgb in value[0..2]
  Float, // scalar broadcast into value[0..2]
  Bool, // in boolValue
  Int, // in intValue
  Enum, // enum ordinal in intValue
  InvalidDf, // the null EDF (default edf()) — an EDF root folds to ProvablyNull
};

// A single IR node. Operand references are indices into EmissionIR::nodes, so
// common subexpressions (MDL `let` temporaries) resolve to ONE shared node —
// giving the fold's `−` rule a decidable exact-identity test.
struct EmissionNode
{
  EmissionNodeKind kind{EmissionNodeKind::Opaque};

  // Constant payload (kind == Constant).
  ConstantKind constantKind{ConstantKind::None};
  std::array<float, 3> value{};
  bool boolValue{false};
  int intValue{0};

  // Parameter / Texture payload: the class-compilation argument this node
  // reads, or -1 for a body-literal/bound texture with no argument slot.
  int parameterIndex{-1};
  std::string parameterName;

  // Texture payload: the DB name of a body-literal bound texture resource
  // (empty when the texture is argument-driven — then parameterIndex is set).
  // The device maps this name to a target-code texture slot; the IR stays
  // device-free.
  std::string resourceName;

  // Call payload (kind == Call): the MDL semantic and operand node indices.
  Semantic semantic{Semantic::DS_UNKNOWN};
  std::vector<int> operands;
};

// The three emission roots of one material side. Each is a node index, or -1 if
// absent (e.g. a material with no emission has edfRoot referencing an
// invalid-df constant, never -1; -1 means the sub-expression was missing).
struct EmissionSlotIR
{
  int edfRoot{-1}; // surface.emission.emission (the EDF)
  int intensityRoot{-1}; // surface.emission.intensity
  int modeRoot{-1}; // surface.emission.mode (an intensity_mode enum)
};

struct EmissionIR
{
  std::vector<EmissionNode> nodes;
  EmissionSlotIR surface;
  EmissionSlotIR backface;
  int thinWalledRoot{-1};

  // Structural dependency sets: every class-compilation argument slot and
  // resource slot reachable from the emission roots, collected across ALL
  // branches of every ternary (dead arms included) so a later condition flip
  // never reads a slot the collection missed. Topology is frozen per compiled
  // material, so a superset is safe.
  std::vector<int> emissionDeps; // parameter indices
  std::vector<std::string> resourceDeps; // bound-texture resource DB names

  bool empty() const
  {
    return nodes.empty();
  }
};

// Build the emission IR from a compiled material. `transaction` must be open
// (it is used only to resolve direct-call definitions to their semantics; no
// SDK pointer is retained past return). Never returns SDK handles.
EmissionIR buildEmissionIR(
    const mi::neuraylib::ICompiled_material *compiledMaterial,
    mi::neuraylib::ITransaction *transaction);

} // namespace visrtx::libmdl

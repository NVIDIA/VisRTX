// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "EmissionFold.h"

#include <algorithm>
#include <cmath>

namespace visrtx::libmdl {

namespace {

constexpr float INV_PI = 0.31830988618379067154f;

// Geometric-state quantities the synthetic next-event hit fabricates, so
// emission that reads them is not faithfully NEE-evaluable (ADR 0007). NOT
// texture_coordinate: uv-driven (textured) emission is the shipped sampleable
// case, evaluated via the sampler mean; nor the globally-uniform state
// (animation_time, wavelengths, scene units) which the synthetic hit
// reproduces.
bool isFabricatedState(Semantic s)
{
  switch (s) {
  case Semantic::DS_INTRINSIC_STATE_POSITION:
  case Semantic::DS_INTRINSIC_STATE_NORMAL:
  case Semantic::DS_INTRINSIC_STATE_GEOMETRY_NORMAL:
  case Semantic::DS_INTRINSIC_STATE_MOTION:
  case Semantic::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U:
  case Semantic::DS_INTRINSIC_STATE_TEXTURE_TANGENT_V:
  case Semantic::DS_INTRINSIC_STATE_TANGENT_SPACE:
  case Semantic::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_U:
  case Semantic::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_V:
  case Semantic::DS_INTRINSIC_STATE_DIRECTION:
  case Semantic::DS_INTRINSIC_STATE_TRANSFORM:
  case Semantic::DS_INTRINSIC_STATE_TRANSFORM_POINT:
  case Semantic::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
  case Semantic::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
  case Semantic::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
  case Semantic::DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL:
  case Semantic::DS_INTRINSIC_STATE_OBJECT_ID:
    return true;
  default:
    return false;
  }
}

EdfKind edfKindOf(Semantic s)
{
  switch (s) {
  case Semantic::DS_INTRINSIC_DF_DIFFUSE_EDF:
    return EdfKind::Diffuse;
  case Semantic::DS_INTRINSIC_DF_SPOT_EDF:
    return EdfKind::Spot;
  case Semantic::DS_INTRINSIC_DF_MEASURED_EDF:
    return EdfKind::Measured;
  default:
    return EdfKind::None;
  }
}

bool isMixSemantic(Semantic s)
{
  switch (s) {
  case Semantic::DS_INTRINSIC_DF_NORMALIZED_MIX:
  case Semantic::DS_INTRINSIC_DF_CLAMPED_MIX:
  case Semantic::DS_INTRINSIC_DF_UNBOUNDED_MIX:
  case Semantic::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
  case Semantic::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
  case Semantic::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX:
    return true;
  default:
    return false;
  }
}

enum class Tri : std::uint8_t
{
  False,
  True,
  Unknown
};

// Abstract value of a scalar/color sub-expression.
struct Scalar
{
  Tri zero{Tri::Unknown}; // is the value identically zero?
  bool magnitudeKnown{false};
  std::array<float, 3> magnitude{}; // meanPositive per channel (>= 0), if known
  EmissionSign sign{EmissionSign::Unknown};
  bool dependsOnState{false};
  bool finite{true};
};

// Abstract value of an EDF sub-expression.
struct Edf
{
  Tri null{Tri::Unknown}; // is the EDF the null (non-emitting) EDF?
  EdfKind kinds{EdfKind::None};
  bool dependsOnState{false};
};

std::array<float, 3> positivePart(const std::array<float, 3> &v)
{
  return {std::max(v[0], 0.0f), std::max(v[1], 0.0f), std::max(v[2], 0.0f)};
}

bool allZero(const std::array<float, 3> &v)
{
  return v[0] == 0.0f && v[1] == 0.0f && v[2] == 0.0f;
}

bool anyNonzero(const std::array<float, 3> &v)
{
  return v[0] != 0.0f || v[1] != 0.0f || v[2] != 0.0f;
}

bool allNonnegative(const std::array<float, 3> &v)
{
  return v[0] >= 0.0f && v[1] >= 0.0f && v[2] >= 0.0f;
}

bool allFinite(const std::array<float, 3> &v)
{
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

class Fold
{
 public:
  Fold(const EmissionIR &ir, const EmissionValueSource &values)
      : m_ir(ir), m_values(values)
  {}

  SlotDescriptor foldSlot(const EmissionSlotIR &slot)
  {
    SlotDescriptor desc;
    if (slot.edfRoot < 0) {
      desc.verdict = EmissionVerdict::ProvablyNull;
      return desc;
    }

    const Edf edf = evalEdf(slot.edfRoot, 0);
    Scalar intensity;
    if (slot.intensityRoot >= 0)
      intensity = evalScalar(slot.intensityRoot, 0);
    else
      intensity.zero = Tri::True; // no intensity ⇒ no emission

    desc.edfKinds = edf.kinds;
    desc.mode = foldMode(slot.modeRoot);
    desc.dependsOnGeometricState =
        edf.dependsOnState || intensity.dependsOnState;

    // verdict = nullness of (EDF · intensity): null iff EDF null OR intensity
    // 0.
    const Tri emissionNull =
        (edf.null == Tri::True || intensity.zero == Tri::True)
        ? Tri::True
        : ((edf.null == Tri::False && intensity.zero == Tri::False)
                  ? Tri::False
                  : Tri::Unknown);
    if (!intensity.finite)
      desc.verdict = EmissionVerdict::Unknown;
    else if (emissionNull == Tri::True)
      desc.verdict = EmissionVerdict::ProvablyNull;
    else if (emissionNull == Tri::False)
      desc.verdict = EmissionVerdict::ProvablyEmissive;
    else
      desc.verdict = EmissionVerdict::Unknown;

    // magnitude = meanPositive(intensity) / PI (diffuse EDF value 1/PI). A unit
    // proxy stands in when the intensity magnitude is not host-known —
    // unbiased, it only weights the Light Pick.
    if (intensity.magnitudeKnown) {
      for (int i = 0; i < 3; ++i)
        desc.magnitude[i] = intensity.magnitude[i] * INV_PI;
    } else {
      desc.magnitude = {1.0f, 1.0f, 1.0f};
    }

    // sign gates registration; the EDF itself is nonnegative.
    desc.sign = intensity.sign;
    return desc;
  }

 private:
  const EmissionNode &node(int i) const
  {
    return m_ir.nodes[std::size_t(i)];
  }

  IntensityMode foldMode(int modeRoot) const
  {
    if (modeRoot < 0)
      return IntensityMode::RadiantExitance; // default
    const auto &n = node(modeRoot);
    if (n.kind == EmissionNodeKind::Constant
        && n.constantKind == ConstantKind::Enum)
      return n.intValue == 0 ? IntensityMode::RadiantExitance
                             : IntensityMode::Power;
    // Non-constant mode: cannot prove radiant-exitance ⇒ treat as power (not
    // registered) rather than silently assume the faithful mode.
    return IntensityMode::Power;
  }

  Edf evalEdf(int index, int depth)
  {
    Edf r;
    if (index < 0 || depth > 64)
      return r; // Unknown
    const auto &n = node(index);

    if (n.kind == EmissionNodeKind::Constant) {
      if (n.constantKind == ConstantKind::InvalidDf) {
        r.null = Tri::True;
        return r;
      }
      return r; // a non-df constant in EDF position: Unknown
    }
    if (n.kind != EmissionNodeKind::Call) {
      r.kinds = EdfKind::Unknown;
      return r;
    }

    const EdfKind leaf = edfKindOf(n.semantic);
    if (leaf != EdfKind::None) {
      r.kinds = leaf;
      r.null = Tri::False; // a present emissive leaf
      return r;
    }

    switch (n.semantic) {
    case Semantic::DS_INTRINSIC_DF_TINT: {
      // tint(color tint, edf base): null iff tint zero OR base null.
      Edf base;
      Scalar tint;
      for (int op : n.operands) {
        if (isEdfNode(op))
          base = evalEdf(op, depth + 1);
        else
          tint = evalScalar(op, depth + 1);
      }
      r.kinds = base.kinds;
      r.dependsOnState = base.dependsOnState || tint.dependsOnState;
      if (tint.zero == Tri::True || base.null == Tri::True)
        r.null = Tri::True;
      else if (tint.zero == Tri::False && base.null == Tri::False)
        r.null = Tri::False;
      else
        r.null = Tri::Unknown;
      return r;
    }
    case Semantic::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR: {
      // Directional emission: kind = directional + base kinds; null iff base
      // null or both endpoint tints cross-channel zero. Endpoints are the two
      // color operands; base is the edf operand.
      Edf base;
      bool sawEdf = false;
      Tri anyTintNonzero = Tri::False;
      bool allTintZeroKnown = true;
      for (int op : n.operands) {
        if (isEdfNode(op)) {
          base = evalEdf(op, depth + 1);
          sawEdf = true;
          r.dependsOnState = r.dependsOnState || base.dependsOnState;
        } else {
          Scalar tint = evalScalar(op, depth + 1);
          r.dependsOnState = r.dependsOnState || tint.dependsOnState;
          if (tint.zero != Tri::True)
            allTintZeroKnown = false;
          if (tint.zero == Tri::False)
            anyTintNonzero = Tri::True;
        }
      }
      r.kinds = EdfKind::Directional;
      if (sawEdf)
        r.kinds |= base.kinds;
      if (base.null == Tri::True || allTintZeroKnown)
        r.null = Tri::True;
      else if (base.null == Tri::False && anyTintNonzero == Tri::True)
        r.null = Tri::False;
      else
        r.null = Tri::Unknown;
      return r;
    }
    default:
      break;
    }

    if (isMixSemantic(n.semantic)) {
      // Union component kinds; null iff every EDF operand is null. A precise
      // per-weight analysis needs the df_component array shape; the
      // conservative union keeps kinds honest (drives the fidelity gate)
      // without proving Emissive, which only affects variance.
      bool allNull = true;
      for (int op : n.operands) {
        if (!isEdfNode(op))
          continue;
        Edf c = evalEdf(op, depth + 1);
        r.kinds |= c.kinds;
        r.dependsOnState = r.dependsOnState || c.dependsOnState;
        if (c.null != Tri::True)
          allNull = false;
      }
      if (r.kinds == EdfKind::None)
        r.kinds = EdfKind::Unknown;
      r.null = allNull ? Tri::True : Tri::Unknown;
      return r;
    }

    // Unmodeled df:: call.
    r.kinds = EdfKind::Unknown;
    return r;
  }

  bool isEdfNode(int index) const
  {
    if (index < 0)
      return false;
    const auto &n = node(index);
    if (n.kind == EmissionNodeKind::Constant)
      return n.constantKind == ConstantKind::InvalidDf;
    if (n.kind == EmissionNodeKind::Call)
      return edfKindOf(n.semantic) != EdfKind::None
          || n.semantic == Semantic::DS_INTRINSIC_DF_TINT
          || n.semantic == Semantic::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR
          || isMixSemantic(n.semantic);
    return false;
  }

  Scalar evalScalar(int index, int depth)
  {
    Scalar r;
    if (index < 0 || depth > 64)
      return r; // Unknown
    const auto &n = node(index);

    switch (n.kind) {
    case EmissionNodeKind::Constant:
      return scalarFromConstant(n);
    case EmissionNodeKind::Parameter:
      return scalarFromParameter(n);
    case EmissionNodeKind::Texture:
      return scalarFromTexture(n);
    case EmissionNodeKind::Call:
      return scalarFromCall(n, depth);
    default:
      return r; // Opaque ⇒ Unknown
    }
  }

  Scalar scalarFromConstant(const EmissionNode &n)
  {
    Scalar r;
    if (n.constantKind == ConstantKind::Color
        || n.constantKind == ConstantKind::Float) {
      r.finite = allFinite(n.value);
      r.zero = allZero(n.value) ? Tri::True
          : anyNonzero(n.value) ? Tri::False
                                : Tri::Unknown;
      r.magnitudeKnown = true;
      r.magnitude = positivePart(n.value);
      r.sign = allNonnegative(n.value) ? EmissionSign::ProvablyNonnegative
                                       : EmissionSign::Unknown;
      return r;
    }
    // Bool/int/enum in a radiance context: leave Unknown (should not occur).
    return r;
  }

  Scalar scalarFromParameter(const EmissionNode &n)
  {
    Scalar r;
    std::array<float, 3> value;
    if (m_values.color(n.parameterIndex, value)) {
      r.finite = allFinite(value);
      r.zero = allZero(value) ? Tri::True
          : anyNonzero(value) ? Tri::False
                              : Tri::Unknown;
      r.magnitudeKnown = true;
      r.magnitude = positivePart(value);
      r.sign = allNonnegative(value) ? EmissionSign::ProvablyNonnegative
                                     : EmissionSign::Unknown;
    }
    // Unknown value ⇒ all Unknown (default), sign Unknown, magnitude unknown.
    return r;
  }

  Scalar scalarFromTexture(const EmissionNode &n)
  {
    Scalar r;
    // A coord/wrap/crop argument that reads a fabricated geometric-state
    // quantity makes the lookup unfaithful at the synthetic hit.
    for (int op : n.operands) {
      Scalar s = evalScalar(op, 1);
      r.dependsOnState = r.dependsOnState || s.dependsOnState;
    }

    ResourceStats stats;
    const bool known = n.parameterIndex >= 0
        ? m_values.resourceByParam(n.parameterIndex, stats)
        : m_values.resourceByName(n.resourceName, stats);
    if (!known)
      return r; // Unknown (state-dependence preserved)

    if (!stats.valid) {
      // Unbound/invalid texture ⇒ lookup folds to 0.
      r.zero = Tri::True;
      r.magnitudeKnown = true;
      r.magnitude = {0.0f, 0.0f, 0.0f};
      r.sign = EmissionSign::ProvablyNonnegative;
      return r;
    }
    r.finite = stats.finite;
    // Zero only provable in sampler-output space (T(0)==0).
    if (stats.transferPreservesZero && allZero(stats.maxAbs))
      r.zero = Tri::True;
    else
      r.zero = Tri::Unknown;
    r.magnitudeKnown = true;
    r.magnitude = stats.meanPositive;
    r.sign = allNonnegative(stats.minValue) ? EmissionSign::ProvablyNonnegative
                                            : EmissionSign::Unknown;
    return r;
  }

  Scalar scalarFromCall(const EmissionNode &n, int depth)
  {
    // State-dependence propagates regardless of whether the op is modeled.
    bool stateFromThis = isFabricatedState(n.semantic);

    switch (n.semantic) {
    case Semantic::DS_MULTIPLY:
      return combineMul(n, depth, stateFromThis);
    case Semantic::DS_PLUS:
      return combineAdd(n, depth, stateFromThis);
    case Semantic::DS_MINUS:
      return combineSub(n, depth, stateFromThis);
    case Semantic::DS_TERNARY:
      return combineTernary(n, depth, stateFromThis);
    case Semantic::DS_CONV_CONSTRUCTOR:
    case Semantic::DS_ELEM_CONSTRUCTOR:
    case Semantic::DS_CONV_OPERATOR:
    case Semantic::DS_COPY_CONSTRUCTOR:
      return combineConstructor(n, depth, stateFromThis);
    default:
      break;
    }

    // Unmodeled call: Unknown value, but still scan operands for state and
    // finiteness so the flags stay sound.
    Scalar r;
    r.dependsOnState = stateFromThis;
    for (int op : n.operands) {
      Scalar s = evalScalar(op, depth + 1);
      r.dependsOnState = r.dependsOnState || s.dependsOnState;
    }
    return r;
  }

  Scalar combineMul(const EmissionNode &n, int depth, bool stateFromThis)
  {
    Scalar r;
    r.zero = Tri::True; // identity for the running product's zero test
    r.magnitudeKnown = true;
    r.magnitude = {1.0f, 1.0f, 1.0f};
    r.sign = EmissionSign::ProvablyNonnegative;
    r.dependsOnState = stateFromThis;
    bool any = false;
    Tri zeroAcc =
        Tri::False; // zero-absorbing: product zero iff any factor zero
    for (int op : n.operands) {
      Scalar s = evalScalar(op, depth + 1);
      any = true;
      r.dependsOnState = r.dependsOnState || s.dependsOnState;
      r.finite = r.finite && s.finite;
      if (s.zero == Tri::True)
        zeroAcc = Tri::True;
      else if (s.zero == Tri::Unknown && zeroAcc != Tri::True)
        zeroAcc = Tri::Unknown;
      r.magnitudeKnown = r.magnitudeKnown && s.magnitudeKnown;
      if (s.magnitudeKnown)
        for (int i = 0; i < 3; ++i)
          r.magnitude[i] *= s.magnitude[i];
      if (s.sign != EmissionSign::ProvablyNonnegative)
        r.sign = EmissionSign::Unknown;
    }
    if (!any)
      return Scalar{};
    r.zero = zeroAcc;
    if (!r.magnitudeKnown)
      r.magnitude = {};
    return r;
  }

  Scalar combineAdd(const EmissionNode &n, int depth, bool stateFromThis)
  {
    Scalar r;
    r.magnitudeKnown = true;
    r.magnitude = {0.0f, 0.0f, 0.0f};
    r.sign = EmissionSign::ProvablyNonnegative;
    r.dependsOnState = stateFromThis;
    bool allZeroKnown = true;
    bool anyNonzero = false;
    bool anyUnknownZero = false;
    for (int op : n.operands) {
      Scalar s = evalScalar(op, depth + 1);
      r.dependsOnState = r.dependsOnState || s.dependsOnState;
      r.finite = r.finite && s.finite;
      if (s.zero != Tri::True)
        allZeroKnown = false;
      if (s.zero == Tri::False)
        anyNonzero = true;
      if (s.zero == Tri::Unknown)
        anyUnknownZero = true;
      r.magnitudeKnown = r.magnitudeKnown && s.magnitudeKnown;
      if (s.magnitudeKnown)
        for (int i = 0; i < 3; ++i)
          r.magnitude[i] += s.magnitude[i];
      if (s.sign != EmissionSign::ProvablyNonnegative)
        r.sign = EmissionSign::Unknown;
    }
    // sum zero iff all zero; nonzero needs nonnegativity (a negative could
    // cancel a positive), so only claim Nonzero when the sum is provably so.
    if (allZeroKnown)
      r.zero = Tri::True;
    else if (anyNonzero && !anyUnknownZero
        && r.sign == EmissionSign::ProvablyNonnegative)
      r.zero = Tri::False;
    else
      r.zero = Tri::Unknown;
    if (!r.magnitudeKnown)
      r.magnitude = {};
    return r;
  }

  Scalar combineSub(const EmissionNode &n, int depth, bool stateFromThis)
  {
    Scalar r;
    r.dependsOnState = stateFromThis;
    // ProvablyZero only on exact node identity (a - a). Otherwise Unknown —
    // never ProvablyNonzero (1 - w with w unknown can be zero).
    if (n.operands.size() == 2 && n.operands[0] == n.operands[1]) {
      r.zero = Tri::True;
      r.magnitudeKnown = true;
      r.magnitude = {0.0f, 0.0f, 0.0f};
      r.sign = EmissionSign::ProvablyNonnegative;
    }
    for (int op : n.operands) {
      Scalar s = evalScalar(op, depth + 1);
      r.dependsOnState = r.dependsOnState || s.dependsOnState;
      r.finite = r.finite && s.finite;
    }
    return r;
  }

  Scalar combineTernary(const EmissionNode &n, int depth, bool stateFromThis)
  {
    if (n.operands.size() != 3)
      return Scalar{};
    Scalar cond = evalScalar(n.operands[0], depth + 1);
    const bool condState = cond.dependsOnState;

    // Fold if the condition is a known constant bool.
    const auto &condNode = node(n.operands[0]);
    if (condNode.kind == EmissionNodeKind::Constant
        && condNode.constantKind == ConstantKind::Bool) {
      const int arm = condNode.boolValue ? 1 : 2;
      Scalar r = evalScalar(n.operands[arm], depth + 1);
      r.dependsOnState = r.dependsOnState || stateFromThis || condState;
      return r;
    }

    // Otherwise join the arms (LUB with Unknown as top).
    Scalar a = evalScalar(n.operands[1], depth + 1);
    Scalar b = evalScalar(n.operands[2], depth + 1);
    Scalar r;
    r.dependsOnState =
        stateFromThis || condState || a.dependsOnState || b.dependsOnState;
    r.finite = a.finite && b.finite;
    r.zero = joinTri(a.zero, b.zero);
    r.sign = (a.sign == EmissionSign::ProvablyNonnegative
                 && b.sign == EmissionSign::ProvablyNonnegative)
        ? EmissionSign::ProvablyNonnegative
        : EmissionSign::Unknown;
    r.magnitudeKnown = a.magnitudeKnown && b.magnitudeKnown;
    if (r.magnitudeKnown)
      for (int i = 0; i < 3; ++i)
        r.magnitude[i] = std::max(a.magnitude[i], b.magnitude[i]);
    return r;
  }

  Scalar combineConstructor(
      const EmissionNode &n, int depth, bool stateFromThis)
  {
    // color(float) / color(r,g,b) / conversions: fold operands into a value
    // when all are known constants; otherwise propagate zero/sign/state
    // conservatively.
    Scalar r;
    r.dependsOnState = stateFromThis;
    if (n.operands.empty())
      return r;

    // Single-operand broadcast (color(float), conversions).
    if (n.operands.size() == 1)
      return applyState(evalScalar(n.operands[0], depth + 1), stateFromThis);

    // Multi-channel constructor: combine per-channel scalars.
    r.magnitudeKnown = true;
    r.magnitude = {0.0f, 0.0f, 0.0f};
    r.sign = EmissionSign::ProvablyNonnegative;
    Tri zeroAcc = Tri::True;
    for (std::size_t i = 0; i < n.operands.size() && i < 3; ++i) {
      Scalar s = evalScalar(n.operands[int(i)], depth + 1);
      r.dependsOnState = r.dependsOnState || s.dependsOnState;
      r.finite = r.finite && s.finite;
      zeroAcc = zeroAcc == Tri::True ? s.zero : joinNonzero(zeroAcc, s.zero);
      r.magnitudeKnown = r.magnitudeKnown && s.magnitudeKnown;
      if (s.magnitudeKnown)
        r.magnitude[i] = s.magnitude[0];
      if (s.sign != EmissionSign::ProvablyNonnegative)
        r.sign = EmissionSign::Unknown;
    }
    r.zero = zeroAcc;
    if (!r.magnitudeKnown)
      r.magnitude = {};
    return r;
  }

  static Scalar applyState(Scalar s, bool stateFromThis)
  {
    s.dependsOnState = s.dependsOnState || stateFromThis;
    return s;
  }

  static Tri joinTri(Tri a, Tri b)
  {
    if (a == b)
      return a;
    return Tri::Unknown;
  }

  // For a channel accumulation where "all zero ⇒ zero, any nonzero ⇒ nonzero".
  static Tri joinNonzero(Tri acc, Tri ch)
  {
    if (acc == Tri::False || ch == Tri::False)
      return Tri::False; // a nonzero channel makes the color nonzero
    if (acc == Tri::True && ch == Tri::True)
      return Tri::True;
    return Tri::Unknown;
  }

  const EmissionIR &m_ir;
  const EmissionValueSource &m_values;
};

} // namespace

EmissionDescriptor foldEmissionDescriptor(
    const EmissionIR &ir, const EmissionValueSource &values)
{
  EmissionDescriptor desc;
  if (ir.empty())
    return desc;
  Fold fold(ir, values);
  desc.surface = fold.foldSlot(ir.surface);
  desc.backface = fold.foldSlot(ir.backface);
  return desc;
}

} // namespace visrtx::libmdl

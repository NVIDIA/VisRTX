// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "EmissionIR.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/iexpression.h>
#include <mi/neuraylib/ivalue.h>

#include <algorithm>
#include <unordered_map>

namespace visrtx::libmdl {

namespace {

using namespace mi::neuraylib;
using mi::base::make_handle;

// Bound the walk so a pathological DAG cannot blow the stack; deeper than this
// no real emission graph goes, and anything truncated folds to Opaque/Unknown.
constexpr int kMaxWalkDepth = 64;

// tex::lookup_color / _float3 / _float — the texture-sampling intrinsics whose
// first argument is the texture. Keyed on semantics below.
bool isTextureLookup(Semantic s)
{
  switch (s) {
  case Semantic::DS_INTRINSIC_TEX_LOOKUP_FLOAT:
  case Semantic::DS_INTRINSIC_TEX_LOOKUP_FLOAT2:
  case Semantic::DS_INTRINSIC_TEX_LOOKUP_FLOAT3:
  case Semantic::DS_INTRINSIC_TEX_LOOKUP_FLOAT4:
  case Semantic::DS_INTRINSIC_TEX_LOOKUP_COLOR:
    return true;
  default:
    return false;
  }
}

class Builder
{
 public:
  Builder(const ICompiled_material *material, ITransaction *transaction)
      : m_material(material), m_transaction(transaction)
  {}

  EmissionIR build()
  {
    EmissionIR ir;
    m_ir = &ir;

    ir.surface = buildSlot("surface");
    ir.backface = buildSlot("backface");
    ir.thinWalledRoot = buildPath("thin_walled");

    collectDeps(ir);
    m_ir = nullptr;
    return ir;
  }

 private:
  EmissionSlotIR buildSlot(const char *side)
  {
    EmissionSlotIR slot;
    slot.edfRoot = buildPath(std::string(side) + ".emission.emission");
    slot.intensityRoot = buildPath(std::string(side) + ".emission.intensity");
    slot.modeRoot = buildPath(std::string(side) + ".emission.mode");
    return slot;
  }

  int buildPath(const std::string &path)
  {
    auto expr = make_handle(m_material->lookup_sub_expression(path.c_str()));
    if (!expr)
      return -1;
    return buildExpr(expr, 0);
  }

  // Resolve `let`-block temporaries and memoize on the temporary index so a
  // subexpression shared through CSE maps to ONE node.
  mi::base::Handle<const IExpression> deref(
      mi::base::Handle<const IExpression> expr, int *sharedTemporary)
  {
    *sharedTemporary = -1;
    while (expr && expr->get_kind() == IExpression::EK_TEMPORARY) {
      auto tmp =
          make_handle(expr->get_interface<const IExpression_temporary>());
      *sharedTemporary = int(tmp->get_index());
      expr = make_handle(m_material->get_temporary(tmp->get_index()));
    }
    return expr;
  }

  int buildExpr(mi::base::Handle<const IExpression> expr, int depth)
  {
    int sharedTemporary = -1;
    expr = deref(expr, &sharedTemporary);
    if (!expr || depth > kMaxWalkDepth)
      return addNode(EmissionNode{}); // Opaque

    if (sharedTemporary >= 0) {
      auto it = m_temporaryNodes.find(sharedTemporary);
      if (it != m_temporaryNodes.end())
        return it->second;
    }

    const int nodeIndex = buildExprUncached(expr, depth);
    if (sharedTemporary >= 0)
      m_temporaryNodes.emplace(sharedTemporary, nodeIndex);
    return nodeIndex;
  }

  int buildExprUncached(mi::base::Handle<const IExpression> expr, int depth)
  {
    switch (expr->get_kind()) {
    case IExpression::EK_CONSTANT:
      return buildConstant(
          make_handle(expr->get_interface<const IExpression_constant>()));
    case IExpression::EK_PARAMETER:
      return buildParameter(
          make_handle(expr->get_interface<const IExpression_parameter>()));
    case IExpression::EK_DIRECT_CALL:
      return buildCall(
          make_handle(expr->get_interface<const IExpression_direct_call>()),
          depth);
    default:
      return addNode(EmissionNode{}); // Opaque
    }
  }

  int buildConstant(mi::base::Handle<const IExpression_constant> constant)
  {
    EmissionNode node;
    node.kind = EmissionNodeKind::Constant;
    auto value = make_handle(constant->get_value());
    if (!value)
      return addNode(EmissionNode{});

    switch (value->get_kind()) {
    case IValue::VK_COLOR: {
      auto color = make_handle(value->get_interface<const IValue_color>());
      node.constantKind = ConstantKind::Color;
      for (int i = 0; i < 3; ++i) {
        auto ch = make_handle(color->get_value(i));
        auto f = make_handle(ch->get_interface<const IValue_float>());
        if (!f)
          return addNode(EmissionNode{}); // non-literal channel: Opaque
        node.value[i] = f->get_value();
      }
      break;
    }
    case IValue::VK_FLOAT: {
      const float f =
          make_handle(value->get_interface<const IValue_float>())->get_value();
      node.constantKind = ConstantKind::Float;
      node.value = {f, f, f};
      break;
    }
    case IValue::VK_BOOL:
      node.constantKind = ConstantKind::Bool;
      node.boolValue =
          make_handle(value->get_interface<const IValue_bool>())->get_value();
      break;
    case IValue::VK_INT:
      node.constantKind = ConstantKind::Int;
      node.intValue =
          make_handle(value->get_interface<const IValue_int>())->get_value();
      break;
    case IValue::VK_ENUM:
      node.constantKind = ConstantKind::Enum;
      node.intValue =
          make_handle(value->get_interface<const IValue_enum>())->get_value();
      break;
    case IValue::VK_INVALID_DF:
      node.constantKind = ConstantKind::InvalidDf;
      break;
    default:
      return addNode(EmissionNode{}); // unmodeled value kind: Opaque
    }
    return addNode(std::move(node));
  }

  int buildParameter(mi::base::Handle<const IExpression_parameter> param)
  {
    EmissionNode node;
    node.kind = EmissionNodeKind::Parameter;
    node.parameterIndex = int(param->get_index());
    const char *name = m_material->get_parameter_name(param->get_index());
    node.parameterName = name ? name : "";
    return addNode(std::move(node));
  }

  Semantic semanticOf(const IExpression_direct_call *call)
  {
    const char *definition = call->get_definition();
    if (!definition)
      return Semantic::DS_UNKNOWN;
    auto fn =
        make_handle(m_transaction->access<IFunction_definition>(definition));
    if (!fn)
      return Semantic::DS_UNKNOWN;
    return fn->get_semantic();
  }

  int buildCall(mi::base::Handle<const IExpression_direct_call> call, int depth)
  {
    const Semantic semantic = semanticOf(call.get());
    auto args = make_handle(call->get_arguments());
    if (!args)
      return addNode(EmissionNode{});

    if (isTextureLookup(semantic))
      return buildTexture(semantic, args, depth);

    EmissionNode node;
    node.kind = EmissionNodeKind::Call;
    node.semantic = semantic;
    for (mi::Size i = 0; i < args->get_size(); ++i) {
      node.operands.push_back(
          buildExpr(make_handle(args->get_expression(i)), depth + 1));
    }
    return addNode(std::move(node));
  }

  int buildTexture(Semantic semantic,
      mi::base::Handle<const IExpression_list> args,
      int depth)
  {
    EmissionNode node;
    node.kind = EmissionNodeKind::Texture;
    node.semantic = semantic;

    int sharedTemporary = -1;
    auto tex =
        deref(make_handle(args->get_expression("tex")), &sharedTemporary);
    if (tex && tex->get_kind() == IExpression::EK_PARAMETER) {
      auto param =
          make_handle(tex->get_interface<const IExpression_parameter>());
      node.parameterIndex = int(param->get_index());
      const char *name = m_material->get_parameter_name(param->get_index());
      node.parameterName = name ? name : "";
    } else if (tex && tex->get_kind() == IExpression::EK_CONSTANT) {
      auto constant =
          make_handle(tex->get_interface<const IExpression_constant>());
      auto value = make_handle(constant->get_value());
      if (value && value->get_kind() == IValue::VK_TEXTURE) {
        auto texValue =
            make_handle(value->get_interface<const IValue_texture>());
        const char *name = texValue->get_value();
        node.resourceName = name ? name : "";
      }
    }
    return addNode(std::move(node));
  }

  int addNode(EmissionNode node)
  {
    m_ir->nodes.push_back(std::move(node));
    return int(m_ir->nodes.size()) - 1;
  }

  void collectDeps(EmissionIR &ir)
  {
    for (const auto &node : ir.nodes) {
      if (node.parameterIndex >= 0)
        ir.emissionDeps.push_back(node.parameterIndex);
      if (node.kind == EmissionNodeKind::Texture && !node.resourceName.empty())
        ir.resourceDeps.push_back(node.resourceName);
    }
    std::sort(ir.emissionDeps.begin(), ir.emissionDeps.end());
    ir.emissionDeps.erase(
        std::unique(ir.emissionDeps.begin(), ir.emissionDeps.end()),
        ir.emissionDeps.end());
    std::sort(ir.resourceDeps.begin(), ir.resourceDeps.end());
    ir.resourceDeps.erase(
        std::unique(ir.resourceDeps.begin(), ir.resourceDeps.end()),
        ir.resourceDeps.end());
  }

  const ICompiled_material *m_material;
  ITransaction *m_transaction;
  EmissionIR *m_ir{nullptr};
  std::unordered_map<int, int> m_temporaryNodes;
};

} // namespace

EmissionIR buildEmissionIR(
    const ICompiled_material *compiledMaterial, ITransaction *transaction)
{
  if (!compiledMaterial || !transaction)
    return {};
  return Builder(compiledMaterial, transaction).build();
}

} // namespace visrtx::libmdl

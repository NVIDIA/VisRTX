// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/scripting/LuaBindings.hpp"
#include "tsd/core/scene/Layer.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include <fmt/format.h>

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerLayerBindings(sol::state &lua)
{
  sol::table tsd = lua["tsd"];

  tsd.new_usertype<core::LayerNodeRef>("LayerNode",
      sol::no_constructor,
      "valid",
      &core::LayerNodeRef::valid,
      "index",
      [](const core::LayerNodeRef &r) -> size_t {
        return r.valid() ? r->index() : core::INVALID_INDEX;
      },
      sol::meta_function::to_string,
      [](const core::LayerNodeRef &r) {
        if (!r.valid())
          return std::string("LayerNode(invalid)");
        return fmt::format("LayerNode({})", r->value().name());
      },
      "parent",
      [](core::LayerNodeRef &r) { return r.valid() ? r->parent() : core::LayerNodeRef{}; },
      "next",
      [](core::LayerNodeRef &r) { return r.valid() ? r->next() : core::LayerNodeRef{}; },
      "sibling",
      [](core::LayerNodeRef &r) { return r.valid() ? r->sibling() : core::LayerNodeRef{}; },
      "isRoot",
      [](const core::LayerNodeRef &r) { return r.valid() && r->isRoot(); },
      "isLeaf",
      [](const core::LayerNodeRef &r) { return r.valid() && r->isLeaf(); },
      "child",
      [](core::LayerNodeRef &r, int idx) -> core::LayerNodeRef {
        if (!r.valid() || idx < 0)
          return {};
        int i = 0;
        for (auto s = r->next(); s && s != r; s = s->sibling()) {
          if (i == idx)
            return s;
          ++i;
        }
        return {};
      },
      "childByName",
      [](core::LayerNodeRef &r, const std::string &name) -> core::LayerNodeRef {
        if (!r.valid())
          return {};
        return find_first_child(r, [&](const core::LayerNodeData &d) {
          return d.name() == name;
        });
      },
      "name",
      sol::property(
          [](const core::LayerNodeRef &r) -> std::string {
            return r.valid() ? r->value().name() : "";
          },
          [](core::LayerNodeRef &r, const std::string &n) {
            if (r.valid())
              r->value().name() = n;
          }),
      "isObject",
      [](const core::LayerNodeRef &r) { return r.valid() && r->value().isObject(); },
      "isTransform",
      [](const core::LayerNodeRef &r) { return r.valid() && r->value().isTransform(); },
      "isEmpty",
      [](const core::LayerNodeRef &r) { return !r.valid() || r->value().isEmpty(); },
      "isEnabled",
      [](const core::LayerNodeRef &r) { return r.valid() && r->value().isEnabled(); },
      "setEnabled",
      [](core::LayerNodeRef &r, bool enabled) {
        if (r.valid())
          r->value().setEnabled(enabled);
      },
      "getTransform",
      [](const core::LayerNodeRef &r) -> math::mat4 {
        return r.valid() ? r->value().getTransform() : math::mat4(math::identity);
      },
      "getTransformSRT",
      [](const core::LayerNodeRef &r) -> math::mat3 {
        return r.valid() ? r->value().getTransformSRT() : math::IDENTITY_MAT3;
      },
      "setAsTransform",
      sol::overload(
          [](core::LayerNodeRef &r, const math::mat4 &m) {
            if (r.valid())
              r->value().setAsTransform(m);
          },
          [](core::LayerNodeRef &r, const math::mat3 &srt) {
            if (r.valid())
              r->value().setAsTransform(srt);
          }),
      "setAsTransformArray",
      sol::overload(
          [](core::LayerNodeRef &r, core::Array &a) {
            if (r.valid())
              r->value().setAsTransformArray(&a);
          },
          [](core::LayerNodeRef &r, core::ArrayRef a) {
            if (r.valid() && a)
              r->value().setAsTransformArray(a.data());
          }),
      "getTransformArray",
      [](const core::LayerNodeRef &r) -> core::Array * {
        return r.valid() ? r->value().getTransformArray() : nullptr;
      });

  using Layer = core::Layer;
  tsd.new_usertype<Layer>("Layer",
      sol::no_constructor,
      "root",
      [](Layer &l) { return l.root(); },
      "size",
      &Layer::size,
      "empty",
      &Layer::empty,
      "at",
      &Layer::at,
      "foreach",
      [](Layer &l, sol::function fn) {
        l.traverse(l.root(), [&fn, &l](core::LayerNode &node, int level) {
            sol::object result = fn(l.at(node.index()), level);
          if (result.is<bool>() && !result.as<bool>())
            return false;
          return true;
        });
      });
}

} // namespace tsd::scripting

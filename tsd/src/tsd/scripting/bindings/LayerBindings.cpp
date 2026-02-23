// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include "tsd/core/scene/Layer.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include "tsd/core/scene/objects/Transform.hpp"
#include "tsd/scripting/LuaBindings.hpp"

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerLayerBindings(sol::state &lua)
{
  sol::table tsd = lua["tsd"];

  tsd.new_usertype<core::LayerNodeRef>(
      "LayerNode",
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
      [](core::LayerNodeRef &r) {
        return r.valid() ? r->parent() : core::LayerNodeRef{};
      },
      "next",
      [](core::LayerNodeRef &r) {
        return r.valid() ? r->next() : core::LayerNodeRef{};
      },
      "sibling",
      [](core::LayerNodeRef &r) {
        return r.valid() ? r->sibling() : core::LayerNodeRef{};
      },
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
        return find_first_child(
            r, [&](const core::LayerNodeData &d) { return d.name() == name; });
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
      [](const core::LayerNodeRef &r) {
        return r.valid() && r->value().isObject();
      },
      "isTransform",
      [](const core::LayerNodeRef &r) {
        return r.valid() && r->value().isTransform();
      },
      "isEmpty",
      [](const core::LayerNodeRef &r) {
        return !r.valid() || r->value().isEmpty();
      },
      "isEnabled",
      [](const core::LayerNodeRef &r) {
        return r.valid() && r->value().isEnabled();
      },
      "setEnabled",
      [](core::LayerNodeRef &r, bool enabled) {
        if (r.valid())
          r->value().setEnabled(enabled);
      },
      "setAsTransform",
      sol::overload(
          [](core::LayerNodeRef &r, const math::mat4 &m) {
            if (!r.valid())
              return;
            if (auto *xfm = r->value().getTransformObject())
              xfm->setTransform(m);
          },
          [](core::LayerNodeRef &r, const math::mat3 &srt) {
            if (!r.valid())
              return;
            if (auto *xfm = r->value().getTransformObject())
              xfm->setTransform(srt);
          }),
      "setAsTransformArray",
      sol::overload(
          [](const core::LayerNodeRef &r, core::Array &a) {
            if (!r.valid())
              return;
            auto arr = a.self();
            if (!arr.valid())
              return;
            if (auto *xfm = r->value().getTransformObject())
              xfm->setTransformArray(arr);
          },
          [](const core::LayerNodeRef &r, core::ArrayRef a) {
            if (!r.valid() || !a.valid())
              return;
            if (auto *xfm = r->value().getTransformObject())
              xfm->setTransformArray(a);
          }),
      "getTransformArray",
      [](const core::LayerNodeRef &r) -> core::Array * {
        if (!r.valid())
          return nullptr;
        if (auto *xfm = r->value().getTransformObject())
          return xfm->getTransformArray();
        return nullptr;
      });

  using Layer = core::Layer;
  tsd.new_usertype<Layer>(
      "Layer",
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

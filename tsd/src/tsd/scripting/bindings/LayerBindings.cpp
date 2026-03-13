// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/scripting/LuaBindings.hpp"
#include "tsd/scene/Layer.hpp"
#include "tsd/scene/objects/Array.hpp"
#include <fmt/format.h>

#include <sol/sol.hpp>

namespace tsd::scripting {

void registerLayerBindings(sol::state &lua)
{
  sol::table tsd = lua["tsd"];

  tsd.new_usertype<scene::LayerNodeRef>("LayerNode",
      sol::no_constructor,
      "valid",
      &scene::LayerNodeRef::valid,
      "index",
      [](const scene::LayerNodeRef &r) -> size_t {
        return r.valid() ? r->index() : core::INVALID_INDEX;
      },
      sol::meta_function::to_string,
      [](const scene::LayerNodeRef &r) {
        if (!r.valid())
          return std::string("LayerNode(invalid)");
        return fmt::format("LayerNode({})", r->value().name());
      },
      "parent",
      [](scene::LayerNodeRef &r) { return r.valid() ? r->parent() : scene::LayerNodeRef{}; },
      "next",
      [](scene::LayerNodeRef &r) { return r.valid() ? r->next() : scene::LayerNodeRef{}; },
      "sibling",
      [](scene::LayerNodeRef &r) { return r.valid() ? r->sibling() : scene::LayerNodeRef{}; },
      "isRoot",
      [](const scene::LayerNodeRef &r) { return r.valid() && r->isRoot(); },
      "isLeaf",
      [](const scene::LayerNodeRef &r) { return r.valid() && r->isLeaf(); },
      "child",
      [](scene::LayerNodeRef &r, int idx) -> scene::LayerNodeRef {
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
      [](scene::LayerNodeRef &r, const std::string &name) -> scene::LayerNodeRef {
        if (!r.valid())
          return {};
        return find_first_child(r, [&](const scene::LayerNodeData &d) {
          return d.name() == name;
        });
      },
      "name",
      sol::property(
          [](const scene::LayerNodeRef &r) -> std::string {
            return r.valid() ? r->value().name() : "";
          },
          [](scene::LayerNodeRef &r, const std::string &n) {
            if (r.valid())
              r->value().name() = n;
          }),
      "isObject",
      [](const scene::LayerNodeRef &r) { return r.valid() && r->value().isObject(); },
      "isTransform",
      [](const scene::LayerNodeRef &r) { return r.valid() && r->value().isTransform(); },
      "isEmpty",
      [](const scene::LayerNodeRef &r) { return !r.valid() || r->value().isEmpty(); },
      "isEnabled",
      [](const scene::LayerNodeRef &r) { return r.valid() && r->value().isEnabled(); },
      "setEnabled",
      [](scene::LayerNodeRef &r, bool enabled) {
        if (r.valid())
          r->value().setEnabled(enabled);
      },
      "getTransform",
      [](const scene::LayerNodeRef &r) -> math::mat4 {
        return r.valid() ? r->value().getTransform() : math::mat4(math::identity);
      },
      "getTransformSRT",
      [](const scene::LayerNodeRef &r) -> math::mat3 {
        return r.valid() ? r->value().getTransformSRT() : math::IDENTITY_MAT3;
      },
      "setAsTransform",
      sol::overload(
          [](scene::LayerNodeRef &r, const math::mat4 &m) {
            if (r.valid())
              r->value().setAsTransform(m);
          },
          [](scene::LayerNodeRef &r, const math::mat3 &srt) {
            if (r.valid())
              r->value().setAsTransform(srt);
          }),
      "setAsTransformArray",
      sol::overload(
          [](scene::LayerNodeRef &r, scene::Array &a) {
            if (r.valid())
              r->value().setAsTransformArray(&a);
          },
          [](scene::LayerNodeRef &r, scene::ArrayRef a) {
            if (r.valid() && a)
              r->value().setAsTransformArray(a.data());
          }),
      "getTransformArray",
      [](const scene::LayerNodeRef &r) -> scene::Array * {
        return r.valid() ? r->value().getTransformArray() : nullptr;
      });

  using Layer = scene::Layer;
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
        l.traverse(l.root(), [&fn, &l](scene::LayerNode &node, int level) {
            sol::object result = fn(l.at(node.index()), level);
          if (result.is<bool>() && !result.as<bool>())
            return false;
          return true;
        });
      });
}

} // namespace tsd::scripting

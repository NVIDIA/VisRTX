// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/tsd_ui_imgui.h"

namespace tsd::ui {

static std::string s_newParameterName;

// Helper functions ///////////////////////////////////////////////////////////

static bool UI_stringList_callback(void *p, int index, const char **out_text)
{
  const auto &stringList = ((tsd::core::Parameter *)p)->stringValues();
  *out_text = stringList[index].c_str();
  return true;
}

static void buildUI_parameter_contextMenu(
    tsd::core::Context &ctx, tsd::core::Object *o, tsd::core::Parameter *p)
{
  if (ImGui::BeginPopup("buildUI_parameter_contextMenu")) {
    if (ImGui::BeginMenu("add new")) {
      ImGui::InputText("name", &s_newParameterName);
      if (ImGui::Button("ok"))
        o->addParameter(s_newParameterName);
      ImGui::EndMenu(); // "add"
    }

    ImGui::Separator();

    if (ImGui::BeginMenu("set type")) {
      if (ImGui::BeginMenu("uniform")) {
        if (ImGui::MenuItem("direction")) {
          p->setValue(tsd::math::float2(0.f));
          p->setUsage(tsd::core::ParameterUsageHint::DIRECTION);
        }

        if (ImGui::BeginMenu("color")) {
          if (ImGui::MenuItem("float3") && p) {
            p->setValue(tsd::math::float3(1));
            p->setUsage(tsd::core::ParameterUsageHint::COLOR);
          }
          if (ImGui::MenuItem("float4") && p) {
            p->setValue(tsd::math::float4(1));
            p->setUsage(tsd::core::ParameterUsageHint::COLOR);
          }
          ImGui::EndMenu(); // "color"
        }

        if (ImGui::BeginMenu("float")) {
          if (ImGui::MenuItem("float1") && p) {
            p->setValue(1.f);
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
          }
          if (ImGui::MenuItem("float2") && p) {
            p->setValue(tsd::math::float2(1.f));
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
          }
          if (ImGui::MenuItem("float3") && p) {
            p->setValue(tsd::math::float3(1.f));
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
          }
          if (ImGui::MenuItem("float4") && p) {
            p->setValue(tsd::math::float4(1.f));
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
          }
          ImGui::EndMenu(); // "float"
        }

        if (ImGui::BeginMenu("int")) {
          if (ImGui::MenuItem("int1") && p) {
            p->setValue(0);
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
          }
          if (ImGui::MenuItem("int2") && p) {
            p->setValue(tsd::math::int2(1));
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
          }
          if (ImGui::MenuItem("int3") && p) {
            p->setValue(tsd::math::int3(1));
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
          }
          if (ImGui::MenuItem("int4") && p) {
            p->setValue(tsd::math::int4(1));
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
          }
          ImGui::EndMenu(); // "float"
        }

        ImGui::EndMenu(); // "uniform"
      }

      ImGui::Separator();

      if (ImGui::MenuItem("attribute")) {
        p->setValue("attribute0");
        p->setStringValues(
            {"attribute0", "attribute1", "attribute2", "attribute3", "color"});
        p->setStringSelection(0);
      }

      ImGui::Separator();

      if (ImGui::BeginMenu("object")) {
        if (ImGui::BeginMenu("new")) {
          if (ImGui::BeginMenu("material")) {
            tsd::core::MaterialRef m;
            if (ImGui::MenuItem("matte")) {
              m = ctx.createObject<tsd::core::Material>(
                  tsd::core::tokens::material::matte);
            }

            if (ImGui::MenuItem("physicallyBased")) {
              m = ctx.createObject<tsd::core::Material>(
                  tsd::core::tokens::material::physicallyBased);
            }

            if (m)
              p->setValue({m->type(), m->index()});
            ImGui::EndMenu(); // "material"
          }
          ImGui::EndMenu(); // "new"
        }

        ImGui::Separator();

#define OBJECT_UI_MENU_ITEM(text, type)                                        \
  if (ImGui::BeginMenu(text)) {                                                \
    if (auto i = buildUI_objects_menulist(ctx, type);                          \
        i != TSD_INVALID_INDEX && p)                                           \
      p->setValue({type, i});                                                  \
    ImGui::EndMenu();                                                          \
  }

        OBJECT_UI_MENU_ITEM("array", ANARI_ARRAY);
        OBJECT_UI_MENU_ITEM("geometry", ANARI_GEOMETRY);
        OBJECT_UI_MENU_ITEM("material", ANARI_MATERIAL);
        OBJECT_UI_MENU_ITEM("sampler", ANARI_SAMPLER);
        OBJECT_UI_MENU_ITEM("spatial field", ANARI_SPATIAL_FIELD);

        ImGui::EndMenu(); // "object"
      }

      ImGui::EndMenu(); // "set type"
    }

    ImGui::Separator();

    if (ImGui::BeginMenu("delete?")) {
      if (ImGui::MenuItem("yes"))
        p->remove();
      ImGui::EndMenu(); // "delete?"
    }

    ImGui::EndPopup();
  }
}

///////////////////////////////////////////////////////////////////////////////

void buildUI_object(tsd::core::Object &o,
    tsd::core::Context &ctx,
    bool useTableForParameters,
    int level)
{
  static anari::DataType typeForSelection = ANARI_UNKNOWN;
  static tsd::core::Parameter *paramForSelection = nullptr;
  static bool openPopup = false;

  ImGui::PushID(&o);

  if (o.type() == ANARI_SURFACE) {
    // no-subtype
    ImGui::Text("[%zu]: '%s'", o.index(), o.name().c_str());
  } else {
    // is-subtyped
    ImGui::Text("[%zu]: '%s' (subtype: '%s')",
        o.index(),
        o.name().c_str(),
        o.subtype().c_str());
  }

  if (o.numParameters() > 0) {
    // regular parameters //

    if (useTableForParameters) {
      const ImGuiTableFlags flags =
          ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable;
      if (ImGui::BeginTable("parameters", 2, flags)) {
        ImGui::TableSetupColumn("Parameter");
        ImGui::TableSetupColumn("Value");
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < o.numParameters(); i++) {
          auto &p = o.parameterAt(i);
          ImGui::TableNextRow();
          buildUI_parameter(o, p, ctx, useTableForParameters);
        }

        ImGui::EndTable();
      }
    } else {
      for (size_t i = 0; i < o.numParameters(); i++)
        buildUI_parameter(o, o.parameterAt(i), ctx);
    }

    // object parameters //

    if (level > 0)
      ImGui::Indent(tsd::ui::INDENT_AMOUNT);

    for (size_t i = 0; i < o.numParameters(); i++) {
      auto &p = o.parameterAt(i);
      auto &pVal = p.value();
      if (!pVal.holdsObject() || anari::isArray(pVal.type()))
        continue;

      ImGui::PushID(i);

      ImGui::NewLine();

      auto *obj = ctx.getObject(pVal);

      static std::string pName;
      pName = p.name().c_str();
      pName += " : ";
      pName += anari::toString(pVal.type());

      ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
      if (ImGui::CollapsingHeader(pName.c_str(), ImGuiTreeNodeFlags_None)) {
        ImGui::BeginDisabled(obj == nullptr);
        if (ImGui::Button("unset"))
          p.setValue({pVal.type()});
        ImGui::EndDisabled();

        ImGui::SameLine();

        if (ImGui::Button("clear"))
          p.setValue({});

        ImGui::SameLine();

        ImGui::BeginDisabled(ctx.numberOfObjects(pVal.type()) == 0);
        if (ImGui::Button("select")) {
          typeForSelection = pVal.type();
          paramForSelection = &p;
          openPopup = true;
        }
        ImGui::EndDisabled();

        if (obj != nullptr)
          buildUI_object(*obj, ctx, useTableForParameters, level + 1);
      }

      ImGui::PopID();
    }

    if (level > 0)
      ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
  }

  ImGui::PopID();

  // popup menu //

  if (level != 0)
    return;

  if (openPopup) {
    ImGui::OpenPopup("buildUI_object_contextMenu");
    openPopup = false;
  }

  if (ImGui::BeginPopup("buildUI_object_contextMenu")) {
    ImGui::Text("%s", anari::toString(typeForSelection));
    ImGui::Separator();
    for (size_t i = 0; i < ctx.numberOfObjects(typeForSelection); i++) {
      auto *obj = ctx.getObject(typeForSelection, i);
      if (!obj)
        continue;

      ImGui::PushID(i);

      static std::string oTitle;
      oTitle = '[';
      oTitle += std::to_string(i);
      oTitle += ']';
      oTitle += obj->name();
      if (ImGui::MenuItem(oTitle.c_str()))
        paramForSelection->setValue({typeForSelection, i});

      ImGui::PopID();
    }
    ImGui::EndPopup();
  }
}

void buildUI_parameter(tsd::core::Object &o,
    tsd::core::Parameter &p,
    tsd::core::Context &ctx,
    bool useTable)
{
  ImGui::PushID(&p);

  bool update = false;

  const char *name = p.name().c_str();

  auto pVal = p.value();
  auto type = pVal.type();
  const auto pMin = p.min();
  const auto pMax = p.max();

  void *value = pVal.data();

  const auto usage = p.usage();
  const bool bounded = pMin || pMax;
  const bool isArray = anari::isArray(type);

  bool enabled = p.isEnabled();

  if (useTable) {
    ImGui::TableSetColumnIndex(0);

    if (ImGui::Checkbox(name, &enabled))
      p.setEnabled(enabled);
    name = "";

    const bool showContextMenu =
        ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right);
    if (showContextMenu) {
      s_newParameterName.reserve(200);
      s_newParameterName = "";
      ImGui::OpenPopup("buildUI_parameter_contextMenu");
    }

    ImGui::TableSetColumnIndex(1);
    ImGui::PushItemWidth(-FLT_MIN); // Right-aligned
  }

  ImGui::BeginDisabled(!enabled);

  switch (type) {
  case ANARI_BOOL:
    update |= ImGui::Checkbox(name, (bool *)value);
    break;
  case ANARI_INT32:
    if (bounded) {
      if (pMin && pMax) {
        update |= ImGui::SliderInt(
            name, (int *)value, pMin.get<int>(), pMax.get<int>());
      } else {
        int min = pMin ? pMin.get<int>() : std::numeric_limits<int>::lowest();
        int max = pMax ? pMax.get<int>() : std::numeric_limits<int>::max();
        update |= ImGui::DragInt(name, (int *)value, 1.f, min, max);
      }
    } else
      update |= ImGui::InputInt(name, (int *)value);
    break;
  case ANARI_FLOAT32:
    if (bounded) {
      if (pMin && pMax) {
        update |= ImGui::SliderFloat(
            name, (float *)value, pMin.get<float>(), pMax.get<float>());
      } else {
        float min =
            pMin ? pMin.get<float>() : std::numeric_limits<float>::lowest();
        float max =
            pMax ? pMax.get<float>() : std::numeric_limits<float>::max();
        update |= ImGui::DragFloat(name, (float *)value, 1.f, min, max);
      }
    } else
      update |= ImGui::DragFloat(name, (float *)value);
    break;
  case ANARI_FLOAT32_VEC2:
  case ANARI_FLOAT32_BOX1:
    if (bounded) {
      if (pMin && pMax) {
        update |= ImGui::SliderFloat2(name,
            (float *)value,
            pMin.get<tsd::math::float2>().x,
            pMax.get<tsd::math::float2>().x);
      } else {
        float min = pMin ? pMin.get<tsd::math::float2>().x
                         : std::numeric_limits<float>::lowest();
        float max = pMax ? pMax.get<tsd::math::float2>().x
                         : std::numeric_limits<float>::max();
        update |= ImGui::DragFloat2(name, (float *)value, 1.f, min, max);
      }
    } else
      update |= ImGui::DragFloat2(name, (float *)value);
    break;
  case ANARI_FLOAT32_VEC3:
    if (usage & tsd::core::ParameterUsageHint::COLOR)
      update |= ImGui::ColorEdit3(name, (float *)value);
    else
      update |= ImGui::DragFloat3(name, (float *)value);
    break;
  case ANARI_FLOAT32_VEC4:
    if (usage & tsd::core::ParameterUsageHint::COLOR)
      update |= ImGui::ColorEdit4(name, (float *)value);
    else
      update |= ImGui::DragFloat4(name, (float *)value);
    break;
  case ANARI_STRING: {
    if (!p.stringValues().empty()) {
      auto ss = p.stringSelection();
      update |= ImGui::Combo(
          name, &ss, UI_stringList_callback, &p, p.stringValues().size());

      if (update) {
        pVal = p.stringValues()[ss].c_str();
        p.setStringSelection(ss);
      }
    } else {
      if (useTable)
        ImGui::Text("\"%s\"", pVal.getString().c_str());
      else
        ImGui::BulletText("%s | '%s'", name, pVal.getString().c_str());
    }
  } break;
  default:
    if (const auto idx = pVal.getAsObjectIndex(); idx != TSD_INVALID_INDEX) {
      if (useTable)
        ImGui::Text("[%zu] %s", idx, anari::toString(type));
      else
        ImGui::BulletText("%s | [%zu] %s", name, idx, anari::toString(type));
    } else {
      if (useTable)
        ImGui::Text("%s", anari::toString(type));
      else
        ImGui::BulletText("%s | %s", name, anari::toString(type));
    }
    break;
  }

  ImGui::EndDisabled();

  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    if (isArray) {
      const auto idx = pVal.getAsObjectIndex();
      const auto &a = *ctx.getObject<tsd::core::Array>(idx);
      ImGui::Text("  idx: [%zu]", idx);
      const auto t = a.type();
      if (t == ANARI_ARRAY3D)
        ImGui::Text(" size: %zu x %zu x %zu", a.dim(0), a.dim(1), a.dim(2));
      else if (t == ANARI_ARRAY2D)
        ImGui::Text(" size: %zu x %zu", a.dim(0), a.dim(1));
      else
        ImGui::Text(" size: %zu", a.dim(0));
      ImGui::Text(" type: %s", anari::toString(a.elementType()));
    } else {
      if (p.description().empty())
        ImGui::Text("%s", anari::toString(type));
      else
        ImGui::Text("%s | %s", anari::toString(type), p.description().c_str());
    }
    ImGui::EndTooltip();
  }

  {
    const bool showContextMenu =
        ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right);
    if (showContextMenu) {
      s_newParameterName.reserve(200);
      s_newParameterName = "";
      ImGui::OpenPopup("buildUI_parameter_contextMenu");
    }
  }

  if (update)
    p.setValue(pVal);

  buildUI_parameter_contextMenu(
      ctx, &o, &p); // NOTE: 'p' can be deleted after this

  ImGui::PopID();
}

size_t buildUI_objects_menulist(
    const tsd::core::Context &ctx, anari::DataType type)
{
  size_t retval = TSD_INVALID_INDEX;

  for (size_t i = 0; i < ctx.numberOfObjects(type); i++) {
    auto *obj = ctx.getObject(type, i);
    if (!obj)
      continue;

    ImGui::PushID(i);

    static std::string oTitle;
    oTitle = '[';
    oTitle += std::to_string(i);
    oTitle += ']';
    oTitle += obj->name();
    if (ImGui::MenuItem(oTitle.c_str()))
      retval = i;

    ImGui::PopID();
  }

  return retval;
}

} // namespace tsd::ui
// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/tsd_ui_imgui.h"
// tsd_core
#include "tsd/core/ColorMapUtil.hpp"

namespace tsd::ui {

static std::string s_newParameterName;

// Helper functions ///////////////////////////////////////////////////////////

static bool UI_stringList_callback(void *p, int index, const char **out_text)
{
  const auto &stringList = ((tsd::core::Parameter *)p)->stringValues();
  *out_text = stringList[index].c_str();
  return true;
}

static void buildUI_array_info_tooltip_text(
    const tsd::core::Scene &scene, size_t idx)
{
  const auto &a = *scene.getObject<tsd::core::Array>(idx);
  ImGui::Text(" idx: [%zu]", idx);
  ImGui::Text("name: '%s'", a.name().c_str());
  const auto t = a.type();
  if (t == ANARI_ARRAY3D)
    ImGui::Text("size: %zu x %zu x %zu", a.dim(0), a.dim(1), a.dim(2));
  else if (t == ANARI_ARRAY2D)
    ImGui::Text("size: %zu x %zu", a.dim(0), a.dim(1));
  else
    ImGui::Text("size: %zu", a.dim(0));
  ImGui::Text("type: %s", anari::toString(a.elementType()));
}

static void buildUI_parameter_contextMenu(
    tsd::core::Scene &scene, tsd::core::Object *o, tsd::core::Parameter *p)
{
  if (ImGui::BeginPopup("buildUI_parameter_contextMenu")) {
    if (ImGui::BeginMenu("set type")) {
      if (ImGui::BeginMenu("uniform")) {
        if (ImGui::MenuItem("direction")) {
          p->setUsage(tsd::core::ParameterUsageHint::DIRECTION);
          p->setValue(tsd::math::float2(0.f));
        }

        if (ImGui::BeginMenu("color")) {
          if (ImGui::MenuItem("rgb") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::COLOR);
            p->setValue(tsd::math::float3(1));
          }
          if (ImGui::MenuItem("rgba") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::COLOR);
            p->setValue(tsd::math::float4(1));
          }
          ImGui::EndMenu(); // "color"
        }

        if (ImGui::BeginMenu("transform")) {
          if (ImGui::MenuItem("identity") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
            p->setValue(tsd::math::scaling_matrix(tsd::math::float3(1.f)));
          }
          ImGui::Separator();
          if (ImGui::MenuItem("value range") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::VALUE_RANGE_TRANSFORM);
            p->setValue(tsd::math::float2(0.f, 1.f));
          }
          ImGui::EndMenu(); // "transform"
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("float")) {
          if (ImGui::MenuItem("float1") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
            p->setValue(1.f);
          }
          if (ImGui::MenuItem("float2") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
            p->setValue(tsd::math::float2(1.f));
          }
          if (ImGui::MenuItem("float3") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
            p->setValue(tsd::math::float3(1.f));
          }
          if (ImGui::MenuItem("float4") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
            p->setValue(tsd::math::float4(1.f));
          }
          ImGui::EndMenu(); // "float"
        }

        if (ImGui::BeginMenu("int")) {
          if (ImGui::MenuItem("int1") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
            p->setValue(0);
          }
          if (ImGui::MenuItem("int2") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
            p->setValue(tsd::math::int2(1));
          }
          if (ImGui::MenuItem("int3") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
            p->setValue(tsd::math::int3(1));
          }
          if (ImGui::MenuItem("int4") && p) {
            p->setUsage(tsd::core::ParameterUsageHint::NONE);
            p->setValue(tsd::math::int4(1));
          }
          ImGui::EndMenu(); // "float"
        }

        ImGui::EndMenu(); // "uniform"
      }

      ImGui::Separator();

      if (ImGui::MenuItem("attribute"))
        p->setToAttribute();

      ImGui::Separator();

      if (ImGui::BeginMenu("object")) {
        if (ImGui::BeginMenu("new")) {
          if (ImGui::BeginMenu("array")) {
            tsd::core::ArrayRef a;

            if (ImGui::BeginMenu("color map (RGB)")) {
#define OBJECT_UI_MENU_ITEM(text, name)                                        \
  if (ImGui::MenuItem(text)) {                                                 \
    a = scene.createArray(ANARI_FLOAT32_VEC3, 128);                            \
    auto colormap = tsd::core::resampleArray(tsd::core::colormap::name, 128);  \
    a->setData(colormap);                                                      \
    a->setName("colormap_" #name);                                             \
  }
              OBJECT_UI_MENU_ITEM("jet", jet);
              OBJECT_UI_MENU_ITEM("cool to warm", cool_to_warm);
              OBJECT_UI_MENU_ITEM("viridis", viridis);
              OBJECT_UI_MENU_ITEM("black body", black_body);
              OBJECT_UI_MENU_ITEM("inferno", inferno);
              OBJECT_UI_MENU_ITEM("ice fire", ice_fire);
              OBJECT_UI_MENU_ITEM("grayscale", grayscale);
#undef OBJECT_UI_MENU_ITEM
              ImGui::EndMenu(); // "color map"
            }

            if (a)
              p->setValue({a->type(), a->index()});
            ImGui::EndMenu(); // "array"
          }

          if (ImGui::BeginMenu("material")) {
            tsd::core::MaterialRef m;
            if (ImGui::MenuItem("matte")) {
              m = scene.createObject<tsd::core::Material>(
                  tsd::core::tokens::material::matte);
            }

            if (ImGui::MenuItem("physicallyBased")) {
              m = scene.createObject<tsd::core::Material>(
                  tsd::core::tokens::material::physicallyBased);
            }

            if (ImGui::MenuItem("mdl")) {
              m = scene.createObject<tsd::core::Material>(
                  tsd::core::tokens::material::mdl);
            }

            if (m)
              p->setValue({m->type(), m->index()});
            ImGui::EndMenu(); // "material"
          }

          if (ImGui::BeginMenu("geometry")) {
            tsd::core::GeometryRef g;

#define OBJECT_UI_MENU_ITEM(text, subtype)                                     \
  if (ImGui::MenuItem(text)) {                                                 \
    g = scene.createObject<tsd::core::Geometry>(                               \
        tsd::core::tokens::geometry::subtype);                                 \
  }
            OBJECT_UI_MENU_ITEM("cone", cone);
            OBJECT_UI_MENU_ITEM("curve", curve);
            OBJECT_UI_MENU_ITEM("cylinder", cylinder);
            OBJECT_UI_MENU_ITEM("isosurface", isosurface);
            OBJECT_UI_MENU_ITEM("neural", neural);
            OBJECT_UI_MENU_ITEM("quad", quad);
            OBJECT_UI_MENU_ITEM("sphere", sphere);
            OBJECT_UI_MENU_ITEM("triangle", triangle);
#undef OBJECT_UI_MENU_ITEM

            if (g)
              p->setValue({g->type(), g->index()});
            ImGui::EndMenu(); // "geometry"
          }

          if (ImGui::BeginMenu("sampler")) {
            tsd::core::SamplerRef s;

#define OBJECT_UI_MENU_ITEM(text, subtype)                                     \
  if (ImGui::MenuItem(text)) {                                                 \
    s = scene.createObject<tsd::core::Sampler>(                                \
        tsd::core::tokens::sampler::subtype);                                  \
  }
            OBJECT_UI_MENU_ITEM("compressedImage2D", compressedImage2D);
            OBJECT_UI_MENU_ITEM("image1D", image1D);
            OBJECT_UI_MENU_ITEM("image2D", image2D);
            OBJECT_UI_MENU_ITEM("image3D", image3D);
            OBJECT_UI_MENU_ITEM("primitive", primitive);
            OBJECT_UI_MENU_ITEM("transform", transform);
#undef OBJECT_UI_MENU_ITEM

            if (s)
              p->setValue({s->type(), s->index()});
            ImGui::EndMenu(); // "material"
          }

          ImGui::EndMenu(); // "new"
        }

        ImGui::Separator();

#define OBJECT_UI_MENU_ITEM(text, type)                                        \
  if (ImGui::BeginMenu(text)) {                                                \
    auto t = type;                                                             \
    if (auto i = buildUI_objects_menulist(scene, t);                           \
        i != TSD_INVALID_INDEX && p)                                           \
      p->setValue({t, i});                                                     \
    ImGui::EndMenu();                                                          \
  }
        OBJECT_UI_MENU_ITEM("array", ANARI_ARRAY);
        OBJECT_UI_MENU_ITEM("geometry", ANARI_GEOMETRY);
        OBJECT_UI_MENU_ITEM("material", ANARI_MATERIAL);
        OBJECT_UI_MENU_ITEM("sampler", ANARI_SAMPLER);
        OBJECT_UI_MENU_ITEM("spatial field", ANARI_SPATIAL_FIELD);
#undef OBJECT_UI_MENU_ITEM

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
} // namespace tsd::ui

///////////////////////////////////////////////////////////////////////////////

void buildUI_object(tsd::core::Object &o,
    tsd::core::Scene &scene,
    bool useTableForParameters,
    int level)
{
  static anari::DataType typeForSelection = ANARI_UNKNOWN;
  static tsd::core::Parameter *paramForSelection = nullptr;
  static bool openContextMenu = false;

  ImGui::PushID(&o);

  ImGui::Text("[%zu]: ", o.index());
  ImGui::SameLine();
  ImGui::InputText("##name", &o.editableName());
  ImGui::Separator();

  if (anari::isArray(o.type())) {
    const auto &a = *(tsd::core::Array *)&o;
    const auto t = a.type();
    ImGui::Text("%s", anari::toString(t));
    if (t == ANARI_ARRAY3D)
      ImGui::Text(" size: %zu x %zu x %zu", a.dim(0), a.dim(1), a.dim(2));
    else if (t == ANARI_ARRAY2D)
      ImGui::Text(" size: %zu x %zu", a.dim(0), a.dim(1));
    else
      ImGui::Text(" size: %zu", a.dim(0));
    ImGui::Text(" type: %s", anari::toString(a.elementType()));
  } else if (o.type() != ANARI_SURFACE) {
    ImGui::Text("   subtype: %s", o.subtype().c_str());
  }

  ImGui::Text("use counts: %zu | %zu | %zu",
      o.useCount(tsd::core::Object::UseKind::APP),
      o.useCount(tsd::core::Object::UseKind::PARAMETER),
      o.useCount(tsd::core::Object::UseKind::LAYER));
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "references to this object:"
        " application | parameter | layer");
  }

  ImGui::Separator();

  // regular parameters //

  if (useTableForParameters) {
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg
        | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame;
    if (ImGui::BeginTable("parameters", 2, flags)) {
      ImGui::TableSetupColumn("Parameter");
      ImGui::TableSetupColumn("Value");
      ImGui::TableHeadersRow();

      for (size_t i = 0; i < o.numParameters(); i++) {
        auto &p = o.parameterAt(i);
        ImGui::TableNextRow();
        buildUI_parameter(o, p, scene, useTableForParameters);
      }

      ImGui::EndTable();
    }
  } else {
    for (size_t i = 0; i < o.numParameters(); i++)
      buildUI_parameter(o, o.parameterAt(i), scene);
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

    auto *obj = scene.getObject(pVal);

    static std::string pName;
    pName = p.name().c_str();
    pName += " : ";
    pName += anari::toString(pVal.type());

    ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
    if (ImGui::CollapsingHeader(pName.c_str(), ImGuiTreeNodeFlags_None)) {
      ImGui::BeginDisabled(obj == nullptr);
      if (ImGui::Button("unset"))
        p.setValue({pVal.type()});
      ImGui::EndDisabled();

      ImGui::SameLine();

      if (ImGui::Button("clear"))
        p.setValue({});

      ImGui::SameLine();

      ImGui::BeginDisabled(scene.numberOfObjects(pVal.type()) == 0);
      if (ImGui::Button("select")) {
        typeForSelection = pVal.type();
        paramForSelection = &p;
        openContextMenu = true;
      }
      ImGui::EndDisabled();

      if (obj != nullptr)
        buildUI_object(*obj, scene, useTableForParameters, level + 1);
    }

    ImGui::PopID();
  }

  if (level > 0)
    ImGui::Unindent(tsd::ui::INDENT_AMOUNT);

  ImGui::Separator();

  if (ImGui::Button("add parameter")) {
    s_newParameterName.reserve(200);
    s_newParameterName = "";
    ImGui::OpenPopup("buildUI_newParameter_popupMenu");
  }

  if (ImGui::BeginPopup("buildUI_newParameter_popupMenu")) {
    ImGui::InputText("name", &s_newParameterName);
    if (ImGui::Button("ok"))
      o.addParameter(s_newParameterName);
    ImGui::EndPopup();
  }

  ImGui::PopID();

  // context menu //

  if (level != 0)
    return;

  if (openContextMenu) {
    ImGui::OpenPopup("buildUI_object_contextMenu");
    openContextMenu = false;
  }

  if (ImGui::BeginPopup("buildUI_object_contextMenu")) {
    ImGui::Text("%s", anari::toString(typeForSelection));
    ImGui::Separator();
    for (size_t i = 0; i < scene.numberOfObjects(typeForSelection); i++) {
      auto *obj = scene.getObject(typeForSelection, i);
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

bool buildUI_parameter(tsd::core::Object &o,
    tsd::core::Parameter &p,
    tsd::core::Scene &scene,
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

    const bool showSceneMenu =
        ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right);
    if (showSceneMenu)
      ImGui::OpenPopup("buildUI_parameter_contextMenu");

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
  case ANARI_FLOAT32_BOX2:
    ImGui::PushID(name);
    ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
    if (ImGui::CollapsingHeader(name)) {
      if (bounded) {
        if (pMin && pMax) {
          update |= ImGui::SliderFloat2("lower",
              (float *)value,
              pMin.get<tsd::math::box2>().lower.x,
              pMax.get<tsd::math::box2>().lower.x);
          update |= ImGui::SliderFloat2("upper",
              (float *)value + 2,
              pMin.get<tsd::math::box2>().upper.x,
              pMax.get<tsd::math::box2>().upper.x);
        } else {
          float minLower = pMin ? pMin.get<tsd::math::box2>().lower.x
                                : std::numeric_limits<float>::lowest();
          float maxLower = pMax ? pMax.get<tsd::math::box2>().lower.x
                                : std::numeric_limits<float>::max();
          update |=
              ImGui::DragFloat2("lower", (float *)value, 1.f, minLower, maxLower);
          float minUpper = pMin ? pMin.get<tsd::math::box2>().upper.x
                                : std::numeric_limits<float>::lowest();
          float maxUpper = pMax ? pMax.get<tsd::math::box2>().upper.x
                                : std::numeric_limits<float>::max();
          update |= ImGui::DragFloat2(
              "upper", (float *)value + 2, 1.f, minUpper, maxUpper);
        }
      } else {
        update |= ImGui::DragFloat2("lower", (float *)value);
        update |= ImGui::DragFloat2("upper", (float *)value + 2);
      }
    }
    ImGui::PopID();
    break;
  case ANARI_FLOAT32_BOX3:
    ImGui::PushID(name);
    ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
    if (ImGui::CollapsingHeader(name)) {
      if (bounded) {
        if (pMin && pMax) {
          update |= ImGui::SliderFloat3("lower",
              (float *)value,
              pMin.get<tsd::math::box3>().lower.x,
              pMax.get<tsd::math::box3>().lower.x);
          update |= ImGui::SliderFloat3("upper",
              (float *)value + 3,
              pMin.get<tsd::math::box3>().upper.x,
              pMax.get<tsd::math::box3>().upper.x);
        } else {
          float minLower = pMin ? pMin.get<tsd::math::box3>().lower.x
                                : std::numeric_limits<float>::lowest();
          float maxLower = pMax ? pMax.get<tsd::math::box3>().lower.x
                                : std::numeric_limits<float>::max();
          update |=
              ImGui::DragFloat3("lower", (float *)value, 1.f, minLower, maxLower);
          float minUpper = pMin ? pMin.get<tsd::math::box3>().upper.x
                                : std::numeric_limits<float>::lowest();
          float maxUpper = pMax ? pMax.get<tsd::math::box3>().upper.x
                                : std::numeric_limits<float>::max();
          update |= ImGui::DragFloat3(
              "upper", (float *)value + 3, 1.f, minUpper, maxUpper);
        }
      } else {
        update |= ImGui::DragFloat3("lower", (float *)value);
        update |= ImGui::DragFloat3("upper", (float *)value + 3);
      }
    }
    ImGui::PopID();
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
      if (useTable) {
        auto val = pVal.getString();
        update |= ImGui::InputText(
            "\"%s\"", &val, ImGuiInputTextFlags_EnterReturnsTrue);
        pVal = val.c_str();
      } else
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
      buildUI_array_info_tooltip_text(scene, idx);
    } else if (type == ANARI_FLOAT32_MAT4) {
      auto *value_f = (const float *)value;
      ImGui::Text("[%.3f %.3f %.3f %.3f]",
          value_f[0],
          value_f[4],
          value_f[8],
          value_f[12]);
      ImGui::Text("[%.3f %.3f %.3f %.3f]",
          value_f[1],
          value_f[5],
          value_f[9],
          value_f[13]);
      ImGui::Text("[%.3f %.3f %.3f %.3f]",
          value_f[2],
          value_f[6],
          value_f[10],
          value_f[14]);
      ImGui::Text("[%.3f %.3f %.3f %.3f]",
          value_f[3],
          value_f[7],
          value_f[11],
          value_f[15]);
    } else {
      if (p.description().empty())
        ImGui::Text("%s", anari::toString(type));
      else
        ImGui::Text("%s | %s", anari::toString(type), p.description().c_str());
    }
    ImGui::EndTooltip();
  }

  {
    const bool showSceneMenu =
        ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right);
    if (showSceneMenu) {
      s_newParameterName.reserve(200);
      s_newParameterName = "";
      ImGui::OpenPopup("buildUI_parameter_contextMenu");
    }
  }

  if (update)
    p.setValue(pVal);

  buildUI_parameter_contextMenu(
      scene, &o, &p); // NOTE: 'p' can be deleted after this

  ImGui::PopID();

  return update;
}

size_t buildUI_objects_menulist(
    const tsd::core::Scene &scene, anari::DataType &type)
{
  size_t retval = TSD_INVALID_INDEX;

  for (size_t i = 0; i < scene.numberOfObjects(type); i++) {
    auto *obj = scene.getObject(type, i);
    if (!obj)
      continue;

    ImGui::PushID(i);

    static std::string oTitle;
    oTitle = '[';
    oTitle += std::to_string(i);
    oTitle += ']';
    oTitle += obj->name();
    if (ImGui::MenuItem(oTitle.c_str())) {
      retval = i;
      type = obj->type();
    }

    if (anari::isArray(type) && ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      buildUI_array_info_tooltip_text(scene, i);
      ImGui::EndTooltip();
    }

    ImGui::PopID();
  }

  return retval;
}

} // namespace tsd::ui
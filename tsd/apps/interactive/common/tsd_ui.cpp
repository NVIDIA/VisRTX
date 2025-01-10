// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd_ui.h"

namespace tsd::ui {

// Helper functions ///////////////////////////////////////////////////////////

static Any parseValue(ANARIDataType type, const void *mem)
{
  if (type == ANARI_STRING)
    return Any(ANARI_STRING, "");
  else if (anari::isObject(type)) {
    ANARIObject nullHandle = ANARI_INVALID_HANDLE;
    return Any(type, &nullHandle);
  } else if (mem)
    return Any(type, mem);
  else
    return {};
}

static bool UI_stringList_callback(void *p, int index, const char **out_text)
{
  const auto &stringList = ((Parameter *)p)->stringValues();
  *out_text = stringList[index].c_str();
  return true;
}

///////////////////////////////////////////////////////////////////////////////

void buildUI_object(tsd::Object &o,
    const tsd::Context &ctx,
    bool useTableForParameters,
    int level)
{
  static anari::DataType typeForSelection = ANARI_UNKNOWN;
  static tsd::Parameter *paramForSelection = nullptr;
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
          if (p.value().holdsObject() && !anari::isArray(p.value().type()))
            continue;
          ImGui::TableNextRow();
          buildUI_parameter(p, ctx, useTableForParameters);
        }

        ImGui::EndTable();
      }
    } else {
      for (size_t i = 0; i < o.numParameters(); i++)
        buildUI_parameter(o.parameterAt(i), ctx);
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

void buildUI_parameter(
    tsd::Parameter &p, const tsd::Context &ctx, bool useTable)
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
        update |= ImGui::SliderFloat2(
            name, (float *)value, pMin.get<float2>().x, pMax.get<float2>().x);
      } else {
        float min =
            pMin ? pMin.get<float2>().x : std::numeric_limits<float>::lowest();
        float max =
            pMax ? pMax.get<float2>().x : std::numeric_limits<float>::max();
        update |= ImGui::DragFloat2(name, (float *)value, 1.f, min, max);
      }
    } else
      update |= ImGui::DragFloat2(name, (float *)value);
    break;
  case ANARI_FLOAT32_VEC3:
    if (usage & tsd::ParameterUsageHint::COLOR)
      update |= ImGui::ColorEdit3(name, (float *)value);
    else
      update |= ImGui::DragFloat3(name, (float *)value);
    break;
  case ANARI_FLOAT32_VEC4:
    if (usage & tsd::ParameterUsageHint::COLOR)
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
#if 1
      if (useTable)
        ImGui::Text("\"%s\"", pVal.getString().c_str());
      else
        ImGui::BulletText("%s | '%s'", name, pVal.getString().c_str());
#else
      constexpr int MAX_LENGTH = 2000;
      p.value.reserveString(MAX_LENGTH);

      if (ImGui::Button("...")) {
        nfdchar_t *outPath = nullptr;
        nfdfilteritem_t filterItem[1] = {{"OBJ Files", "obj"}};
        nfdresult_t result = NFD_OpenDialog(&outPath, filterItem, 1, nullptr);
        if (result == NFD_OKAY) {
          p.value = std::string(outPath).c_str();
          update = true;
          NFD_FreePath(outPath);
        } else {
          printf("NFD Error: %s\n", NFD_GetError());
        }
      }

      ImGui::SameLine();

      auto text_cb = [](ImGuiInputTextCallbackData *cbd) {
        auto &p = *(ui::Parameter *)cbd->UserData;
        p.value.resizeString(cbd->BufTextLen);
        return 0;
      };
      update |= ImGui::InputText(name,
          (char *)value,
          MAX_LENGTH,
          ImGuiInputTextFlags_CallbackEdit,
          text_cb,
          &p);
#endif
    }
  } break;
  default:
    if (useTable)
      ImGui::Text("%s", anari::toString(type));
    else
      ImGui::BulletText("%s | %s", name, anari::toString(type));
    break;
  }

  ImGui::EndDisabled();

  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    if (isArray) {
      const auto &a = *ctx.getObject<Array>(pVal.getAsObjectIndex());
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

  if (update)
    p.setValue(pVal);

  ImGui::PopID();
}

void addDefaultRendererParameters(Object &o)
{
  o.addParameter("background")
      .setValue(float4(0.05f, 0.05f, 0.05f, 1.f))
      .setDescription("background color")
      .setUsage(ParameterUsageHint::COLOR);
  o.addParameter("ambientRadiance")
      .setValue(0.25f)
      .setDescription("intensity of ambient light")
      .setMin(0.f);
  o.addParameter("ambientColor")
      .setValue(float3(1.f))
      .setDescription("color of ambient light")
      .setUsage(ParameterUsageHint::COLOR);
}

Object parseANARIObject(
    anari::Device d, ANARIDataType objectType, const char *subtype)
{
  Object retval(objectType, subtype);

  if (objectType == ANARI_RENDERER)
    addDefaultRendererParameters(retval);

  auto *parameter = (const ANARIParameter *)anariGetObjectInfo(
      d, objectType, subtype, "parameter", ANARI_PARAMETER_LIST);

  for (; parameter && parameter->name != nullptr; parameter++) {
    tsd::Token name(parameter->name);
    if (retval.parameter(name))
      continue;

    auto *description = (const char *)anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "description",
        ANARI_STRING);

    const void *defaultValue = anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "default",
        parameter->type);

    const void *minValue = anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "minimum",
        parameter->type);

    const void *maxValue = anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "maximum",
        parameter->type);

    const auto **stringValues = (const char **)anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "value",
        ANARI_STRING_LIST);

    auto &p = retval.addParameter(name);
    p.setValue(Any(parameter->type, nullptr));
    p.setDescription(description ? description : "");
    p.setValue(parseValue(parameter->type, defaultValue));
    if (minValue)
      p.setMin(parseValue(parameter->type, minValue));
    if (maxValue)
      p.setMax(parseValue(parameter->type, maxValue));

    std::vector<std::string> svs;
    for (; stringValues && *stringValues; stringValues++)
      svs.push_back(*stringValues);
    if (!svs.empty())
      p.setStringValues(svs);
  }

  return retval;
}

} // namespace tsd::ui
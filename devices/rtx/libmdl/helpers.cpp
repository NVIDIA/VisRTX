// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "Core.h"

#include <mi/base/enums.h>
#include <mi/base/handle.h>
#include <mi/base/ilogger.h>
#include <mi/neuraylib/ilogging_configuration.h>

#include <fmt/format.h>

#include <cstdarg>
#include <string>

using namespace std::string_literals;

namespace visrtx::libmdl {

std::tuple<std::string, std::string> parseMaterialSourceName(
    std::string_view name, Core *core)
{
  if (name.empty()) {
    core->logMessage(mi::base::MESSAGE_SEVERITY_WARNING,
        "The provided material name is empty.");
    return {};
  }

  std::string outModuleName;
  std::string outMaterialName;

  outModuleName = "";
  outMaterialName = "";
  auto p_left_paren = name.rfind('(');
  if (p_left_paren == std::string::npos)
    p_left_paren = name.size();
  auto p_last = name.rfind("::", p_left_paren - 1);

  auto starts_with_colons =
      name.length() > 2 && name[0] == ':' && name[1] == ':';

  // check for mdle
  if (!starts_with_colons) {
    auto potential_path = std::string(name);
    auto potential_material_name = "main"s;

    // input already has ::main attached (optional)
    if (p_last != std::string::npos) {
      potential_path = name.substr(0, p_last);
      potential_material_name = name.substr(p_last + 2, name.size() - p_last);
    }

    // is it an mdle?
    if (potential_path.length() >= 5
        && potential_path.substr(potential_path.length() - 5) == ".mdle") {
      if (potential_material_name != "main") {
        core->logMessage(mi::base::MESSAGE_SEVERITY_INFO,
            "Material and module name cannot be extracted from "
            "'{}'.\nThe module was detected as MDLE but the selected material is "
            "different from 'main'",
            name);
        return {};
      }
      outModuleName = potential_path;
      outMaterialName = potential_material_name;
      return {outModuleName, outMaterialName};
    }
  }

  if (!starts_with_colons) {
    core->logMessage(mi::base::MESSAGE_SEVERITY_INFO,
        "The provided name '{}' is not an absolute fully-qualified"
        " material name, a leading '::' has been added.",
        name);
    outModuleName = "::";
  }

  outModuleName.append(name.substr(0, p_last));
  outMaterialName = name.substr(p_last + 2, name.size() - p_last);
  return {outModuleName, outMaterialName};
}

} // namespace visrtx::libmdl

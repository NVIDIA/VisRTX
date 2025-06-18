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

std::tuple<std::string, std::string> parseCmdArgumentMaterialName(
    std::string_view argument, Core *core)
{
  std::string outModuleName;
  std::string outMaterialName;

  outModuleName = "";
  outMaterialName = "";
  auto p_left_paren = argument.rfind('(');
  if (p_left_paren == std::string::npos)
    p_left_paren = argument.size();
  auto p_last = argument.rfind("::", p_left_paren - 1);

  auto starts_with_colons =
      argument.length() > 2 && argument[0] == ':' && argument[1] == ':';

  // check for mdle
  if (!starts_with_colons) {
    auto potential_path = std::string(argument);
    auto potential_material_name = "main"s;

    // input already has ::main attached (optional)
    if (p_last != std::string::npos) {
      potential_path = argument.substr(0, p_last);
      potential_material_name =
          argument.substr(p_last + 2, argument.size() - p_last);
    }

    // is it an mdle?
    if (potential_path.length() >= 5
        && potential_path.substr(potential_path.length() - 5) == ".mdle") {
      if (potential_material_name != "main") {
        core->logMessage(mi::base::MESSAGE_SEVERITY_INFO,
                       "Material and module name cannot be extracted from "
                "'{}'.\nThe module was detected as MDLE but the selected material is "
                "different from 'main'",
                argument);
        return {};
      }
      outModuleName = potential_path;
      outMaterialName = potential_material_name;
      return {outModuleName, outMaterialName};
    }
  }

  if (!starts_with_colons) {
    core->logMessage(mi::base::MESSAGE_SEVERITY_INFO,
            "The provided argument '{}' is not an absolute fully-qualified"
            " material name, a leading '::' has been added.",
            argument);
    outModuleName = "::";
  }

  outModuleName.append(argument.substr(0, p_last));
  outMaterialName = argument.substr(p_last + 2, argument.size() - p_last);
  return {outModuleName, outMaterialName};
}

} // namespace visrtx::libmdl

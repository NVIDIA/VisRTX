#pragma once

#include "Core.h"

#include <string>
#include <string_view>
#include <tuple>

using namespace std::string_literals;

namespace visrtx::libmdl {

std::tuple<std::string, std::string> parseCmdArgumentMaterialName(
    std::string_view argument, Core *logger);

} // namespace visrtx::libmdl

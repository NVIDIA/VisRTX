// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ArrayHelpers.hpp"
#include "../ObjectPoolRefRegister.hpp"
#include "../ObjectRefBindings.hpp"
#include "tsd/scene/objects/Array.hpp"
#include "tsd/scripting/Sol2Helpers.hpp"

#include <sol/sol.hpp>

#include <stdexcept>

namespace tsd::scripting {

void registerArrayRef(sol::table &tsd)
{
  auto arrayRefType = registerObjectPoolRef<scene::Array>(tsd, "Array");

  arrayRefType["elementType"] = +[](const scene::ArrayRef &r) -> anari::DataType {
    return r.valid() ? r.data()->elementType() : ANARI_UNKNOWN;
  };
  arrayRefType["size"] = +[](const scene::ArrayRef &r) -> size_t {
    return r.valid() ? r.data()->size() : 0;
  };
  arrayRefType["elementSize"] = +[](const scene::ArrayRef &r) -> size_t {
    return r.valid() ? r.data()->elementSize() : 0;
  };
  arrayRefType["isEmpty"] = +[](const scene::ArrayRef &r) -> bool {
    return r.valid() ? r.data()->isEmpty() : true;
  };
  arrayRefType["dim"] = +[](const scene::ArrayRef &r, size_t d) -> size_t {
    return r.valid() ? r.data()->dim(d) : 0;
  };
  arrayRefType["setData"] =
      [](scene::ArrayRef &r, sol::table data, sol::this_state s) {
        if (!r.valid())
          throw std::runtime_error("attempt to setData on invalid Array");
        arraySetDataFromLua(*r.data(), data, s);
      };
  arrayRefType["getData"] = [](scene::ArrayRef &r, sol::this_state s) {
    if (!r.valid())
      throw std::runtime_error("attempt to getData on invalid Array");
    return arrayGetDataAsLua(*r.data(), s);
  };
}

} // namespace tsd::scripting

// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "libmdl/Core.h"
#include "libmdl/SceneDataDescriptor.h"
#include "libmdl/source_name_utils.h"

#include <mi/base/handle.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

using visrtx::libmdl::Core;
using visrtx::libmdl::SceneDataDescriptor;

namespace {

void require(bool condition, const char *message)
{
  if (!condition)
    throw std::runtime_error(message);
}

void testValueSemantics()
{
  SceneDataDescriptor descriptor;
  require(descriptor.m_namesById.empty(), "default descriptor is not empty");
  require(!descriptor.find("missing"), "empty descriptor resolved a name");

  descriptor.m_namesById = {"", "temperature", "", "tint", "Temperature"};
  require(descriptor.find("temperature") == 1, "IDs were compacted");
  require(descriptor.find("tint") == 3, "empty slot changed subsequent IDs");
  require(descriptor.find("Temperature") == 4, "names are not case sensitive");
  require(!descriptor.find("TEMPERATURE"), "case mismatch resolved a name");
  require(!descriptor.find(""), "empty/reserved name was exposed");
  require(!descriptor.find("missing"), "unknown name resolved");

  auto copy = descriptor;
  descriptor.m_namesById[1] = "changed";
  require(copy.find("temperature") == 1, "copy shares mutable strings");
  SceneDataDescriptor assigned;
  assigned = copy;
  copy.m_namesById.clear();
  require(assigned.find("tint") == 3, "copy assignment lost ownership");
  auto moved = std::move(assigned);
  require(moved.find("temperature") == 1, "move lost original IDs");
}

SceneDataDescriptor extract(Core &core,
    mi::neuraylib::ITransaction *transaction,
    const std::string &name)
{
  using mi::base::make_handle;
  const std::string source =
      "mdl 1.7; import ::df::*; import ::scene::*;\n"
      "export material main() = material(surface: material_surface("
      "emission: material_emission(emission: df::diffuse_edf(),"
      " intensity: scene::data_lookup_color(\""
      + name + "\", color(0)))));\n";
  const auto moduleName = visrtx::libmdl::makeInlineModuleName(source);
  auto module =
      make_handle(core.loadModuleFromString(moduleName, source, transaction));
  require(module.is_valid_interface(), "could not load test module");
  auto definition = make_handle(
      core.getFunctionDefinition(module.get(), "main", transaction));
  require(definition.is_valid_interface(), "could not find test material");
  auto compiled = make_handle(core.getCompiledMaterial(definition.get()));
  require(compiled.is_valid_interface(), "could not compile test material");
  auto target = make_handle(core.getPtxTargetCode(compiled.get(), transaction));
  require(target.is_valid_interface(), "could not generate target code");

  SceneDataDescriptor descriptor(*target);
  require(descriptor.m_namesById.size() == target->get_string_constant_count(),
      "descriptor changed string-table size");
  for (std::size_t id = 0; id < descriptor.m_namesById.size(); ++id) {
    const char *expected = target->get_string_constant(id);
    require(descriptor.m_namesById[id] == (expected ? expected : ""),
        "descriptor changed a string ID");
  }
  require(descriptor.find(name).has_value(), "scene-data name missing");
  return descriptor;
}

} // namespace

int main()
{
  try {
    testValueSemantics();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  std::unique_ptr<Core> core;
  try {
    core = std::make_unique<Core>();
  } catch (const std::exception &e) {
    std::printf("SKIP MDL SDK unavailable: %s\n", e.what());
    return 77;
  }

  try {
    using mi::base::make_handle;
    const std::string firstName = "temperature_name_longer_than_a_small_string";
    const std::string secondName =
        "replacement_name_longer_than_a_small_string";
    SceneDataDescriptor first;
    SceneDataDescriptor second;
    {
      auto scope = make_handle(core->createScope("MdlSceneDataDescriptorTest"));
      auto transaction = make_handle(core->createTransaction(scope.get()));
      first = extract(*core, transaction.get(), firstName);
      second = extract(*core, transaction.get(), secondName);
      transaction->commit();
      core->removeScope(scope.get());
    }
    const auto firstId = first.find(firstName);
    const auto secondId = second.find(secondName);
    core.reset();

    // No SDK objects or target-code handles remain alive here.
    require(
        first.find(firstName) == firstId, "first descriptor lost its names");
    require(second.find(secondName) == secondId,
        "second descriptor lost its names");
    require(!first.find(secondName), "source replacement polluted first table");
    require(!second.find(firstName), "source replacement retained old names");
    first = second;
    second.m_namesById.clear();
    require(
        first.find(secondName) == secondId, "descriptor copy lost its names");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  std::puts("MDL scene-data descriptor checks passed");
  return 0;
}

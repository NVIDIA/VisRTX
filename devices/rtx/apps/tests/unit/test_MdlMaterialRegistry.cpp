// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

#include "../MdlModuleReuseFixture.h"
#include "libmdl/source_name_utils.h"
#include "mdl/MaterialRegistry.h"

#include <mi/neuraylib/imdle_api.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/istructure.h>
#include <nonstd/scope.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;
using mi::base::make_handle;
using visrtx::TaskExecutor;
using visrtx::libmdl::Core;
using visrtx::libmdl::Uuid;
using visrtx::mdl::MaterialRegistry;

namespace {

void check(bool condition, const char *message)
{
  if (!condition)
    throw std::runtime_error(message);
}

std::string source(int value)
{
  return "mdl 1.6; import ::df::*; export material test() = material("
         "surface: material_surface(emission: material_emission("
         "emission: df::diffuse_edf(), intensity: color("
      + std::to_string(value) + ".0))));";
}

void writeFile(const fs::path &path, const std::string &contents)
{
  fs::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  file << contents;
  check(bool(file), "cannot write test fixture");
}

void testSnapshot(MaterialRegistry &registry, TaskExecutor &coordinator)
{
  auto acquire = [&](int value) {
    auto result =
        registry.acquireMaterialAsync(coordinator, source(value), "test", true)
            .get();
    check(result.uuid != Uuid{}, "inline compile failed");
    return result;
  };
  auto release = [&](const auto &material) {
    coordinator.invokeSerial([&] { registry.releaseMaterial(material.uuid); });
  };
  auto snapshot = [&] {
    return coordinator.invokeSerial([&] { return registry.ptxSnapshot(); });
  };
  auto index = [&](const auto &material) {
    return coordinator.invokeSerial(
        [&] { return registry.getMaterialImplementationIndex(material.uuid); });
  };

  auto first = acquire(2);
  auto second = acquire(3);
  const auto firstIndex = index(first);
  const auto secondIndex = index(second);
  check(firstIndex != secondIndex, "test materials unexpectedly deduplicated");
  auto captured = snapshot();
  const auto savedBytes = captured.blobs;
  check(!savedBytes.at(firstIndex).empty(), "missing first PTX blob");
  check(!savedBytes.at(secondIndex).empty(), "missing second PTX blob");

  // Deterministically mutate between snapshot and consumption, not a timing
  // dependent release/render race. Empty slots must keep their SBT positions.
  release(first);
  auto hole = snapshot();
  check(hole.timestamp > captured.timestamp, "release did not advance version");
  check(hole.blobs.size() == savedBytes.size(), "snapshot compacted slots");
  check(hole.blobs.at(firstIndex).empty(), "released slot was not empty");
  check(hole.blobs.at(secondIndex) == savedBytes.at(secondIndex),
      "release changed another slot");

  auto replacement = acquire(4);
  check(index(replacement) == firstIndex, "empty slot was not reused");
  auto current = snapshot();
  check(current.timestamp > hole.timestamp, "insert did not advance version");
  check(captured.blobs == savedBytes, "snapshot changed after slot reuse");
  check(hole.blobs.at(firstIndex).empty(), "old empty snapshot changed");
  check(current.blobs.at(firstIndex) != savedBytes.at(firstIndex),
      "replacement should have distinct PTX");

  // Keep reading an old snapshot while the coordinator clears the live slots.
  auto clear = coordinator.submit([&registry,
                                      &coordinator,
                                      secondUuid = second.uuid,
                                      replacementUuid = replacement.uuid] {
    coordinator.invokeSerial([&] {
      registry.releaseMaterial(secondUuid);
      registry.releaseMaterial(replacementUuid);
    });
  });
  check(captured.blobs == savedBytes, "snapshot changed during release");
  clear.get();
  check(captured.blobs == savedBytes, "snapshot lost storage on release");
  check(coordinator.invokeSerial([&] {
    return registry.numRegisteredMaterials();
  }) == 0,
      "snapshot test leaked registry entries");
}

void testModuleReuseFixture(
    MaterialRegistry &registry, TaskExecutor &coordinator)
{
  auto first =
      registry
          .acquireMaterialAsync(coordinator, moduleReuseSource(0), "test", true)
          .get();
  auto second =
      registry
          .acquireMaterialAsync(coordinator, moduleReuseSource(1), "test", true)
          .get();
  check(first.uuid != Uuid{} && second.uuid != Uuid{},
      "reuse fixture failed to compile");
  check(first.uuid != second.uuid, "reuse fixture needs distinct UUIDs");
  coordinator.invokeSerial([&] {
    const auto a = registry.getMaterialImplementationIndex(first.uuid);
    const auto b = registry.getMaterialImplementationIndex(second.uuid);
    const auto snapshot = registry.ptxSnapshot();
    check(a != b, "reuse fixture needs distinct registry slots");
    check(!snapshot.blobs.at(a).empty(), "reuse fixture PTX is empty");
    check(snapshot.blobs.at(a) == snapshot.blobs.at(b),
        "reuse fixture needs identical PTX");
    registry.releaseMaterial(first.uuid);
    registry.releaseMaterial(second.uuid);
  });
}

void exportMdle(Core &core, const fs::path &path)
{
  auto transaction = make_handle(core.createTransaction());
  auto abort = nonstd::make_scope_exit([&] { transaction->abort(); });
  const auto code = source(9);
  auto module = make_handle(core.loadModuleFromString(
      visrtx::libmdl::makeInlineModuleName(code), code, transaction.get()));
  check(bool(module), "MDLE prototype module failed to load");
  auto definition = make_handle(
      core.getFunctionDefinition(module.get(), "test", transaction.get()));
  check(bool(definition), "MDLE prototype definition missing");
  auto data = make_handle(transaction->create<mi::IStructure>("Mdle_data"));
  check(bool(data), "could not create MDLE export data");
  auto prototype = make_handle(data->get_value<mi::IString>("prototype_name"));
  prototype->set_c_str(transaction->name_of(definition.get()));
  auto api = make_handle(
      core.getINeuray()->get_api_component<mi::neuraylib::IMdle_api>());
  check(api->export_mdle(
            transaction.get(), path.string().c_str(), data.get(), nullptr)
          == 0,
      "MDLE export failed");
}

void testSources(Core &core,
    MaterialRegistry &registry,
    TaskExecutor &coordinator,
    const fs::path &root)
{
  const auto library = root / "library";
  writeFile(library / "visrtx_test_shared.mdl",
      "mdl 1.6; export float strength() = 6.0;");
  const std::string textured = R"mdl(mdl 1.6;
import ::df::*;
import ::tex::*;
import ::visrtx_test_shared::*;
export material test() = material(surface: material_surface(
    emission: material_emission(emission: df::diffuse_edf(),
        intensity: tex::lookup_color(texture_2d("./pixel.ppm"), float2(0.0))
            * visrtx_test_shared::strength())));
)mdl";
  writeFile(library / "named.mdl", textured);
  writeFile(library / "other.mdl", textured);
  writeFile(library / "pixel.ppm", "P6\n1 1\n255\n\xff\x80\x40");
  writeFile(library / "recovered.mdl", source(7));
  // An invalid scene copy forces Core's canonical-name recovery. The worker
  // must use the successful preload's DB name, not retry the original path.
  const auto brokenCopy = root / "scene" / "library" / "recovered.mdl";
  writeFile(brokenCopy, "this is not an MDL module");
  std::vector<fs::path> paths{library, root / "scene"};
  core.setMdlSearchPaths(paths);

  std::vector<std::future<MaterialRegistry::AcquiredMaterial>> pending;
  pending.push_back(
      registry.acquireMaterialAsync(coordinator, "::named", "test", false));
  pending.push_back(registry.acquireMaterialAsync(
      coordinator, (library / "other.mdl").string(), "test", false));
  pending.push_back(registry.acquireMaterialAsync(
      coordinator, brokenCopy.string(), "test", false));
  pending.push_back(
      registry.acquireMaterialAsync(coordinator, source(8), "test", true));
  std::vector<MaterialRegistry::AcquiredMaterial> acquired;
  for (auto &job : pending) {
    acquired.push_back(job.get());
    check(acquired.back().uuid != Uuid{},
        "source acquire failed (no fallback permitted)");
  }
  for (std::size_t i = 0; i < 2; ++i) {
    const auto &textures =
        acquired[i].arguments.m_defaultAndBodyTextureDescriptors;
    check(!textures.empty(), "relative texture descriptor missing");
    check(fs::equivalent(textures.front().url, library / "pixel.ppm"),
        "relative texture resolved against the wrong module");
  }

  const auto mdlePath = root / "exported.mdle";
  coordinator.invokeSerial([&] { exportMdle(core, mdlePath); });
  auto mdle =
      registry
          .acquireMaterialAsync(coordinator, mdlePath.string(), "main", false)
          .get();
  check(mdle.uuid != Uuid{}, "MDLE acquire failed");
  coordinator.invokeSerial([&] {
    for (const auto &material : acquired)
      registry.releaseMaterial(material.uuid);
    registry.releaseMaterial(mdle.uuid);
  });
  check(coordinator.invokeSerial([&] {
    return registry.numRegisteredMaterials();
  }) == 0,
      "source test leaked registry entries");
}

} // namespace

int main()
{
  const auto root = fs::temp_directory_path()
      / ("visrtx_mdl_registry_"
          + std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count()));
  auto cleanup = nonstd::make_scope_exit([&] {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  });
  try {
    Core core;
    MaterialRegistry registry(&core);
    TaskExecutor coordinator(4);
    testSnapshot(registry, coordinator);
    testModuleReuseFixture(registry, coordinator);
    testSources(core, registry, coordinator, root);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  std::puts("MDL registry snapshot and source handoff passed");
  return 0;
}

// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <mi/neuraylib/imdle_api.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/istructure.h>
#include <nonstd/scope.hpp>
#include "libmdl/Core.h"
#include "libmdl/source_name_utils.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::atomic<unsigned> errors{0};

void statusFunc(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    std::fprintf(stderr, "FATAL: %s\n", message);
    std::exit(1);
  }
  if (severity == ANARI_SEVERITY_ERROR) {
    ++errors; // One intentionally broken source must exercise fallback.
    std::fprintf(stderr, "ERROR: %s\n", message);
  }
}

std::string source(int value)
{
  return "mdl 1.6; import ::df::*; export material main() = material("
         "surface: material_surface(emission: material_emission("
         "emission: df::diffuse_edf(), intensity: color("
      + std::to_string(value) + ".0))));";
}

void exportMdle(const std::filesystem::path &path)
{
  // Export before creating the device: only one SDK instance can be owned at
  // once. This Core is destroyed before the device creates its own instance.
  using mi::base::make_handle;
  visrtx::libmdl::Core core;
  auto transaction = make_handle(core.createTransaction());
  auto abort = nonstd::make_scope_exit([&] { transaction->abort(); });
  const auto code = source(1000);
  auto module = make_handle(core.loadModuleFromString(
      visrtx::libmdl::makeInlineModuleName(code), code, transaction.get()));
  if (!module)
    throw std::runtime_error("MDLE prototype module failed");
  auto definition = make_handle(
      core.getFunctionDefinition(module.get(), "main", transaction.get()));
  if (!definition)
    throw std::runtime_error("MDLE prototype definition failed");
  auto data = make_handle(transaction->create<mi::IStructure>("Mdle_data"));
  auto prototype = make_handle(data->get_value<mi::IString>("prototype_name"));
  prototype->set_c_str(transaction->name_of(definition.get()));
  auto api = make_handle(
      core.getINeuray()->get_api_component<mi::neuraylib::IMdle_api>());
  if (api->export_mdle(
          transaction.get(), path.string().c_str(), data.get(), nullptr)
      != 0)
    throw std::runtime_error("MDLE export failed");
}

} // namespace

int main()
{
  // CTest pins VISRTX_MDL_COMPILE_THREADS=1. No render is needed: property
  // queries flush commit/finalize and exercise the actual MDL source paths.
  const auto path = std::filesystem::temp_directory_path()
      / ("visrtx_serial_"
          + std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count())
          + ".mdle");
  auto cleanup = nonstd::make_scope_exit([&] {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  });
  try {
    exportMdle(path);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  auto device = makeVisRTXDevice(statusFunc);
  std::vector<anari::Material> materials;
  auto add = [&](const char *type, const std::string &code) {
    auto material = anari::newObject<anari::Material>(device, "mdl");
    anari::setParameter(device, material, "sourceType", type);
    anari::setParameter(device, material, "source", code);
    anari::setParameter(device, material, "materialName", "main");
    anari::commitParameters(device, material);
    materials.push_back(material);
  };

  // MDLE and invalid source finalize alongside prefetched, cold-cache code.
  // Neither MDLE nor the default fallback may compile on the coordinator.
  add("mdle", path.string());
  add("code", "mdl 1.6; export material main() = invalid;");
  constexpr uint32_t validMaterials = 24;
  for (uint32_t i = 0; i < validMaterials; ++i)
    add("code", source(i + 1));

  uint32_t count = 0, peak = 0;
  const bool haveCount = anari::getProperty(
      device, device, "numRegisteredMdlMaterials", count, ANARI_WAIT);
  const bool havePeak = anari::getProperty(
      device, device, "maxConcurrentMdlCompiles", peak, ANARI_WAIT);
  // 24 distinct code materials + MDLE + one default fallback. Count catches
  // accidental fallback of the MDLE or any supposedly valid source.
  const bool ok = haveCount && havePeak && count == validMaterials + 2
      && peak == 1 && errors.load() > 0;
  std::printf("count=%u expected=%u peakConcurrentCompiles=%u errors=%u\n",
      count,
      validMaterials + 2,
      peak,
      errors.load());
  for (auto material : materials)
    anari::release(device, material);
  anari::release(device, device);
  if (!ok)
    std::fprintf(stderr,
        "FAIL: mixed batch did not compile successfully and serially\n");
  return ok ? 0 : 1;
}

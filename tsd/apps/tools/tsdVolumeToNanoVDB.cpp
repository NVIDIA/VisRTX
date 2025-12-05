// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <optional>
#include <string>
#include <string_view>
#include <tsd/core/Logging.hpp>
#include <tsd/core/scene/Scene.hpp>
#include <tsd/io/importers.hpp>
#include <tsd/io/serialization.hpp>

static void printUsage(std::string_view progName)
{
  fmt::print("usage: {} [options] <input_volume> <output.vdb>\n", progName);
  fmt::print("\n");
  fmt::print("Options:\n");
  fmt::print("  --help, -h              Show this help message\n");
  fmt::print(
      "  --undefined <value>     Skip voxels with this undefined value\n");
  fmt::print("  -u <value>              Short form of --undefined\n");
  fmt::print(
      "  --precision <type>      Quantization precision (fp4|fp8|fp16|fpn|half|float32)\n");
  fmt::print("                          Default: fp16\n");
  fmt::print("  -p <type>               Short form of --precision\n");
  fmt::print("  --dither                Enable dithering for quantization\n");
  fmt::print("  -d                      Short form of --dither\n");
  fmt::print("\n");
  fmt::print("Examples:\n");
  fmt::print("  {} input.raw output.vdb\n", progName);
  fmt::print("  {} --undefined 0.0 input.vti output.vdb\n", progName);
  fmt::print("  {} --undefined 0.5 input.mhd output.vdb\n", progName);
  fmt::print("  {} --precision fp8 --dither input.mhd output.vdb\n", progName);
  fmt::print("\n");
  fmt::print("Supported input formats: .raw, .vti, .vtu, .mhd, .hdf5, .nvdb\n");
}

int main(int argc, const char *argv[])
{
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  // Check for help flag
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    }
  }

  std::optional<float> undefinedValue;
  std::optional<std::string> inputFile;
  std::optional<std::string> outputFile;
  tsd::io::VDBPrecision precision = tsd::io::VDBPrecision::Fp16;
  bool enableDithering = false;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};

    if (arg == "--undefined" || arg == "-u") {
      if (i + 1 >= argc) {
        tsd::core::logError("Option {} requires a value", arg);
        printUsage(argv[0]);
        return 1;
      }
      try {
        undefinedValue = std::stof(std::string(argv[++i]));
      } catch (const std::exception &e) {
        tsd::core::logError("Invalid undefined value: {}", argv[i]);
        printUsage(argv[0]);
        return 1;
      }
    } else if (arg == "--precision" || arg == "-p") {
      if (i + 1 >= argc) {
        tsd::core::logError("Option --precision requires a value");
        printUsage(argv[0]);
        return 1;
      }
      std::string_view precStr = argv[++i];
      if (precStr == "fp4") {
        precision = tsd::io::VDBPrecision::Fp4;
      } else if (precStr == "fp8") {
        precision = tsd::io::VDBPrecision::Fp8;
      } else if (precStr == "fp16") {
        precision = tsd::io::VDBPrecision::Fp16;
      } else if (precStr == "fpn") {
        precision = tsd::io::VDBPrecision::FpN;
      } else if (precStr == "half") {
        precision = tsd::io::VDBPrecision::Half;
      } else if (precStr == "float32") {
        precision = tsd::io::VDBPrecision::Float32;
      } else {
        tsd::core::logError("Unknown precision type: %s", precStr);
        printUsage(argv[0]);
        return 1;
      }
    } else if (arg == "--dither" || arg == "-d") {
      enableDithering = true;
    } else if (argv[i][0] != '-') {
      if (!inputFile) {
        inputFile = std::string(arg);
      } else if (!outputFile) {
        outputFile = std::string(arg);
      } else {
        tsd::core::logError("Unexpected positional argument: {}", arg);
        printUsage(argv[0]);
        return 1;
      }
    } else {
      tsd::core::logError("Unknown option: {}", arg);
      printUsage(argv[0]);
      return 1;
    }
  }

  if (!inputFile || !outputFile) {
    tsd::core::logError("Missing required arguments");
    printUsage(argv[0]);
    return 1;
  }

  tsd::core::setLogToStdout();

  tsd::core::logStatus("Loading volume from: {}", *inputFile);

  tsd::core::Scene scene;
  auto volume = tsd::io::import_volume(scene, inputFile->c_str());

  if (!volume) {
    tsd::core::logError("Failed to load volume");
    return 1;
  }

  const auto *spatialField =
      volume->parameterValueAsObject<tsd::core::SpatialField>("value");

  if (!spatialField) {
    tsd::core::logError("Volume does not have a spatial field");
    return 1;
  }

  tsd::io::export_StructuredRegularVolumeToVDB(spatialField,
      outputFile->c_str(),
      undefinedValue.has_value(),
      undefinedValue.value_or(0.0f),
      precision,
      enableDithering);

  tsd::core::logStatus("Export complete");
  return 0;
}

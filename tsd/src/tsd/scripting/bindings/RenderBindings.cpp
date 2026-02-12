// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <sol/sol.hpp>
#include "tsd/core/scene/Scene.hpp"
#include "tsd/rendering/index/RenderIndexAllLayers.hpp"
#include "tsd/rendering/pipeline/RenderPipeline.h"
#include "tsd/scripting/LuaBindings.hpp"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "stb_image_write.h"

namespace tsd::scripting {

struct LuaAnariDevice
{
  anari::Library library{nullptr};
  anari::Device device{nullptr};
  std::string libraryName;

  ~LuaAnariDevice()
  {
    if (device)
      anari::release(device, device);
    if (library)
      anari::unloadLibrary(library);
  }
};

struct LuaCameraSetup
{
  math::float3 position{0.f, 0.f, 5.f};
  math::float3 direction{0.f, 0.f, -1.f};
  math::float3 up{0.f, 1.f, 0.f};
  float fovy{40.f};
  float aspect{1.777f}; // 16:9
};

void registerRenderBindings(sol::state &lua)
{
  sol::table tsd = lua["tsd"];
  sol::table render = tsd["render"];

  tsd.new_usertype<LuaCameraSetup>("CameraSetup",
      sol::constructors<LuaCameraSetup()>(),
      "position",
      &LuaCameraSetup::position,
      "direction",
      &LuaCameraSetup::direction,
      "up",
      &LuaCameraSetup::up,
      "fovy",
      &LuaCameraSetup::fovy,
      "aspect",
      &LuaCameraSetup::aspect);

  tsd.new_usertype<LuaAnariDevice>("AnariDevice",
      sol::no_constructor,
      "libraryName",
      sol::readonly(&LuaAnariDevice::libraryName));

  render["loadDevice"] =
      [](const std::string &libraryName) -> std::shared_ptr<LuaAnariDevice> {
    auto statusFunc = [](const void *,
                          ANARIDevice,
                          ANARIObject,
                          ANARIDataType,
                          ANARIStatusSeverity severity,
                          ANARIStatusCode,
                          const char *message) {
      if (severity == ANARI_SEVERITY_FATAL_ERROR) {
        fprintf(stderr, "[ANARI][FATAL] %s\n", message);
      } else if (severity == ANARI_SEVERITY_ERROR) {
        fprintf(stderr, "[ANARI][ERROR] %s\n", message);
      }
    };

    auto dev = std::make_shared<LuaAnariDevice>();
    dev->libraryName = libraryName;
    dev->library = anari::loadLibrary(libraryName.c_str(), statusFunc);
    if (!dev->library) {
      throw std::runtime_error(
          fmt::format("Failed to load ANARI library: {}", libraryName));
    }
    dev->device = anari::newDevice(dev->library, "default");
    if (!dev->device) {
      anari::unloadLibrary(dev->library);
      dev->library = nullptr;
      throw std::runtime_error(fmt::format(
          "Failed to create ANARI device from library: {}", libraryName));
    }
    return dev;
  };

  tsd.new_usertype<rendering::RenderIndexAllLayers>(
      "RenderIndex",
      sol::constructors<rendering::RenderIndexAllLayers(
          core::Scene &, anari::Device)>(),
      "populate",
      [](rendering::RenderIndexAllLayers &ri) { ri.populate(); },
      "world",
      &rendering::RenderIndexAllLayers::world,
      "device",
      &rendering::RenderIndexAllLayers::device);

  render["createRenderIndex"] = [](core::Scene &scene,
                                    std::shared_ptr<LuaAnariDevice> dev)
      -> std::shared_ptr<rendering::RenderIndexAllLayers> {
    if (!dev || !dev->device) {
      throw std::runtime_error("createRenderIndex: device handle is null");
    }
    return std::make_shared<rendering::RenderIndexAllLayers>(
        scene, dev->device);
  };

  render["getWorldBounds"] =
      [](std::shared_ptr<LuaAnariDevice> dev,
          std::shared_ptr<rendering::RenderIndexAllLayers> index,
          sol::this_state s) -> sol::table {
    if (!dev || !dev->device) {
      throw std::runtime_error("getWorldBounds: device handle is null");
    }
    if (!index) {
      throw std::runtime_error("getWorldBounds: render index handle is null");
    }

    math::float3 bounds[2] = {{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};
    anariGetProperty(dev->device,
        index->world(),
        "bounds",
        ANARI_FLOAT32_BOX3,
        &bounds[0],
        sizeof(bounds),
        ANARI_WAIT);

    sol::state_view lua(s);
    sol::table result = lua.create_table();
    result["min"] = bounds[0];
    result["max"] = bounds[1];
    return result;
  };

  tsd.new_usertype<rendering::RenderPipeline>("RenderPipeline",
      sol::constructors<rendering::RenderPipeline(),
          rendering::RenderPipeline(int, int)>(),
      "setDimensions",
      &rendering::RenderPipeline::setDimensions,
      "render",
      &rendering::RenderPipeline::render,
      "size",
      &rendering::RenderPipeline::size,
      "empty",
      &rendering::RenderPipeline::empty,
      "clear",
      &rendering::RenderPipeline::clear);

  render["createPipeline"] =
      [](int width,
          int height,
          std::shared_ptr<LuaAnariDevice> dev,
          std::shared_ptr<rendering::RenderIndexAllLayers> index,
          const LuaCameraSetup &camera,
          sol::optional<sol::table> rendererParams)
      -> std::shared_ptr<rendering::RenderPipeline> {
    if (width <= 0 || height <= 0) {
      throw std::runtime_error("createPipeline: width and height must be > 0");
    }
    if (!dev || !dev->device) {
      throw std::runtime_error("createPipeline: device handle is null");
    }
    if (!index) {
      throw std::runtime_error("createPipeline: render index handle is null");
    }

    auto pipeline = std::make_shared<rendering::RenderPipeline>(width, height);

    auto cam = anari::newObject<anari::Camera>(dev->device, "perspective");
    anari::setParameter(dev->device, cam, "aspect", camera.aspect);
    anari::setParameter(dev->device, cam, "fovy", math::radians(camera.fovy));
    anari::setParameter(dev->device, cam, "position", camera.position);
    anari::setParameter(dev->device, cam, "direction", camera.direction);
    anari::setParameter(dev->device, cam, "up", camera.up);
    anari::commitParameters(dev->device, cam);

    auto renderer = anari::newObject<anari::Renderer>(dev->device, "default");
    if (rendererParams) {
      for (const auto &kv : *rendererParams) {
        std::string key = kv.first.as<std::string>();
        sol::object val = kv.second;
        if (val.is<bool>())
          anari::setParameter(
              dev->device, renderer, key.c_str(), val.as<bool>());
        else if (val.is<std::string>())
          anari::setParameter(
              dev->device, renderer, key.c_str(), val.as<std::string>());
        else if (val.is<int>())
          anari::setParameter(
              dev->device, renderer, key.c_str(), val.as<int>());
        else if (val.is<float>() || val.is<double>())
          anari::setParameter(
              dev->device, renderer, key.c_str(), val.as<float>());
      }
    }
    anari::commitParameters(dev->device, renderer);

    auto *pass =
        pipeline->emplace_back<rendering::AnariSceneRenderPass>(dev->device);
    pass->setWorld(index->world());
    pass->setRenderer(renderer);
    pass->setCamera(cam);
    pass->setRunAsync(false);

    anari::release(dev->device, cam);
    anari::release(dev->device, renderer);

    return pipeline;
  };

  render["renderToFile"] = [](std::shared_ptr<rendering::RenderPipeline>
                                   pipeline,
                               int samples,
                               const std::string &filename,
                               int width,
                               int height) {
    if (!pipeline) {
      throw std::runtime_error("renderToFile: pipeline handle is null");
    }
    if (samples < 1) {
      throw std::runtime_error("renderToFile: samples must be >= 1");
    }
    if (width <= 0 || height <= 0) {
      throw std::runtime_error("renderToFile: width and height must be > 0");
    }

    // Sync buffer dimensions with file dimensions so indexing is safe
    pipeline->setDimensions(
        static_cast<uint32_t>(width), static_cast<uint32_t>(height));

    for (int i = 0; i < samples; i++) {
      pipeline->render();
    }

    const uint32_t *pixels = pipeline->getColorBuffer();
    if (!pixels) {
      throw std::runtime_error("No color buffer available");
    }

    size_t dotPos = filename.find_last_of('.');
    if (dotPos == std::string::npos) {
      throw std::runtime_error(
          "Cannot determine image format: no file extension");
    }
    std::string ext = filename.substr(dotPos + 1);

    for (char &c : ext) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    // OpenGL convention -> image convention
    std::vector<uint32_t> flipped(width * height);
    for (int y = 0; y < height; y++) {
      std::memcpy(&flipped[y * width],
          &pixels[(height - 1 - y) * width],
          width * sizeof(uint32_t));
    }

    int result = 0;
    if (ext == "png") {
      result = stbi_write_png(
          filename.c_str(), width, height, 4, flipped.data(), width * 4);
    } else if (ext == "jpg" || ext == "jpeg") {
      // JPEG requires RGB (no alpha)
      std::vector<uint8_t> rgb(width * height * 3);
      for (int i = 0; i < width * height; i++) {
        uint32_t pixel = flipped[i];
        rgb[i * 3 + 0] = (pixel >> 0) & 0xFF;
        rgb[i * 3 + 1] = (pixel >> 8) & 0xFF;
        rgb[i * 3 + 2] = (pixel >> 16) & 0xFF;
      }
      result =
          stbi_write_jpg(filename.c_str(), width, height, 3, rgb.data(), 95);
    } else if (ext == "bmp") {
      result =
          stbi_write_bmp(filename.c_str(), width, height, 4, flipped.data());
    } else if (ext == "tga") {
      result =
          stbi_write_tga(filename.c_str(), width, height, 4, flipped.data());
    } else if (ext == "ppm") {
      FILE *fp = fopen(filename.c_str(), "wb");
      if (!fp) {
        throw std::runtime_error(
            fmt::format("Failed to open file: {}", filename));
      }
      fprintf(fp, "P6\n%d %d\n255\n", width, height);
      bool writeOk = true;
      for (int i = 0; i < width * height && writeOk; i++) {
        uint32_t pixel = flipped[i];
        unsigned char rgb[3];
        rgb[0] = (pixel >> 0) & 0xFF;
        rgb[1] = (pixel >> 8) & 0xFF;
        rgb[2] = (pixel >> 16) & 0xFF;
        if (fwrite(rgb, 1, 3, fp) != 3)
          writeOk = false;
      }
      if (fclose(fp) != 0)
        writeOk = false;
      result = writeOk ? 1 : 0;
    } else {
      throw std::runtime_error(fmt::format(
          "Unsupported image format '{}'. Supported: png, jpg/jpeg, bmp, tga, ppm",
          ext));
    }

    if (result == 0) {
      throw std::runtime_error(
          fmt::format("Failed to write image '{}': {} (errno={})",
              filename,
              std::strerror(errno),
              errno));
    }
  };
}

} // namespace tsd::scripting

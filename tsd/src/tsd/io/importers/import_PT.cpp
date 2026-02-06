/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"

#if TSD_USE_TORCH
#include <immintrin.h> // For _cvtss_sh
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
// Torch
#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>

namespace {
const float DEFAULT_ROUGHNESS = 0.5f;
const float DEFAULT_METALLIC = 0.5f;
} // namespace

namespace tsd::io {

using LayerArrays = std::vector<std::vector<float>>;

struct ModelData
{
  torch::jit::script::Module module;
  LayerArrays layerArrays;
  int n_layers{0};
  int hidden_dim{0};
  size_t total_parameters{0};
  float3 aabb_min;
  float3 aabb_max;
  float threshold{0.1f};
};

ModelData loadModel(const char *filename)
{
  ModelData data;
  data.module = torch::jit::load(filename);
  auto state_dict = data.module.named_parameters();

  // Log model attributes
  logInfo("=== Model Attributes ===");
  auto bounds_min = data.module.attr("mesh_bounds_min").toTensor();
  auto bounds_max = data.module.attr("mesh_bounds_max").toTensor();
  data.threshold = data.module.attr("threshold").toTensor().item<float>();

  auto min_accessor = bounds_min.accessor<float, 1>();
  auto max_accessor = bounds_max.accessor<float, 1>();

  data.aabb_min = float3(min_accessor[0], min_accessor[1], min_accessor[2]);
  data.aabb_max = float3(max_accessor[0], max_accessor[1], max_accessor[2]);

  std::string min_values = "[" + std::to_string(data.aabb_min.x) + ", "
      + std::to_string(data.aabb_min.y) + ", " + std::to_string(data.aabb_min.z)
      + "]";
  std::string max_values = "[" + std::to_string(data.aabb_max.x) + ", "
      + std::to_string(data.aabb_max.y) + ", " + std::to_string(data.aabb_max.z)
      + "]";

  logInfo(("  mesh_bounds_min: " + min_values).c_str());
  logInfo(("  mesh_bounds_max: " + max_values).c_str());
  logInfo(("  threshold      : " + std::to_string(data.threshold)).c_str());
  logInfo("======================");

  // Process the parameters
  for (const auto &pair : state_dict) {
    auto tensor = pair.value;
    std::vector<float> array(
        tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    data.layerArrays.push_back(array);
    logInfo(("  " + pair.name + " : " + std::to_string(array.size())).c_str());

    // Count parameters and analyze layer structure
    if (pair.name.find("weight") != std::string::npos) {
      data.n_layers++;
      data.total_parameters += tensor.numel();

      if (data.hidden_dim == 0 && tensor.size(1) != 1) { // Not the output layer
        data.hidden_dim = tensor.size(1);
      }
    }
  }

  const uint32_t NEURAL_NB_MAX_LAYERS = 5; // TODO: change this
  if (data.n_layers > NEURAL_NB_MAX_LAYERS) {
    throw std::runtime_error(
        "Maximum number of layers is " + std::to_string(NEURAL_NB_MAX_LAYERS));
  }

  // Log network architecture summary
  logInfo("=== Network Architecture Summary ===");
  logInfo(("Total layers: " + std::to_string(data.n_layers)).c_str());
  logInfo(("Hidden dimension: " + std::to_string(data.hidden_dim)).c_str());
  logInfo(
      ("Total parameters: " + std::to_string(data.total_parameters)).c_str());
  logInfo(("Memory usage: "
      + std::to_string(data.total_parameters * sizeof(float) / 1024.0f) + " KB")
          .c_str());
  logInfo("=================================");

  return data;
}

void import_PT(Scene &scene, const char *filename, LayerNodeRef location)
{
  try {
    ModelData data = loadModel(filename);

    auto material =
        scene.createObject<Material>(tokens::material::physicallyBased);
    float3 baseColor(1.f, 1.f, 1.f);
    const float metallic = DEFAULT_METALLIC;
    const float roughness = DEFAULT_ROUGHNESS;
    material->setParameter("baseColor", ANARI_FLOAT32_VEC3, &baseColor);
    material->setParameter("metallic", ANARI_FLOAT32, &metallic);
    material->setParameter("roughness", ANARI_FLOAT32, &roughness);

    const auto neuralLocation = scene.defaultLayer()->insert_first_child(
        location, tsd::core::Any(ANARI_GEOMETRY, 1));

    const std::string basename =
        std::filesystem::path(filename).stem().string();

    // Create transform as parent of neural object
    const auto xformNode = scene.insertChildTransformNode(
        neuralLocation, tsd::math::mat4(tsd::math::identity), basename.c_str());

    auto neural = scene.createObject<Geometry>(tokens::geometry::neural);
    const std::string name = "neural_geometry_t";
    neural->setName(name.c_str());

    // Add number of layers and threshold parameters
    neural->setParameter("n_layers", ANARI_UINT32, &data.n_layers);
    neural->setParameter("threshold", ANARI_FLOAT32, &data.threshold);

    // Add bounds parameters
    neural->setParameter("aabb_min", ANARI_FLOAT32_VEC3, &data.aabb_min);
    neural->setParameter("aabb_max", ANARI_FLOAT32_VEC3, &data.aabb_max);

    // Add each layer's weights and biases as parameters
    int layer_idx = 0;
    for (const auto &array : data.layerArrays) {
      std::string param_name = "layer_" + std::to_string(layer_idx);
      // Convert float32 to float16
      std::vector<uint16_t> dataAsFloat16(array.size());
      for (size_t i = 0; i < array.size(); i++) {
        dataAsFloat16[i] = _cvtss_sh(array[i], 0);
      }
      auto layerArray = scene.createArray(ANARI_UINT16, array.size());
      layerArray->setData(dataAsFloat16.data());
      neural->setParameterObject(param_name.c_str(), *layerArray);
      layer_idx++;
    }

    const auto surface = scene.createSurface(name.c_str(), neural, material);

    // Insert surface as child of transform
    scene.insertChildObjectNode(xformNode, surface);

  } catch (const std::exception &e) {
    logError(("Error: " + std::string(e.what())).c_str());
  }
}
} // namespace tsd
#else
namespace tsd::io {
void import_PT(Scene &scene, const char *filename, LayerNodeRef location)
{
  logError("[import_PT] PyTorch not enabled in TSD build.");
}
} // namespace tsd
#endif

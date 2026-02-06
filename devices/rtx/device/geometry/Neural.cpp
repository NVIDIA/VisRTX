// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Neural.h"

#include <fstream>
#include <iostream>

using namespace std;

namespace visrtx {

Neural::Neural(DeviceGlobalState *d) : Geometry(d) {}

Neural::~Neural() {}

void Neural::commitParameters()
{
  Geometry::commitParameters();

  // Get layers parameters from TSD object
  const auto n_layers = getParam<uint32_t>("n_layers", 0);
  for (uint32_t i = 0; i < 2 * n_layers; i++) {
    const auto layer_name = "layer_" + to_string(i);
    auto layer_data = getParamObject<Array1D>(layer_name.c_str());
    m_layers.push_back(layer_data);
  }
}

void Neural::finalize()
{
  Geometry::finalize();
  m_aabbs.resize(1);
  m_aabbs.dataHost()[0] = m_aabb;
  m_aabbs.upload();
  m_aabbsBufferPtr = (CUdeviceptr)m_aabbs.dataDevice();
}

bool Neural::isValid() const
{
  return Geometry::isValid();
}

void Neural::populateBuildInput(OptixBuildInput &input) const
{
  input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

  input.customPrimitiveArray.aabbBuffers = &m_aabbsBufferPtr;
  input.customPrimitiveArray.numPrimitives = m_aabbs.size();

  static uint32_t buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  input.customPrimitiveArray.flags = buildInputFlags;
  input.customPrimitiveArray.numSbtRecords = 1;
}

GeometryGPUData Neural::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::NEURAL;

  auto &neural = retval.neural;
  neural.nb_layers = m_layers.size() / 2;

  // Read bounds from parameters
  neural.boundMin = getParam<glm::vec3>("aabb_min", glm::vec3(-1.f));
  neural.boundMax = getParam<glm::vec3>("aabb_max", glm::vec3(1.f));

  neural.threshold = getParam<float>("threshold", 0.1f);

  for (uint32_t i = 0; i < neural.nb_layers; i++) {
    neural.weights[i] = m_layers[2 * i]
        ? reinterpret_cast<__half *>(const_cast<uint16_t *>(
              m_layers[2 * i]->beginAs<uint16_t>(AddressSpace::GPU)))
        : nullptr;

    neural.biases[i] = m_layers[2 * i + 1]
        ? reinterpret_cast<__half *>(const_cast<uint16_t *>(
              m_layers[2 * i + 1]->beginAs<uint16_t>(AddressSpace::GPU)))
        : nullptr;
  }

  return retval;
}

int Neural::optixGeometryType() const
{
  return OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
}

} // namespace visrtx
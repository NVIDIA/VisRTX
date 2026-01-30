/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <functional>
#include <map>
#include <string>
#include <mutex>

namespace visrtx {

// Forward declaration
class DeviceGlobalState;
struct SpatialField;

// Factory function type for creating spatial fields
using SpatialFieldFactory = std::function<SpatialField*(DeviceGlobalState*)>;

/**
 * @brief Global registry for spatial field types
 * 
 * Allows external plugins to register custom field types at static 
 * initialization time, enabling runtime extension of supported field types.
 */
class SpatialFieldRegistry {
public:
  static SpatialFieldRegistry& instance() {
    static SpatialFieldRegistry registry;
    return registry;
  }
  
  // Register a new spatial field type
  void registerType(const std::string& type, SpatialFieldFactory factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    factories_[type] = factory;
  }
  
  // Create a spatial field by type (returns nullptr if not found)
  SpatialField* create(DeviceGlobalState* d, const std::string& type) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = factories_.find(type);
    if (it != factories_.end()) {
      return it->second(d);
    }
    return nullptr;
  }
  
  // Check if a type is registered
  bool hasType(const std::string& type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return factories_.find(type) != factories_.end();
  }
  
private:
  SpatialFieldRegistry() = default;
  SpatialFieldRegistry(const SpatialFieldRegistry&) = delete;
  SpatialFieldRegistry& operator=(const SpatialFieldRegistry&) = delete;
  
  std::map<std::string, SpatialFieldFactory> factories_;
  mutable std::mutex mutex_;
};

// Helper macro for registration (use in .cpp files)
// Creates a static object that registers the field type before main() runs
#define VISRTX_REGISTER_SPATIAL_FIELD(TYPE_NAME, CLASS_NAME) \
  namespace { \
    struct CLASS_NAME##Registrar { \
      CLASS_NAME##Registrar() { \
        visrtx::SpatialFieldRegistry::instance().registerType(TYPE_NAME, \
          [](visrtx::DeviceGlobalState* d) -> visrtx::SpatialField* { \
            return new CLASS_NAME(d); \
          }); \
      } \
    }; \
    static CLASS_NAME##Registrar g_##CLASS_NAME##Registrar; \
  }

// Function to register custom analytical fields from external applications
// Called by VolumetricPlanets or other projects to register their field types
void registerAnalyticalField(const std::string& typeName, SpatialFieldFactory factory);

} // namespace visrtx

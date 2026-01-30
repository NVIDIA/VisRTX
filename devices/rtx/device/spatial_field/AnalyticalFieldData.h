// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu/gpu_math.h"
#include <optix.h>

/**
 * @file AnalyticalFieldData.h
 * @brief GPU data structures for custom analytical spatial fields
 * 
 * This file provides a type-agnostic framework for analytical fields.
 * External VisRTX only knows about the generic AnalyticalFieldGPUData struct.
 * Specific field implementations (magnetic, aurora, etc.) are defined in the
 * VolumetricPlanets project and dispatched at runtime based on the type field.
 */

namespace visrtx {

/**
 * @brief Type identifier for analytical field subtypes
 * 
 * This enum is used for runtime dispatch in the analytical field sampler.
 * New field types should be added here when extending the system.
 */
enum class AnalyticalFieldType : uint32_t
{
  Unknown = 0,
  Magnetic,
  IMF,
  Aurora,
  Planet,
  Cloud,
  // Add new field types here
  Count
};

} // namespace visrtx

// Magnetic dipole field parameters
struct MagneticFieldData
{
  float equatorStrength;  // Magnetic field strength at equator (nT)
  float poleStrength;     // Magnetic field strength at poles (nT)
  float dipoleTilt;       // Dipole tilt angle (radians, converted from degrees)
  float invMaxRadius;     // 1/maxRadius for bounding optimization
};

// Interplanetary Magnetic Field (IMF) parameters
struct IMFFieldData
{
  visrtx::vec3 solarWindVelocity;  // Solar wind velocity vector (km/s)
  float solarWindDensity;          // Solar wind density (particles/cm³)
  visrtx::vec3 imfDirection;       // IMF direction (normalized)
  float imfStrength;               // IMF strength (nT)
  float parkSpiralAngle;           // Parker spiral angle (radians)
  float padding[3];                // Alignment padding
};

// Aurora field parameters
struct AuroraFieldData
{
  float minAltitude;          // Minimum altitude (km)
  float maxAltitude;          // Maximum altitude (km)
  float minLatitude;          // Minimum magnetic latitude (radians)
  float maxLatitude;          // Maximum magnetic latitude (radians)
  float intensity;            // Aurora intensity multiplier
  float planetRadius;         // Planet radius for coordinate conversion
  float padding[2];           // Alignment padding
};

// Planet surface field parameters
struct PlanetFieldData
{
  cudaTextureObject_t elevationMap;  // Elevation texture
  cudaTextureObject_t diffuseMap;    // Diffuse color texture
  cudaTextureObject_t normalMap;     // Normal map texture
  float planetRadius;                // Planet radius
  float elevationScale;              // Elevation scale factor
  float atmosphereThickness;         // Atmosphere thickness
  int hasElevation;                  // Boolean: has elevation data
  int hasDiffuse;                    // Boolean: has diffuse data
  int hasNormal;                     // Boolean: has normal data
  float padding[2];                  // Alignment padding
};

// Cloud field parameters
struct CloudFieldData
{
  cudaTextureObject_t cloudData;     // 3D cloud density texture
  visrtx::vec3 cloudDims;            // Cloud volume dimensions
  float planetRadius;                // Planet radius for spherical mapping
  float atmosphereThickness;         // Atmosphere layer thickness
  float minLat;                      // Minimum latitude in degrees
  float maxLat;                      // Maximum latitude in degrees
  float minLon;                      // Minimum longitude in degrees
  float maxLon;                      // Maximum longitude in degrees
  float normalEpsilon;               // Epsilon for gradient computation
  int computeNormals;                // Boolean: compute gradients for lighting
};

/**
 * @brief Unified GPU data structure for all analytical field types
 * 
 * This struct is used by external VisRTX. The type field determines
 * which union member contains valid data, and the PTX sampler uses
 * this for runtime dispatch.
 * 
 * Note: Must be trivially constructible to be used in a union.
 */
struct AnalyticalFieldGPUData
{
  visrtx::AnalyticalFieldType type;
  union
  {
    MagneticFieldData magnetic;
    IMFFieldData imf;
    AuroraFieldData aurora;
    PlanetFieldData planet;
    CloudFieldData cloud;
  };
  
  AnalyticalFieldGPUData() = default;
};

// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/objects/Camera.hpp"
#include "tsd/core/scene/Scene.hpp"
// std
#include <string>
#include <cmath>

namespace tsd::core {

Camera::Camera(Token subtype) : Object(ANARI_CAMERA, subtype)
{
  // Common parameters for all camera types (as per ANARI spec)
  addParameter("position")
      .setValue(float3(0.f, 0.f, 0.f))
      .setDescription("position of the camera in world-space");

  addParameter("direction")
      .setValue(float3(0.f, 0.f, -1.f))
      .setDescription("main viewing direction of the camera");

  addParameter("up")
      .setValue(float3(0.f, 1.f, 0.f))
      .setDescription("up direction of the camera");

  // Image region in normalized screen-space coordinates
  // Using float4 for FLOAT32_BOX2 ((0, 0), (1, 1))
  addParameter("imageRegion")
      .setValue(float4(0.f, 0.f, 1.f, 1.f))
      .setDescription("region of the sensor in normalized screen-space coordinates");

  // Depth of field parameters (KHR_CAMERA_DEPTH_OF_FIELD extension)
  addParameter("apertureRadius")
      .setValue(0.f)
      .setDescription("size of the aperture, controls the depth of field")
      .setMin(0.f);

  addParameter("focusDistance")
      .setValue(1.f)
      .setDescription("distance at where the image is sharpest when depth of field is enabled")
      .setMin(0.001f);

  // Stereo parameters (KHR_CAMERA_STEREO extension)
  addParameter("stereoMode")
      .setValue("none")
      .setDescription("stereo mode: none, left, right, sideBySide, topBottom")
      .setStringValues({"none", "left", "right", "sideBySide", "topBottom"});

  addParameter("interpupillaryDistance")
      .setValue(0.0635f)
      .setDescription("distance between left and right eye when stereo is enabled")
      .setMin(0.f);

  // Shutter parameters (KHR_CAMERA_SHUTTER extension)
  // Using float2 for FLOAT32_BOX1 [0.5, 0.5]
  addParameter("shutter")
      .setValue(float2(0.5f, 0.5f))
      .setDescription("start and end of shutter time, clamped to [0, 1]");

  // Rolling shutter parameters (KHR_CAMERA_ROLLING_SHUTTER extension)
  addParameter("rollingShutterDirection")
      .setValue("none")
      .setDescription("rolling direction of the shutter: none, left, right, down, up")
      .setStringValues({"none", "left", "right", "down", "up"});

  addParameter("rollingShutterDuration")
      .setValue(0.f)
      .setDescription("the 'open' time per line, clamped to [0, shutter.upper-shutter.lower]")
      .setMin(0.f);

  // Camera subtype-specific parameters
  if (subtype == tokens::camera::perspective) {
    // KHR_CAMERA_PERSPECTIVE extension
    addParameter("fovy")
        .setValue(float(M_PI) / 3.0f)
        .setDescription("the field of view (angle in radians) of the frame's height")
        .setMin(0.001f)
        .setMax(float(M_PI) - 0.001f);

    addParameter("aspect")
        .setValue(1.f)
        .setDescription("ratio of width by height of the frame (and image region)");

    addParameter("near")
        .setDescription("near clip plane distance")
        .setMin(0.f);

    addParameter("far")
        .setDescription("far clip plane distance")
        .setMin(0.f);

  } else if (subtype == tokens::camera::orthographic) {
    // KHR_CAMERA_ORTHOGRAPHIC extension
    addParameter("aspect")
        .setValue(1.f)
        .setDescription("ratio of width by height of the frame (and image region)");

    addParameter("height")
        .setValue(1.f)
        .setDescription("height of the image plane in world units")
        .setMin(0.001f);

    addParameter("near")
        .setDescription("near clip plane distance")
        .setMin(0.f);

    addParameter("far")
        .setDescription("far clip plane distance")
        .setMin(0.f);

  } else if (subtype == tokens::camera::omnidirectional) {
    // KHR_CAMERA_OMNIDIRECTIONAL extension
    addParameter("layout")
        .setValue("equirectangular")
        .setDescription("pixel layout: equirectangular")
        .setStringValues({"equirectangular"});
  }
}

IndexedVectorRef<Camera> Camera::self() const
{
  return scene() ? scene()->getObject<Camera>(index())
                 : IndexedVectorRef<Camera>{};
}

anari::Object Camera::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Camera>(d, subtype().c_str());
}

namespace tokens::camera {

const Token perspective = "perspective";
const Token orthographic = "orthographic";
const Token omnidirectional = "omnidirectional";

} // namespace tokens::camera

} // namespace tsd::core

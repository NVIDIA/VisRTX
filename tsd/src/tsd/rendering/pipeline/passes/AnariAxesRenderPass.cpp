// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AnariAxesRenderPass.h"
// std
#include <cstring>
// tsd
#include "tsd/core/Logging.hpp"

namespace tsd::rendering {

AnariAxesRenderPass::AnariAxesRenderPass(
    anari::Device d, const anari::Extensions &e)
    : m_device(d)
{
  anari::retain(d, d);

  if (m_deviceUsable = checkNeededExtensions(e); !m_deviceUsable)
    return;

  m_frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, m_frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, m_frame, "channel.depth", ANARI_FLOAT32);
  anari::setParameter(d, m_frame, "accumulation", true);

  auto renderer = anari::newObject<anari::Renderer>(d, "default");
  anari::setParameter(d, renderer, "ambientRadiance", 1.f);
  anari::setParameter(d, renderer, "background", tsd::math::float4(0.f));
  anari::commitParameters(d, renderer);
  anari::setParameter(d, m_frame, "renderer", renderer);
  anari::release(d, renderer);

  if (e.ANARI_KHR_CAMERA_ORTHOGRAPHIC) {
    m_camera = anari::newObject<anari::Camera>(d, "orthographic");
    anari::setParameter(d, m_camera, "height", 2.5f);
  } else {
    tsd::core::logWarning(
        "[AnariAxesRenderPass] 'orthographic' camera not available,"
        " using 'perspective'");
    m_camera = anari::newObject<anari::Camera>(d, "perspective");
    anari::setParameter(d, m_camera, "fovy", tsd::math::radians(5.f));
  }
  anari::commitParameters(d, m_camera);
  anari::setParameter(d, m_frame, "camera", m_camera);

  setupWorld();

  anari::commitParameters(d, m_frame);
}

AnariAxesRenderPass::~AnariAxesRenderPass()
{
  if (isValid()) {
    anari::discard(m_device, m_frame);
    anari::wait(m_device, m_frame);
    anari::release(m_device, m_camera);
    anari::release(m_device, m_frame);
  }

  anari::release(m_device, m_device);
}

void AnariAxesRenderPass::setView(
    const tsd::math::float3 &dir, const tsd::math::float3 &up)
{
  if (!isValid())
    return;

  anari::setParameter(m_device, m_camera, "direction", dir);
  anari::setParameter(m_device, m_camera, "position", -dir * 30.f);
  anari::setParameter(m_device, m_camera, "up", up);
  anari::commitParameters(m_device, m_camera);
}

bool AnariAxesRenderPass::checkNeededExtensions(const anari::Extensions &e)
{
  if (!e.ANARI_KHR_CAMERA_ORTHOGRAPHIC || !e.ANARI_KHR_CAMERA_PERSPECTIVE) {
    tsd::core::logWarning(
        "[AnariAxesRenderPass] "
        "unable to activate pass -- no camera subtypes available");
    return false;
  }

  if (!e.ANARI_KHR_GEOMETRY_CYLINDER) {
    tsd::core::logWarning(
        "[AnariAxesRenderPass] "
        "unable to activate pass -- 'cylinder' geometry not implemented");
    return false;
  }

  if (!e.ANARI_KHR_MATERIAL_MATTE) {
    tsd::core::logWarning(
        "[AnariAxesRenderPass] "
        "unable to activate pass -- 'matte' material not implemented");
    return false;
  }

  return true;
}

bool AnariAxesRenderPass::isValid() const
{
  return m_deviceUsable;
}

void AnariAxesRenderPass::setupWorld()
{
  auto geometry = anari::newObject<anari::Geometry>(m_device, "cylinder");
  tsd::math::float3 vertex_position[] = {
      tsd::math::float3(-0.f, 0.f, 0.f),
      tsd::math::float3(1.f, 0.f, 0.f),
      tsd::math::float3(0.f, -0.f, 0.f),
      tsd::math::float3(0.f, 1.f, 0.f),
      tsd::math::float3(0.f, 0.f, -0.f),
      tsd::math::float3(0.f, 0.f, 1.f),
  };
  constexpr float minIntensity = 0.2f;
  constexpr float maxIntensity = 1.0f;
  tsd::math::float3 vertex_color[] = {
      tsd::math::float3(minIntensity, 0.f, 0.f),
      tsd::math::float3(maxIntensity, 0.f, 0.f),
      tsd::math::float3(0.f, minIntensity, 0.f),
      tsd::math::float3(0.f, maxIntensity, 0.f),
      tsd::math::float3(0.f, 0.f, minIntensity),
      tsd::math::float3(0.f, 0.f, maxIntensity),
  };
  anari::setParameterArray1D(
      m_device, geometry, "vertex.position", vertex_position, 6);
  anari::setParameterArray1D(
      m_device, geometry, "vertex.color", vertex_color, 6);
  anari::setParameter(m_device, geometry, "radius", 0.05f);
  anari::commitParameters(m_device, geometry);

  auto material = anari::newObject<anari::Material>(m_device, "matte");
  anari::setParameter(m_device, material, "color", "color");
  anari::commitParameters(m_device, material);

  auto surface = anari::newObject<anari::Surface>(m_device);
  anari::setParameter(m_device, surface, "geometry", geometry);
  anari::setParameter(m_device, surface, "material", material);
  anari::commitParameters(m_device, surface);

  auto world = anari::newObject<anari::World>(m_device);
  anari::setParameterArray1D(m_device, world, "surface", &surface, 1);
  anari::commitParameters(m_device, world);

  anari::setParameter(m_device, m_frame, "world", world);

  anari::release(m_device, geometry);
  anari::release(m_device, material);
  anari::release(m_device, surface);
  anari::release(m_device, world);
}

void AnariAxesRenderPass::updateSize()
{
  if (!isValid())
    return;

  auto size = tsd::math::uint2(getDimensions() * 0.1f);
  anari::setParameter(
      m_device, m_frame, "size", tsd::math::uint2(size.x, size.x));
  anari::commitParameters(m_device, m_frame);
}

void AnariAxesRenderPass::render(RenderBuffers &b, int /*stageId*/)
{
  if (!isValid())
    return;

  if (m_firstFrame) {
    anari::render(m_device, m_frame);
    anari::wait(m_device, m_frame);
    m_firstFrame = false;
  }

  if (anari::isReady(m_device, m_frame)) {
    const tsd::math::uint2 fbSize(getDimensions());
    auto pixels = anari::map<uint32_t>(m_device, m_frame, "channel.color");
    for (uint32_t y = 0; y < pixels.height; y++) {
      auto *start = pixels.data + (y * pixels.width);
      auto *end = start + pixels.width;
      auto *dst = b.color + (y * fbSize.x);
      for (auto *c = start; c < end; c++, dst++) {
        if ((*c & 0xFF000000) != 0)
          *dst = *c;
      }
    }
    anari::unmap(m_device, m_frame, "channel.color");
    anari::render(m_device, m_frame);
  }
}

} // namespace tsd::rendering

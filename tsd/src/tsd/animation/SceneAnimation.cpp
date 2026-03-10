// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "SceneAnimation.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <algorithm>
#include <cmath>

namespace tsd::animation {

SceneAnimation::SceneAnimation(tsd::scene::Scene &scene) : m_scene(scene) {}

void SceneAnimation::setTimeChangedCallback(TimeChangedCallback cb)
{
  m_timeChangedCallback = std::move(cb);
}

Animation &SceneAnimation::addAnimation(const std::string &name)
{
  return m_animations.emplace_back(Animation{name, {}, {}});
}

std::vector<Animation> &SceneAnimation::animations()
{
  return m_animations;
}

const std::vector<Animation> &SceneAnimation::animations() const
{
  return m_animations;
}

void SceneAnimation::removeAnimation(size_t index)
{
  if (index < m_animations.size())
    m_animations.erase(m_animations.begin() + index);
}

void SceneAnimation::removeAllAnimations()
{
  m_animations.clear();
}

void SceneAnimation::setAnimationTime(float time)
{
  m_time = time;

  if (!m_animations.empty()) {
    auto result = evaluate(m_animations, time);
    applyResults(result, m_scene);
  }

  if (m_timeChangedCallback)
    m_timeChangedCallback(time);
}

float SceneAnimation::getAnimationTime() const
{
  return m_time;
}

void SceneAnimation::setAnimationIncrement(float increment)
{
  m_incrementSize = increment;
  if (increment > 0.5f) {
    tsd::core::logWarning(
        "[scene] setting animation increment > 0.5 will cause odd"
        " animation behavior.");
  }
}

float SceneAnimation::getAnimationIncrement() const
{
  return m_incrementSize;
}

void SceneAnimation::incrementAnimationTime()
{
  auto newTime = m_time + m_incrementSize;
  if (newTime > 1.f)
    newTime = 0.f;
  setAnimationTime(newTime);
}

int SceneAnimation::getAnimationTotalFrames() const
{
  return m_totalFrames;
}

void SceneAnimation::setAnimationTotalFrames(int frames)
{
  m_totalFrames = std::max(2, frames);
}

float SceneAnimation::getAnimationFPS() const
{
  return m_fps;
}

void SceneAnimation::setAnimationFPS(float fps)
{
  if (fps > 0.f)
    m_fps = fps;
}

int SceneAnimation::getAnimationFrame() const
{
  return static_cast<int>(std::round(m_time * (m_totalFrames - 1)));
}

void SceneAnimation::setAnimationFrame(int frame)
{
  int clamped = std::clamp(frame, 0, m_totalFrames - 1);
  setAnimationTime(static_cast<float>(clamped) / (m_totalFrames - 1));
}

void SceneAnimation::incrementAnimationFrame()
{
  int frame = getAnimationFrame() + 1;
  if (frame >= m_totalFrames)
    frame = 0;
  setAnimationFrame(frame);
}

} // namespace tsd::animation

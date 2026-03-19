// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AnimationManager.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <algorithm>
#include <cmath>

namespace tsd::animation {

AnimationManager::AnimationManager(tsd::scene::Scene *scene) : m_scene(scene)
{
  m_defragToken = m_scene->addDefragCallback([this](const auto &remap) {
    for (auto &anim : m_animations) {
      for (auto &binding : anim.bindings) {
        binding.target.updateDefragmentedIndex(
            remap(binding.target.get()->type(), binding.target.get()->index()));

        // TODO: If this is an object time series, update each time step index
        //       to the new remapped index.
      }
    }
  });
}

AnimationManager::~AnimationManager()
{
  m_scene->removeDefragCallback(m_defragToken);
}

void AnimationManager::setTimeChangedCallback(TimeChangedCallback cb)
{
  m_timeChangedCallback = std::move(cb);
}

Animation &AnimationManager::addAnimation(const std::string &name)
{
  return m_animations.emplace_back(Animation{name, {}, {}});
}

std::vector<Animation> &AnimationManager::animations()
{
  return m_animations;
}

const std::vector<Animation> &AnimationManager::animations() const
{
  return m_animations;
}

void AnimationManager::removeAnimation(size_t index)
{
  if (index < m_animations.size())
    m_animations.erase(m_animations.begin() + index);
}

void AnimationManager::removeAllAnimations()
{
  m_animations.clear();
}

void AnimationManager::setAnimationTime(float time)
{
  m_time = time;

  if (!m_animations.empty()) {
    auto result = evaluate(m_animations, time);
    applyResults(result, *m_scene);
  }

  if (m_timeChangedCallback)
    m_timeChangedCallback(time);
}

float AnimationManager::getAnimationTime() const
{
  return m_time;
}

void AnimationManager::setAnimationIncrement(float increment)
{
  m_incrementSize = increment;
  if (increment > 0.5f) {
    tsd::core::logWarning(
        "[scene] setting animation increment > 0.5 will cause odd"
        " animation behavior.");
  }
}

float AnimationManager::getAnimationIncrement() const
{
  return m_incrementSize;
}

void AnimationManager::incrementAnimationTime()
{
  auto newTime = m_time + m_incrementSize;
  if (newTime > 1.f)
    newTime = 0.f;
  setAnimationTime(newTime);
}

int AnimationManager::getAnimationTotalFrames() const
{
  return m_totalFrames;
}

void AnimationManager::setAnimationTotalFrames(int frames)
{
  m_totalFrames = std::max(2, frames);
}

int AnimationManager::getAnimationFrame() const
{
  return static_cast<int>(std::round(m_time * (m_totalFrames - 1)));
}

void AnimationManager::setAnimationFrame(int frame)
{
  int clamped = std::clamp(frame, 0, m_totalFrames - 1);
  setAnimationTime(static_cast<float>(clamped) / (m_totalFrames - 1));
}

void AnimationManager::incrementAnimationFrame()
{
  int frame = getAnimationFrame() + 1;
  if (frame >= m_totalFrames)
    frame = 0;
  setAnimationFrame(frame);
}

} // namespace tsd::animation

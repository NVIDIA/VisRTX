// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Animation.hpp"

// std
#include <functional>

namespace tsd::scene {
struct Scene;
} // namespace tsd::scene

namespace tsd::animation {

/*
 * Central owner of all animations in a scene; provides time and frame control
 * and dispatches animation evaluation to each owned Animation instance.
 *
 * Example:
 *   AnimationManager mgr(&scene);
 *   Animation &a = mgr.addAnimation("fade");
 *   mgr.setAnimationTime(1.0f);
 *   mgr.incrementAnimationFrame();
 */
struct AnimationManager
{
  TSD_NOT_COPYABLE(AnimationManager)
  TSD_NOT_MOVEABLE(AnimationManager)

  AnimationManager(tsd::scene::Scene *scene);
  ~AnimationManager();

  using TimeChangedCallback = std::function<void(float)>;
  void setTimeChangedCallback(TimeChangedCallback cb);

  scene::Scene *scene() const;

  // Animation collection
  Animation &addAnimation(const std::string &name = "<unnamed_animation>");
  std::vector<Animation> &animations();
  const std::vector<Animation> &animations() const;
  void removeAnimation(size_t index);
  void removeAllAnimations();

  // Time control
  void setAnimationTime(float time);
  float getAnimationTime() const;
  void setAnimationIncrement(float increment);
  float getAnimationIncrement() const;
  void incrementAnimationTime();

  // Frame control
  int getAnimationTotalFrames() const;
  void setAnimationTotalFrames(int frames);
  int getAnimationFrame() const;
  void setAnimationFrame(int frame);
  void incrementAnimationFrame();

 private:
  scene::Scene *m_scene{nullptr};
  TimeChangedCallback m_timeChangedCallback;
  float m_incrementSize{0.01f};
  float m_time{0.f};
  int m_totalFrames{100};
  std::vector<Animation> m_animations;
  size_t m_defragToken{0};
};

} // namespace tsd::animation

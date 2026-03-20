## Animation Library (`tsd_animation`)

`tsd_animation` provides time-based parameter and transform interpolation for
TSD scenes.

### Components

- **`Binding`** — base class that associates a binding with its owning
  `scene::Scene`. `ObjectParameterBinding` and `TransformBinding` extend it.

- **`core::AnyArray`** — owning, type-aware keyframe value buffer (from
  `tsd_core`) backed by `std::vector<uint8_t>`. Provides typed accessors
  (`dataAs<T>()`, `elementAt(i)`) and automatic copy/move semantics.

- **`Interpolation.hpp`** — defines `InterpolationRule` (STEP, LINEAR, SLERP)
  and `TimeSample` (lo/hi bracket indices + normalized alpha), the lookup
  result of a time query against a sorted time-base.

- **`ObjectParameterBinding`** — drives a single named ANARI parameter on a
  `scene::Object` over time. Stores typed keyframe values in a `core::AnyArray`
  buffer alongside a float time-base and an `InterpolationRule`. The target
  object is tracked via `AnyObjectUsePtr<UseKind::ANIM>`, which provides
  reference-counted, defrag-safe lifetime management. Keyframes can be added
  or removed with `insertKeyframe()` / `removeKeyframe()`.

- **`TransformBinding`** — animates a scene-graph node's transform by
  interpolating decomposed rotation (quaternion `float4`), translation
  (`float3`), and scale (`float3`) keyframes stored in parallel vectors.
  Rotation uses SLERP; translation and scale use linear interpolation.
  Keyframes can be added or removed with `insertKeyframe()` /
  `removeKeyframe()`.

- **`Animation`** — named collection of `ObjectParameterBinding` and
  `TransformBinding` channels. Calling `setAnimationTime(t)` evaluates all
  bindings at `t` and writes the interpolated values directly into the owning
  scene; evaluation and application are encapsulated as private implementation
  details. Supports serialization via `toDataNode()`/`fromDataNode()`.

- **`AnimationManager`** — owns the animation collection for a scene and
  provides time/frame control. Key API:
  - `addAnimation(name)` / `removeAnimation(i)` / `removeAllAnimations()`
  - `setAnimationTime(t)` / `getAnimationTime()`
  - `setAnimationIncrement(dt)` / `incrementAnimationTime()`
  - `setAnimationFrame(n)` / `getAnimationFrame()` /
    `incrementAnimationFrame()` / `getAnimationTotalFrames()`
  - `setTimeChangedCallback(cb)` — optional callback invoked on each time
    change

### Typical Usage

```cpp
// Set up
AnimationManager mgr(&scene);
Animation &anim = mgr.addAnimation("spin");

// Add a transform channel
anim.addTransformBinding(nodeRef, times, rotations, translations, scales, n);

// Add a parameter channel
anim.addObjectParameterBinding(
    obj, "opacity", ANARI_FLOAT32, data, times, count,
    InterpolationRule::LINEAR);

// Advance time — evaluates and applies all bindings to the scene
mgr.setAnimationTime(1.5f);
mgr.incrementAnimationFrame();
```

### Design Notes

- `evaluate()` and `applyResults()` are private to `Animation`; the only
  mutation point visible to callers is `setAnimationTime()` (and the
  frame-control helpers on `AnimationManager` that delegate to it).
- `AnyObjectUsePtr<UseKind::ANIM>` in `ObjectParameterBinding` keeps binding
  targets alive and valid across object pool defragmentation; call
  `Animation::updateObjectDefragmentedIndices()` after a defrag to remap
  stored indices.
- `AnimationManager` is non-copyable and non-moveable; animations live
  entirely inside the manager and are accessed by reference or index.

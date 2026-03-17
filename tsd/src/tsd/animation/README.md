## Animation Library (`tsd_animation`)

`tsd_animation` provides time-based parameter and transform interpolation for
TSD scenes.

### High-Level Concepts

- `Animation` groups named collections of `ObjectParameterBinding` and
  `TransformBinding` channels, each with its own time base and interpolation
  rule (step, linear, slerp).
- `TimeSamples` is a typed 1D data buffer for keyframe values, with support for
  host, CUDA, and proxy memory modes.
- `AnimObjectRef` is an RAII handle that tracks animation binding targets via
  `UseKind::ANIM` reference counting and survives pool defragmentation through
  index remapping.
- `evaluate()` samples all bindings at a given time and returns an
  `EvaluationResult` — a pure data structure of parameter and transform
  substitutions — without mutating the scene.
- `applyResults()` writes an `EvaluationResult` into a live `Scene`.
- `AnimationManager` owns a scene's animation collection and provides time/frame
  control with an optional time-changed callback.

### Why Use This Library

- You need keyframed parameter or transform animation on ANARI scene objects.
- You want evaluation decoupled from application: `evaluate()` is a pure
  function, `applyResults()` is the only mutation point.
- You need reference-counted binding targets that stay valid across object pool
  defragmentation.

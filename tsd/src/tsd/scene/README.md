## Scene Library (`tsd_scene`)

`tsd_scene` is TSD's central scene description library, modeled around ANARI
object concepts and layered scene-graph instancing.

### High-Level Concepts

- `Scene` owns object databases for arrays, geometry, materials, volumes,
  samplers, lights, cameras, renderers, and surfaces.
- Layer graph representation:
  each `Layer` is a transform/object tree (`Forest`) with active/inactive layer
  control for selective rendering.
- Object model:
  `Object` + `Parameter` support typed values, metadata, parameter batching, and
  use-count tracking for cleanup/garbage collection.
- Update notifications:
  `BaseUpdateDelegate` and `MultiUpdateDelegate` forward object/layer/animation
  changes to rendering, networking, or UI subsystems. Every `Scene`
  intrinsically owns a `MultiUpdateDelegate` root exposed via
  `scene.updateDelegate()`.
- Animation system:
  time-step arrays and keyframe channels for parameter and transform
  interpolation.
- Scene maintenance tools:
  remove unused objects, defragment object pools, and cleanup operations.

### Why Use This Library

- You need a mutable C++ scene model that is close to ANARI's object hierarchy.
- You want explicit layer/instance control for view composition and filtering.
- You need scene edits to propagate through rendering/network/UI via a single
  delegate interface.

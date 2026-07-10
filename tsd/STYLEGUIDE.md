# TSD Style Guide

This document covers TSD-specific coding conventions. All general C++ and CUDA
rules from [`../STYLEGUIDE.md`](../STYLEGUIDE.md) apply here as well.

---

## Prefer TSD Primitives Over Standard Alternatives

Before reaching for a standard container or writing a data structure from
scratch, check whether a TSD primitive already fits:

| Need | Use instead of | TSD type |
|---|---|---|
| Ordered key→value map (small, stable keys) | `std::map` / `std::unordered_map` | `FlatMap<K,V>` (`tsd/core/FlatMap.hpp`) |
| Linked list or parent–child tree | `std::list` / hand-rolled | `Forest<T>` / `ForestNode<T>` (`tsd/core/Forest.hpp`) |
| Stable-handle object pool | `std::vector` + index | `ObjectPool<T>` + `ObjectPoolRef<T>` (`tsd/core/ObjectPool.hpp`) |
| ANARI-typed parameter value | `void *` / `std::any` | `tsd::core::Any` (`tsd/core/Any.hpp`) |

---

## Scene Mutation and Notification

- Subclass `BaseUpdateDelegate` for any consumer that needs to react to scene
  mutations (renderer synchronization, network replication, UI refresh).
- Use `MultiUpdateDelegate` to fan notifications out to multiple consumers
  without writing fan-out logic yourself.
- Never bypass the delegate system by mutating scene objects and then calling
  renderer internals directly — that breaks the synchronization contract.

---

## Parameter Builder Pattern

Chain `Parameter` setters rather than setting each property separately:

```cpp
p->setDescription("Sphere radius")
  .setValue(0.5f)
  .setMin(0.f)
  .setMax(10.f);
```

---

## Library Layering

Respect the dependency order — never introduce an upward dependency:

```
tsd_core  →  tsd_scene  →  tsd_io  →  tsd_rendering  →  tsd_app
```

Optional libraries (`tsd_ui_imgui`, `tsd_mpi`, `tsd_network`, `tsd_lua`) may
depend on any layer but must remain optional (CMake-gated).

---

## GPU Computation (`TSD_USE_CUDA`)

Prefer Thrust algorithms over hand-written `__global__` kernels:

```cpp
// Prefer:
thrust::transform(thrust::cuda::par.on(stream), begin, end, out, op);

// Over:
myCustomKernel<<<grid, block, 0, stream>>>(begin, end, out);
```

Write a custom kernel only when no suitable Thrust primitive exists and the
algorithm cannot be composed from existing ones.

---

## File I/O: `verb_FORMAT` Free Functions

Importers, exporters, and generators are **free functions** (never methods),
one per file, named `verb_NOUN` where the noun is an ALL-CAPS format token or a
PascalCase type. The filename mirrors the primary function.

```cpp
void import_OBJ(Scene &, animation::AnimationManager &, const char *filename,
    LayerNodeRef location = {}, bool useDefaultMaterial = false);   // import_OBJ.cpp
SpatialFieldRef import_RAW(Scene &, const char *filename);          // import_RAW.cpp
bool export_SceneToUSD(const Scene &, const char *filename);        // export_*.cpp
void generate_randomSpheres(Scene &, LayerNodeRef location = {});   // generate_*.cpp
```

Full-scene importers share the leading signature
`(Scene &, animation::AnimationManager &, const char *filename, LayerNodeRef location = {}, ...)`.
Resolve the target location with the standard idiom:

```cpp
auto root = location ? location : scene.defaultLayer()->root();
```

**Serializable archives** expose a fixed five-verb family, with the verbs kept
semantically distinct (see `io/CONTEXT.md`):

```cpp
bool                    serialize_ObjectArchive(const Object &, DataNode &);
ArchiveValidationResult validate_ObjectArchive(DataNode &);
Object *                deserialize_ObjectArchive(Scene &, DataNode &, ...);
bool                    save_ObjectArchive(const Object &, const char *filename);
Object *                load_ObjectArchive(Scene &, const char *filename, ...);
```

**Dispatch is a deliberate `if`/`else if` chain** keyed on a format enum or file
extension — there is intentionally **no self-registration factory**. Do not add
one; extend the existing chain (e.g. `io/importers/import_file.cpp`).

---

## Concrete ANARI Object Skeleton

Every object type under `scene/objects/` follows the same skeleton. Use it
verbatim for new object types:

```cpp
struct Geometry : public Object
{
  DECLARE_OBJECT_DEFAULT_LIFETIME(Geometry); // movable, not copyable

  Geometry(Token subtype = tokens::unknown);
  virtual ~Geometry() = default;

  ObjectPoolRef<Geometry> self() const;
  anari::Object makeANARIObject(anari::Device d) const override;
};

using GeometryRef = ObjectPoolRef<Geometry>;

namespace tokens::geometry {
extern const Token cone;   // ... subtype constants, defined in the .cpp
} // namespace tokens::geometry
```

- `DECLARE_OBJECT_DEFAULT_LIFETIME(T)` (`scene/Object.hpp`) is the required
  lifetime declaration for object types — it *is* the object-layer spelling of
  "movable, not copyable." Use it here; use the general `TSD_*` macros (parent
  §6) everywhere else. Do not hand-roll `= delete`/`= default` lists.
- `self()` reacquires a handle null-safely and is always the same one-liner:
  ```cpp
  return scene() ? scene()->getObject<Geometry>(index()) : GeometryRef{};
  ```
- Subtypes and other well-known keys are `extern const Token` constants declared
  in a `namespace tokens::<type>` and defined in the `.cpp` — never bare string
  literals at call sites.
- Type aliases: `XRef = ObjectPoolRef<X>` for a raw pooled handle;
  `XAppRef = ObjectUsePtr<X, Object::UseKind::APP>` for a use-counted handle.
- Seed default parameters in the constructor with the fluent builder (see the
  Parameter Builder Pattern above).

---

## Mirrored CPU/CUDA Algorithms

Algorithms live in `tsd/algorithms/` as **free functions** mirrored across
`tsd::algorithms::cpu::` and `tsd::algorithms::cuda::` with identical
signatures and semantics. CUDA variants ship two overloads: an explicit-stream
version and a convenience wrapper forwarding to stream `0`. Shared enums are
declared once in the CPU header and re-included by the CUDA header.

Call sites select the backend with a guarded fall-through, never an `#else`:

```cpp
#ifdef TSD_ALGORITHMS_HAS_CUDA
  if (b.stream) {
    tsd::algorithms::cuda::outlineObject(b.stream, ...);
    return;
  }
#endif
  tsd::algorithms::cpu::outlineObject(...);
```

Portable device functions use the `TSD_HOST_DEVICE_FCN` / `TSD_DEVICE_FCN`
decorators from `algorithms/math/device_macros.h`; CPU parallelism goes through
the `parallel_for` / `parallel_reduce` shims (which fall back to serial loops
without TBB) — never call TBB directly.

---

## Container Traversal and Sentinels

- TSD containers (`ObjectPool`, `Forest`) are traversed with the free
  `foreach_*` / `forall_*` / `find_*_if` function family taking a forwarding
  functor, **not** STL iterators. Const traversals use the `_const` name suffix
  (`foreach_item_const`, `forall_children_const`) because the callback signature
  cannot disambiguate an overload.
- The "container mechanics mirror `std::`" methods use `snake_case`
  (`insert_first_child`, `erase_subtree`, `is_dense`); higher-level domain
  methods use `camelCase` (`isLeaf`, `numChildren`). This split is deliberate.
- Use the `INVALID_INDEX` sentinel (`core/ObjectPool.hpp`) for absent indices;
  prefer the `TSD_INVALID_INDEX` macro alias at call sites.

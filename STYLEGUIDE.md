# C++ Style Guide

This guide covers coding conventions for the VisRTX project. Mechanical formatting
is handled by `.clang-format`; this document covers everything else.

See [`tsd/STYLEGUIDE.md`](tsd/STYLEGUIDE.md) for TSD-specific addenda.

---

## 1. Formatting

- **Canonical formatter**: `clang-format` with the project's `.clang-format` file.
  Run it before committing:
  ```bash
  clang-format -i <file>
  ```
- **Line length**: 80–100 characters. Longer lines are acceptable when breaking
  would harm readability (e.g., long string literals, complex template signatures).
- **Brace style** (set by `.clang-format`, mixed): opening brace on its **own
  line** for classes/structs, enums, and function definitions; on the **same
  line** for control statements (`if`/`else`, loops), namespaces, and
  `extern "C"` blocks.
- **`clang-format` overrides**: use `// clang-format off` / `// clang-format on`
  sparingly — only for data tables, enum lists, or structured blocks where
  column alignment would otherwise be destroyed.

---

## 2. Naming Conventions

| Entity | Style | Example |
|---|---|---|
| Classes / structs | PascalCase | `ObjectPool`, `RenderIndex` |
| Functions / methods | camelCase | `setParameter()`, `valid()` |
| Private member variables | `m_` + snake_case | `m_storage`, `m_freeIndices` |
| Public data members | camelCase | `size`, `colorType` |
| Local variables | camelCase | `controlPoints`, `diff` |
| Namespaces | lowercase | `tsd::core`, `tsd::scene` |
| Macros | UPPER_SNAKE_CASE | `TSD_NOT_COPYABLE`, `VISRTX_DEVICE` |
| `constexpr` constants | UPPER_SNAKE_CASE | `INVALID_INDEX`, `MAX_LOCAL_STORAGE` |
| Type aliases | PascalCase with suffix | `Ptr`, `Ref`, `element_t` |
| GPU data structs | PascalCase + `GPUData` suffix | `FrameGPUData`, `MaterialGPUData` |

Additional rules:

- No `C`/`I` prefix on class or interface names.
- Getter methods prefer the bare property name (`name()`, `type()`) over
  `getName()`/`getType()`.
- Boolean query methods follow the pattern: `is<T>()`, `valid()`, `empty()`,
  `contains()`.

---

## 3. Headers

- **`#pragma once`** exclusively — no `#ifndef`/`#define` include guards.
- **Include order** (blank line between each group):
  1. Project headers (`"tsd/..."`, `"visrtx/..."`)
  2. External library headers (`<anari/...>`, `<helium/...>`)
  3. Standard library headers (`<vector>`, `<memory>`, …)
- Forward-declare types in headers where possible to minimize include depth.
  Use fully-qualified names in forward declarations.

---

## 4. Namespaces

- Hierarchical, 2–3 levels deep: `tsd::core`, `tsd::scene`, `tsd::rendering`,
  `tsd::io`, `tsd::app`; device namespaces: `visrtx`, `visgl`.
- Sub-namespaces for implementation details: `detail`, `tokens`, `colormap`.
- Anonymous namespaces (`namespace { ... }`) for translation-unit–local linkage
  in `.cpp` files.
- `using namespace` is allowed inside `.cpp` files and inside namespace bodies.
  **Never** at global scope in a header.

---

## 5. Class Layout

Order within a class/struct body:

1. Public type aliases and nested types
2. Public constructors / destructor
3. Lifetime macros (`TSD_NOT_COPYABLE`, `TSD_DEFAULT_MOVEABLE`, …)
4. Public method *declarations* (queries before mutators)
5. `protected` virtual hooks / overrides — *declarations* only
6. `private` data members

**Method definitions are never written inside the class/struct body** — even
trivial one-liners. Sole exception: a truly empty body (`{}`), including
constructors that are only a member-init list, may stay inline. All other
definitions (inline, `constexpr`, and template) go in a clearly delimited
section *after* the class declaration:

```cpp
struct Foo
{
  int bar() const;
  void setBaz(int v);
  // ...
 private:
  int m_bar{0};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline int Foo::bar() const
{
  return m_bar;
}

inline void Foo::setBaz(int v)
{
  m_bar = v;
}
```

This keeps the class declaration readable as an interface, with all
implementation detail below.

---

## 6. Lifetime and Memory Management

- **No raw `new`/`delete`.**
  - `std::unique_ptr<T>` for exclusive ownership.
  - `std::shared_ptr<T>` for shared ownership (use sparingly).
- Non-owning references use raw pointers or project-specific ref wrappers
  (`ObjectPoolRef<T>`).
- Express copy/move intent explicitly. In TSD-layer code, use the macros from
  `tsd/core/TypeMacros.hpp` rather than hand-rolled `= delete`/`= default`
  lists:
  ```cpp
  TSD_NOT_COPYABLE(MyClass)
  TSD_DEFAULT_MOVEABLE(MyClass)
  ```
  (Device code under `devices/` has no TSD dependency — spell intent with
  explicit `= delete`/`= default` there.)
- RAII everywhere — resources are owned and released by objects, never managed
  manually.

---

## 7. C++17 Usage

Use C++17 features freely where they improve clarity:

- `if constexpr` — compile-time branching in templates.
- `std::optional<T>` — nullable return values.
- `std::string_view` — read-only string parameters.
- `std::byte` — raw byte buffers.
- Structured bindings — where they aid readability.

Avoid features that obscure intent or have poor tooling support.

---

## 8. `auto`

**Use `auto` when:**
- The type is immediately obvious from the right-hand side.
- The spelled-out type would be verbose (iterators, template instantiations,
  results of explicit casts).

**Avoid `auto` in:**
- Public API declarations and function signatures.
- Situations where the type is not clear without additional context.

---

## 9. `const` and `constexpr`

- Mark all member functions that do not mutate state `const`.
- Pass large or non-trivial types by `const &`; pass scalars and cheap types
  by value.
- Use `constexpr` for all compile-time constants — not `#define` or
  `static const`.
- Prefer `const` local variables whenever the value does not change after
  initialization.

---

## 10. Templates

- Place full template definitions in headers. They belong in the *Inlined
  definitions* section after the class declaration (same rule as non-template
  inline methods — never inside the class body).
- Use `static_assert` to enforce template parameter constraints early, with a
  clear diagnostic message.
- Avoid CRTP unless virtual dispatch is genuinely unacceptable for performance.
  Prefer virtual functions (see §13) for most extensibility patterns.

---

## 11. Error Handling

| Scenario | Mechanism |
|---|---|
| API misuse (programmer error) | `throw std::runtime_error(...)` |
| Compile-time invariants | `static_assert(condition, "message")` |
| Recoverable / expected failure | Return `bool` or `std::optional` |

Do not use `try`/`catch` inside library code — let exceptions propagate to the
application layer.

### Fallible Returns

Distinguish the two kinds of "not found" by return type:

- **Missing value** → `std::optional<T>`, returned as `return {};` on absence.
- **Missing object / pointer** → raw `T *`, returned as `nullptr` on absence.

Do not mix the two for the same kind of lookup within a type.

---

## 12. Comments

- `/* ... */` block comments for class-level and file-level documentation.
- `//` inline comments sparingly, only where the logic is non-obvious.
- Section divider pattern:
  ```cpp
  // Section name ///////////////////////////////////////////////////////////////
  ```
- No Doxygen `///` triple-slash style.
- Comments explain *why*, not *what* — the code itself should convey what it
  does.

---

## 13. Polymorphic Class Hierarchies

The project's canonical extensibility shape is an abstract base with a
**public non-virtual driving API**, a small set of **protected pure-virtual
hooks** subclasses must implement, and optional virtual hooks with default
no-op bodies. Prefer this over CRTP (see §10).

```cpp
struct ImagePass
{
  // public non-virtual API driven by the owner
  void setEnabled(bool e);
  virtual const char *name() const = 0;

 protected:
  virtual void render(ImageBuffers &b, int stageId) = 0; // required hook
  virtual void updateSize();                             // optional, no-op default

 private:
  friend struct ImagePipeline; // owner drives the protected API
};
```

- Notification/observer bases (e.g. `BaseUpdateDelegate`, `RenderIndex`) expose
  many narrow `signalXxx(...)`/`preChildren`/`postChildren` virtual hooks;
  subclasses override only the ones they care about.
- Provide a Null-Object subclass (all hooks no-op) where a "do nothing" default
  is useful (`EmptyUpdateDelegate`).
- Always null-check an optional delegate/observer before signaling it:
  `if (m_delegate) m_delegate->signalX(...);`

---

## 14. CUDA and OptiX (`devices/rtx`)

### File Types

| Extension | Purpose |
|---|---|
| `.cu` | OptiX programs and CUDA kernels compiled to PTX |
| `.cuh` | Device-side inline utilities and declarations |
| `.cpp` | Host-side management: object lifetime, memory upload, pipeline setup |
| `.h` | Shared definitions visible to both host and device (guarded by `#ifdef __CUDACC__`) |

### Qualifier Macros

Always use the project macros defined in `gpu_decl.h` — never raw CUDA
qualifiers directly:

| Macro | Expands to | Use for |
|---|---|---|
| `VISRTX_HOST_DEVICE` | `inline __host__ __device__` | Math helpers callable from both sides |
| `VISRTX_DEVICE` | `inline __device__` | Device-only helper functions |
| `VISRTX_GLOBAL` | `extern "C" __global__` | CUDA kernels (non-OptiX) |
| `VISRTX_CALLABLE` | `extern "C" __device__` | OptiX direct-callable programs |

### OptiX Program Naming

Follow the OptiX double-underscore convention with an optional descriptive
suffix:

```cpp
VISRTX_GLOBAL void __raygen__()            { ... }
VISRTX_GLOBAL void __closesthit__primary() { ... }
VISRTX_GLOBAL void __anyhit__shadow()      { ... }
VISRTX_GLOBAL void __miss__()              { ... }
VISRTX_CALLABLE void __direct_callable__init() { ... }
```

### GPU Data Structures

Structures passed to the device via launch parameters or constant memory:

- Suffix with `GPUData`: `FrameGPUData`, `MaterialGPUData`, `GeometryGPUData`.
- Fields must be device pointers (`const T *`) — never host-side smart pointers.
- Declare `__constant__` launch parameters with the `DECLARE_FRAME_DATA(name)`
  macro.

### Memory Allocation

- Use `DeviceBuffer` wrappers for managed GPU memory. Avoid bare `cudaMalloc`
  in new code.
- Prefer `thrust` algorithms (`thrust::transform`, `thrust::scan`,
  `thrust::fill`, etc.) over custom `__global__` kernels when a standard
  operation suffices.
- Host-side array objects expose a `gpuData()` method returning the
  corresponding `GPUData` struct with device pointers.

### Host/Device Boundary

- `#ifdef __CUDACC__` guards control CUDA-specific paths in shared headers.
- The `VISRTX_*` macros keep shared math headers compilable by both the C++
  and CUDA compilers without duplication.
- Data crosses the boundary via explicit `cudaMemcpy` or `DeviceBuffer` upload —
  never implicitly.

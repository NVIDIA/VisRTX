# Sharing one MDL SDK instance between ANARI devices

The MDL SDK's `mi_factory` hands out at most **one live `INeuray` per loaded
library image**, and `dlopen`/`LoadLibrary` of the same path returns the same
refcounted image. Any second consumer that calls `mi_factory` gets `nullptr`.

This bites in two places.

## Two VisRTX devices — handled internally

Every `libmdl::Core` in a process shares one refcounted `SharedNeuray`
(`libmdl/Core.cpp`). The SDK is created and started by the first `Core` and
shut down and unloaded when the last one goes away, so creating several VisRTX
devices — overlapping, sequentially, or concurrently — needs nothing from the
application.

Two consequences follow from the instance being shared, both of which the
implementation has to handle because they used to be per-device:

- **Search paths** are a property of the `INeuray` instance, not of a device.
  See `mdl-search-paths.md`.
- **Logging** has a single receiving logger per instance. See below.

## Logging

The SDK takes one receiving logger per `INeuray`, so there is one *master*
logger per instance: the first device to initialize MDL installs its own sink,
and later devices leave it in place. SDK-internal diagnostics — module
compilation and the like — therefore reach that master, whichever device
provoked them.

Messages VisRTX logs itself never pass through the SDK's logger, so they still
reach the device they belong to.

This is a deliberate trade. Attributing SDK diagnostics per device is possible
— route through a shim that consults a thread-local "currently active device"
— but it costs a routing logger, a scope guard on every SDK entry point, and
self-reference guards for instances handed back and forth, to redirect a
trickle of compiler warnings. ANARI status messages are diagnostic rather than
programmatic, so nothing an application does depends on which device a
compiler warning arrives on. An application that cares installs its own logger
on its own `INeuray` and hands that over, which makes the question moot.

So: if you need MDL diagnostics attributed precisely, create and configure the
`INeuray` yourself and pass it in.

## A VisRTX device plus another vendor's — application-mediated

Another MDL-capable ANARI device (Barney, say) in the same process cannot see
VisRTX's internals, and VisRTX cannot see its. The instance therefore has to be
handed over by the application. `VISRTX_MATERIAL_MDL` declares, on the device
object:

- parameter `ineuray` (`ANARI_VOID_POINTER`) — use this instance instead of
  acquiring one. Read before the device initializes MDL; ignored afterwards.
- property `ineuray` (`ANARI_VOID_POINTER`) — the instance this device is
  using, readable once it has initialized MDL.

So the application reads the property from whichever device initialized MDL
first, and sets it as the parameter on every other one:

```c
void *instance = nullptr;
anariGetProperty(first, first, "ineuray", ANARI_VOID_POINTER,
    &instance, sizeof(instance), ANARI_WAIT);
anariSetParameter(second, second, "ineuray", ANARI_VOID_POINTER, &instance);
```

Ownership stays on the providing side. A recipient must not `start()` or
`shutdown()` the instance, and must not outlive it.

Concretely: the instance is shut down and its library unloaded when the last
VisRTX `Core` that *owns* it is destroyed. A device that merely received the
pointer holds a reference to the `INeuray` interface, which is not a lease on
the started SDK or on the loaded image — so **destroy the recipients before the
devices that provided the instance**. Where several VisRTX devices share one
instance automatically they all report the same pointer and none of them is
individually "the" owner; the instance lives until the last of them goes.

Handing one VisRTX device's instance to another VisRTX device is unnecessary —
they already share automatically — and loses that protection, so don't.

> Properties are not introspectable in ANARI: the query generator's info
> vocabulary has no `property` entry, which is why the core specification's own
> `geometryMaxIndex` is absent from generated queries too. The `ineuray`
> property is discoverable from this document, not at runtime.

### Why not something automatic

An automatic rendezvous — devices discovering each other through a shared
broker library and refcounting a published instance — was prototyped and
rejected. It needs a shim library loaded into the global symbol scope, a
published record with its own ABI, cross-binary allocation and locking rules,
and SDK-version arbitration; and it still cannot help when one participant
does not implement it, because that participant takes the instance from
`mi_factory` regardless. The handoff above is a few lines in the application
and has none of that surface.

## Status

Tested on Linux/glibc against MDL SDK 2026.0.2, with
`VISRTX_USE_MDL_FOR_PHYSICALLY_BASED=OFF`.

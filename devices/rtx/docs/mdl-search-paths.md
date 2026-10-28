# MDL search paths across devices

MDL search paths are a property of the `INeuray` instance, not of a device,
and every device in a process shares one instance
(`mdl-instance-sharing.md`). Two consequences follow.

## The SDK only offers clear-and-rebuild

`IMdl_configuration` exposes `add_mdl_path` and `clear_mdl_paths`, with no way
to withdraw a single root or to add one on behalf of a particular owner. A
device that simply applied its own parameter would therefore have to clear
first, silently dropping every other device's roots — and dropping its own
previous roots is exactly what it *does* want, so neither "always clear" nor
"never clear" is right.

`detail::SearchPathRegistry` in `libmdl/Core.cpp` keeps each `Core`'s
contribution separately, keyed by instance and then by owner, and rebuilds the
instance's path list as their union on every change. So:

- setting one device's `mdlSearchPaths` leaves the others' roots in place;
- re-setting the same device's parameter replaces only that device's
  contribution, retiring its previous roots;
- destroying a device withdraws its contribution.

Duplicate roots across devices are added once. System and user paths are
re-appended after every rebuild, since the clear removes them too.

Two limits follow from the clear-and-rebuild shape:

- The registry only knows roots that went through it. On an
  application-supplied instance, roots the application or another vendor's
  device installed directly are not preserved across a rebuild — and
  symmetrically, another vendor doing its own clear-and-rebuild would drop
  VisRTX's. The guarantee covers participating VisRTX devices, not arbitrary
  users of a shared instance.
- Contributions are keyed by owner address, so the order roots are applied in
  is unspecified. If two devices supply roots that each contain a module of
  the same name, which one resolves is unspecified.

The registry is keyed on the `INeuray*` rather than on VisRTX's internal
shared-instance object, so it equally covers devices that adopted an
application-supplied instance through the `ineuray` parameter.

## Withdrawing a root does not unload a module

A module that has been loaded stays in the MDL database and keeps resolving by
name no matter what the search paths later say. Verified against the SDK:
after loading `::m` and then clearing the search paths, `load_module("::m")`
still succeeds.

So withdrawing a root only affects modules that have not been loaded yet. This
is SDK behaviour, not something the registry can change, and it is why
`TestMdlSearchPaths` asserts that new roots *start* resolving rather than that
retired roots stop.

## Testing

`TestMdlSearchPaths` writes a module per directory, each emitting on a
different colour channel, and loads them by module name so resolution has to
come off the search path. The rendered pixel identifies which module was
found; a black pixel means the root was lost. It covers up to three concurrent
devices with disjoint roots, devices leaving, a device updating its own roots,
and a device updating its roots while another is live.

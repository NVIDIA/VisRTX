# Give each dataset its bound animations

A Dataset Archive serializes every core TSD animation whose bindings target
that dataset's layer subtree or object closure, including ordinary animations
created by nominally static importers. Loading and removing the dataset
reconstructs and removes those animations with the dataset. An animation with
bindings spanning multiple datasets is rejected because it has no valid owner
among the self-contained Archives. Dataset Archives do not persist
animation-manager playback state such as the current frame, FPS, play state, or
total timeline length; those belong to the shot that drives the shared
animation manager. A file animation's runtime binding is the exception to
direct animation persistence: it is recreated from its owning dataset's
authoritative source list rather than serving as a second persisted definition.

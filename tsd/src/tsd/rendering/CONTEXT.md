# TSD Rendering

Turns TSD scenes into images: render indexes feed ANARI devices, and an image
pipeline of composable passes produces the final per-pixel output.

## Language

**Image Pipeline**:
An ordered sequence of Image Passes that all read and write a shared set of
per-pixel buffers, executed in insertion order each frame.

**Image Pass**:
A single, independently enable-able stage of an Image Pipeline.

**Outline**:
An image-space silhouette border around an object or primitive, derived from
the rendered ID buffers. An Outline traces what was rendered; it does not
project geometry.
_Avoid_: highlight, selection border

**Box Outline**:
The wireframe formed by the twelve edges of an axis-aligned box, projected
through a camera view. A Box Outline is a generic box drawing; it is not
inherently a bounding box — bounding is one client's interpretation.
_Avoid_: bounding box pass, box wireframe

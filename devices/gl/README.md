# VisGL

VisGL is an ANARI backend written against the OpenGL 4.3 and GLES 3.2 specifications.

## Dependencies

- OpenGL 4.3 or GLES 3.2
- EGL and/or GLX

## OpenGL Context Management

When no additional parameters are provided VisGL attempts to guess how to create an OpenGL context.

If a GLX context is current on the calling thread VisGL will create a sibling context and make it current on an internal thread. Therefore GLX/X needs to allow concurrent threads (see `XInitThreads`). To allow correct shutdown the ANARI device has to be released before freeing the related GLX/X resources.

In case no GLX context is detected VisGL first checks for a current `EGLDisplay` and if that fails it will attempt to create an offscreen EGL context. Alternatively an `EGLDisplay` can be passed to the `"EGLDisplay"` device parameter.

The underlying API can be selected by setting the device parameter `"glAPI"` to `"OpenGL"` or `"OpenGL_ES"`. Setting the `"glDebug"` boolean parameter to true enables the OpenGL debug output and forwards it to the ANARI status callback.

## Renderer

VisGL only has a single renderer type `"default"`.

In addition to the `KHR_RENDERER_BACKGROUND_COLOR` and `KHR_RENDERER_AMBIENT_LIGHT` features the renderer supports the following parameters:

| Name            | Type         | Default   | Description                                                    |
|:----------------|:-------------|----------:|:---------------------------------------------------------------|
| shadowMapSize   | INT32        |         0 | Shadow map width and height. Implementation defined if 0.      |
| occlusionMode   | STRING       |  `"none"` | Allowed values: `"none"`, `"incremental"`, `"firstFrame"`      |

If `occlusionMode` is set to a value other than `none` ambient occlusion is approximated by baking per vertex/primitive occlusion into the geometry.

In `incremental` mode occlusion samples are collected each frame up to a threshold this allows the renderer to remain interactive while the occlusion is converging. In `firstFrame` mode the baking is fully computed before the first frame is rendered. This mode is intended for offline rendering situations where every frame should be fully converged.

# Known Issues

WGL/Windows support is not yet implemented

Some of the reported ANARI features are still aspirational. The following may not be (fully) implemented yet:

- Cone geometry
- Curve geometry
- Quad geometry
- Omnidirectional Camera
- Indexed Spheres
- Only directional lights cast shadows

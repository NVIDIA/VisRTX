// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "Renderer.h"

namespace visgl2 {

// Static data ////////////////////////////////////////////////////////////////

static const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 MVP;
void main() {
    gl_Position = MVP * vec4(aPos, 1.0);
})";

static const char *fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 0.5, 0.2, 1.0);
})";

// Renderer definitions ///////////////////////////////////////////////////////

Renderer::Renderer(VisGL2DeviceGlobalState *s) : Object(ANARI_RENDERER, s)
{
  gl_enqueue_method(this, &Renderer::ogl_initShaders).wait();
}

Renderer *Renderer::createInstance(
    std::string_view /* subtype */, VisGL2DeviceGlobalState *s)
{
  return new Renderer(s);
}

void Renderer::commitParameters()
{
  m_background = getParam<vec4>("background", vec4(vec3(0.f), 1.f));
}

vec4 Renderer::background() const
{
  return m_background;
}

void Renderer::ogl_initShaders()
{
  auto &state = *deviceState();
  auto &gl = state.gl.glAPI;

  auto vertexShader = gl.CreateShader(GL_VERTEX_SHADER);
  gl.ShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
  gl.CompileShader(vertexShader);

  auto fragmentShader = gl.CreateShader(GL_FRAGMENT_SHADER);
  gl.ShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
  gl.CompileShader(fragmentShader);

  auto shaderProgram = gl.CreateProgram();
  gl.AttachShader(shaderProgram, vertexShader);
  gl.AttachShader(shaderProgram, fragmentShader);
  gl.LinkProgram(shaderProgram);

  gl.DeleteShader(vertexShader);
  gl.DeleteShader(fragmentShader);

  m_glState.shaderProgram = shaderProgram;
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::Renderer *);

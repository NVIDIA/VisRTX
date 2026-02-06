/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#define MATERIAL_COMMIT_ATTRIBUTE(PARAM, TYPE, INDEX)                          \
  if (current.PARAM.type() == TYPE) {                                          \
    std::array<float, 4> color = {0, 0, 0, 1};                                 \
    current.PARAM.get(TYPE, color.data());                                     \
    thisDevice->materials.set(material_index + INDEX, color);                  \
  } else if (current.PARAM.type() == ANARI_SAMPLER) {                          \
    if (auto sampler = acquire<SamplerObjectBase *>(current.PARAM)) {          \
      auto meta = sampler->metadata();                                         \
      thisDevice->materials.setMem(material_index + INDEX, &meta);             \
    }                                                                          \
  }

#define ALLOCATE_SAMPLERS(PARAM, SLOT)                                         \
  if (current.PARAM.type() == ANARI_SAMPLER) {                                 \
    if (auto sampler = acquire<SamplerObjectBase *>(current.PARAM)) {          \
      sampler->allocateResources(surf, SLOT);                                  \
    }                                                                          \
  }

#define MATERIAL_DRAW_COMMAND(PARAM, SLOT)                                     \
  if (current.PARAM.type() == ANARI_SAMPLER) {                                 \
    if (auto sampler = acquire<SamplerObjectBase *>(current.PARAM)) {          \
      int index = surf->resourceIndex(SLOT);                                   \
      sampler->drawCommand(index, command);                                    \
    }                                                                          \
  }

#define MATERIAL_FRAG_DECL(PARAM, SLOT)                                        \
  if (current.PARAM.type() == ANARI_SAMPLER) {                                 \
    if (auto sampler = acquire<SamplerObjectBase *>(current.PARAM)) {          \
      int index = surf->resourceIndex(SLOT);                                   \
      sampler->declare(index, shader);                                         \
    }                                                                          \
  }

#define MATERIAL_FRAG_SAMPLE(VAR, PARAM, TYPE, INDEX, SLOT)                    \
  shader.append("  vec4 " VAR " = ");                                          \
  if (current.PARAM.type() == TYPE) {                                          \
    shader.append("materials[instanceIndices.y+" #INDEX "u];\n");              \
  } else if (current.PARAM.type() == ANARI_STRING) {                           \
    shader.append(current.PARAM.getString());                                  \
    shader.append(semicolon);                                                  \
  } else if (current.PARAM.type() == ANARI_SAMPLER) {                          \
    if (auto sampler = acquire<SamplerObjectBase *>(current.PARAM)) {          \
      int index = surf->resourceIndex(SLOT);                                   \
      sampler->sample(                                                         \
          index, shader, "materials[instanceIndices.y+" #INDEX "u]\n");        \
    } else {                                                                   \
      shader.append("vec4(1.0, 0.0, 1.0, 1.0);\n");                            \
    }                                                                          \
  } else {                                                                     \
    shader.append("vec4(1.0, 0.0, 1.0, 1.0);\n");                              \
  }

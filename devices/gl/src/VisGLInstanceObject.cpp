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

#include "VisGLSpecializations.h"
#include "math_util.h"

namespace visgl {

Object<InstanceTransform>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  transform_index = thisDevice->transforms.allocate(3);

  commit();
}

void Object<InstanceTransform>::commit()
{
  DefaultObject::commit();

  current.transform.get(ANARI_FLOAT32_MAT4, instanceTransform.data());

  setInverse(inverseTransform.data(), instanceTransform.data());
  setNormalTransform(normalTransform.data(), instanceTransform.data());

  dirty = true;
}

void Object<InstanceTransform>::update()
{
  DefaultObject::update();
  if (dirty) {
    thisDevice->transforms.set(transform_index, instanceTransform);
    thisDevice->transforms.set(transform_index + 1, inverseTransform);
    thisDevice->transforms.set(transform_index + 2, normalTransform);
    dirty = false;
  }
}

const std::array<float, 16> &Object<InstanceTransform>::transform()
{
  return instanceTransform;
}

uint32_t Object<InstanceTransform>::index()
{
  return transform_index;
}

} // namespace visgl

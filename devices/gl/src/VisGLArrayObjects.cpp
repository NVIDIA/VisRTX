/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "anari2gl_types.h"
#include "VisGLObject.h"
#include "VisGLArrayObjects.h"
#include "shader_blocks.h"
#include "math_util.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>

namespace visgl {

static ANARIDataType promotion_type(ANARIDataType t)
{
  switch (t) {
  case ANARI_INT8_VEC3: return ANARI_INT8_VEC4;
  case ANARI_UINT8_VEC3: return ANARI_UINT8_VEC4;

  case ANARI_INT16_VEC3: return ANARI_INT16_VEC4;
  case ANARI_UINT16_VEC3: return ANARI_UINT16_VEC4;

  case ANARI_FIXED8_VEC3: return ANARI_FIXED8_VEC4;
  case ANARI_UFIXED8_VEC3: return ANARI_UFIXED8_VEC4;

  case ANARI_FIXED16_VEC3: return ANARI_FIXED16_VEC4;
  case ANARI_UFIXED16_VEC3: return ANARI_UFIXED16_VEC4;

  default: return t;
  }
}

template <int T>
struct buffer_type
{
  static const int value = T;
};

template <>
struct buffer_type<ANARI_INT8_VEC3>
{
  static const int value = ANARI_INT8_VEC4;
};

template <>
struct buffer_type<ANARI_UINT8_VEC3>
{
  static const int value = ANARI_UINT8_VEC4;
};

template <>
struct buffer_type<ANARI_INT16_VEC3>
{
  static const int value = ANARI_INT16_VEC4;
};

template <>
struct buffer_type<ANARI_UINT16_VEC3>
{
  static const int value = ANARI_UINT16_VEC4;
};

template <>
struct buffer_type<ANARI_FIXED8_VEC3>
{
  static const int value = ANARI_FIXED8_VEC4;
};

template <>
struct buffer_type<ANARI_UFIXED8_VEC3>
{
  static const int value = ANARI_UFIXED8_VEC4;
};

template <>
struct buffer_type<ANARI_FIXED16_VEC3>
{
  static const int value = ANARI_FIXED16_VEC4;
};

template <>
struct buffer_type<ANARI_UFIXED16_VEC3>
{
  static const int value = ANARI_UFIXED16_VEC4;
};

template <int T1, int T2>
static void converting_copy(void *dst0, const void *src0, uint64_t N)
{
  using props1 = anari::ANARITypeProperties<T1>;
  using props2 = anari::ANARITypeProperties<T2>;

  typename props1::base_type *dst =
      static_cast<typename props1::base_type *>(dst0);
  const typename props2::base_type *src =
      static_cast<const typename props2::base_type *>(src0);

  for (uint64_t i = 0; i < N; ++i) {
    float vec4[4];
    props2::toFloat4(vec4, src + props2::components * i);
    props1::fromFloat4(dst + props1::components * i, vec4);
  }
}

static void managed_deleter(const void *, const void *appMemory)
{
  std::free(const_cast<void *>(appMemory));
}

template <int T, typename Enable = void>
class TypedArray1D : public DefaultObject<Array1D, ArrayObjectBase>
{
 public:
  TypedArray1D(ANARIDevice d,
      ANARIObject handle,
      const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType type,
      uint64_t numItems1)
      : DefaultObject(d, handle)
  {}

  uint64_t size() const override
  {
    return 0;
  }
  void *map() override
  {
    return nullptr;
  }
  void unmap() override {}
  ANARIDataType getElementType() const override
  {
    return T;
  }
  int dims(uint64_t *d) const override
  {
    d[0] = 0;
    return 1;
  }
  std::array<float, 4> at(uint64_t) const
  {
    return std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f};
  }
};

template <int T>
class TypedArray1D<T, typename std::enable_if<T >= 1000>::type>
    : public DefaultObject<Array1D, DataArray1D>
{
  using props = anari::ANARITypeProperties<T>;
  using base_type = typename props::base_type;
  using array_type = typename props::array_type;

  static const int elementType = T;
  static const int bufferType = buffer_type<T>::value;

  const void *appMemory;
  ANARIMemoryDeleter deleter;
  const void *userdata;
  uint64_t numItems1;

  void *mapping = nullptr;
  GLuint buffer = 0;
  GLuint texture = 0;
  std::future<void> future;

  std::array<float, 6> bounds;

  static void array_allocate_buffer(ObjectRef<TypedArray1D<T>> arrayObj)
  {
    auto &gl = arrayObj->thisDevice->gl;

    uint64_t allocationSize = arrayObj->numItems1 * anari::sizeOf(bufferType);

    gl.GenBuffers(1, &arrayObj->buffer);
    gl.BindBuffer(GL_ARRAY_BUFFER, arrayObj->buffer);
    if (elementType == bufferType) {
      gl.BufferData(GL_ARRAY_BUFFER,
          allocationSize,
          arrayObj->appMemory,
          GL_DYNAMIC_DRAW);
    } else {
      gl.BufferData(GL_ARRAY_BUFFER, allocationSize, nullptr, GL_DYNAMIC_DRAW);

      void *dst = gl.MapBufferRange(GL_ARRAY_BUFFER,
          0,
          allocationSize,
          GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
      const void *src = arrayObj->appMemory;

      converting_copy<bufferType, elementType>(dst, src, arrayObj->numItems1);

      gl.UnmapBuffer(GL_ARRAY_BUFFER);
    }
  }

  static void array_map(
      ObjectRef<TypedArray1D<T>> arrayObj, void **mapping, uint64_t size)
  {
    auto &gl = arrayObj->thisDevice->gl;
    gl.BindBuffer(GL_ARRAY_BUFFER, arrayObj->buffer);
    *mapping = gl.MapBufferRange(GL_ARRAY_BUFFER,
        0,
        size,
        GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
  }

  static void array1d_crate_texture(ObjectRef<TypedArray1D<T>> arrayObj)
  {
    auto &gl = arrayObj->thisDevice->gl;
    if (gl.ES_VERSION_3_2) {
      gl.GenTextures(1, &arrayObj->texture);
      gl.BindTexture(GL_TEXTURE_2D, arrayObj->texture);
      gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, arrayObj->buffer);
      gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1);
      gl.TexImage2D(GL_TEXTURE_2D,
          0,
          gl_internal_format(bufferType),
          arrayObj->numItems1,
          1,
          0,
          gl_format(bufferType),
          gl_type(bufferType),
          0);
    } else {
      gl.GenTextures(1, &arrayObj->texture);
      gl.BindTexture(GL_TEXTURE_1D, arrayObj->texture);
      gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, arrayObj->buffer);
      gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1);
      gl.TexImage1D(GL_TEXTURE_1D,
          0,
          gl_internal_format(bufferType),
          arrayObj->numItems1,
          0,
          gl_format(bufferType),
          gl_type(bufferType),
          0);
    }
  }

  static void array1d_update_texture(ObjectRef<TypedArray1D<T>> &arrayObj)
  {
    auto &gl = arrayObj->thisDevice->gl;
    if (gl.ES_VERSION_3_2) {
      gl.BindTexture(GL_TEXTURE_2D, arrayObj->texture);
      gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, arrayObj->buffer);
      gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1);
      gl.TexImage2D(GL_TEXTURE_2D,
          0,
          gl_internal_format(bufferType),
          arrayObj->numItems1,
          1,
          0,
          gl_format(bufferType),
          gl_type(bufferType),
          0);
    } else {
      gl.BindTexture(GL_TEXTURE_1D, arrayObj->texture);
      gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, arrayObj->buffer);
      gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1);
      gl.TexImage1D(GL_TEXTURE_1D,
          0,
          gl_internal_format(bufferType),
          arrayObj->numItems1,
          0,
          gl_format(bufferType),
          gl_type(bufferType),
          0);
    }
  }

  static void array_unmap(ObjectRef<TypedArray1D<T>> arrayObj)
  {
    auto &gl = arrayObj->thisDevice->gl;

    if (bufferType != elementType) {
      uint64_t bufferSize = anari::sizeOf(bufferType) * arrayObj->numItems1;
      uint64_t hostSize = anari::sizeOf(elementType) * arrayObj->numItems1;
      uint64_t offset = bufferSize - hostSize;

      converting_copy<bufferType, elementType>(arrayObj->mapping,
          (const char *)arrayObj->mapping + offset,
          arrayObj->numItems1);
    }

    gl.BindBuffer(GL_ARRAY_BUFFER, arrayObj->buffer);
    gl.UnmapBuffer(GL_ARRAY_BUFFER);
    if (arrayObj->texture) {
      array1d_update_texture(arrayObj);
    }
  }

  static void array_unmap_copy(ObjectRef<TypedArray1D<T>> arrayObj)
  {
    auto &gl = arrayObj->thisDevice->gl;
    gl.BindBuffer(GL_ARRAY_BUFFER, arrayObj->buffer);

    uint64_t allocationSize = anari::sizeOf(bufferType) * arrayObj->numItems1;

    if (elementType == bufferType) {
      gl.BufferData(GL_ARRAY_BUFFER,
          allocationSize,
          arrayObj->appMemory,
          GL_DYNAMIC_DRAW);
    } else {
      void *dst = gl.MapBufferRange(GL_ARRAY_BUFFER,
          0,
          allocationSize,
          GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
      const void *src = arrayObj->appMemory;
      converting_copy<bufferType, elementType>(dst, src, arrayObj->numItems1);
      gl.UnmapBuffer(GL_ARRAY_BUFFER);
    }

    if (arrayObj->texture) {
      array1d_update_texture(arrayObj);
    }
  }

  void calc_bounds(const void *mem)
  {
    if (mem == nullptr) {
      return;
    }
    bounds[0] = bounds[1] = bounds[2] = FLT_MAX;
    bounds[3] = bounds[4] = bounds[5] = -FLT_MAX;

    const base_type *array = static_cast<const base_type *>(mem);
    if (T == ANARI_FLOAT32_VEC3) {
      for (uint64_t i = 0; i < numItems1; ++i) {
        float vec4[4];
        props::toFloat4(vec4, array + i * props::components);
        bounds[0] = fast_minf(bounds[0], vec4[0]);
        bounds[1] = fast_minf(bounds[1], vec4[1]);
        bounds[2] = fast_minf(bounds[2], vec4[2]);
        bounds[3] = fast_maxf(bounds[3], vec4[0]);
        bounds[4] = fast_maxf(bounds[4], vec4[1]);
        bounds[5] = fast_maxf(bounds[5], vec4[2]);
      }
    } else if (T == ANARI_FLOAT32) {
      for (uint64_t i = 0; i < numItems1; ++i) {
        float vec4[4];
        props::toFloat4(vec4, array + i * props::components);
        bounds[0] = fast_minf(bounds[0], -vec4[0]);
        bounds[1] = fast_minf(bounds[1], -vec4[0]);
        bounds[2] = fast_minf(bounds[2], -vec4[0]);
        bounds[3] = fast_maxf(bounds[3], vec4[0]);
        bounds[4] = fast_maxf(bounds[4], vec4[0]);
        bounds[5] = fast_maxf(bounds[5], vec4[0]);
      }
    }
  }

 public:
  TypedArray1D(ANARIDevice d,
      ANARIObject handle,
      const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType type,
      uint64_t numItems1)
      : DefaultObject(d, handle),
        appMemory(appMemory),
        deleter(deleter),
        userdata(userdata),
        numItems1(numItems1)
  {
  }

  void init() override
  {
    if (!anari::isObject(elementType)) {
      future = thisDevice->queue.enqueue(array_allocate_buffer, this);
      calc_bounds(appMemory);
    }
  }

  void *map() override
  {
    if (appMemory) {
      return const_cast<void *>(appMemory);
    } else {
      mapping = nullptr;
      uint64_t bufferSize = anari::sizeOf(bufferType) * numItems1;
      uint64_t hostSize = anari::sizeOf(elementType) * numItems1;

      thisDevice->queue.enqueue(array_map, this, &mapping, bufferSize).wait();

      uint64_t offset = bufferSize - hostSize;
      return ((char *)mapping) + offset;
    }
  }

  void unmap() override
  {
    if (appMemory) {
      calc_bounds(appMemory);
      future = thisDevice->queue.enqueue(array_unmap_copy, this);
    } else {
      calc_bounds(mapping);
      thisDevice->queue.enqueue(array_unmap, this);
    }
    lastEpoch = anariIncrementEpoch(thisDevice, this);
  }

  std::array<float, 6> getBounds() override
  {
    return bounds;
  }

  std::array<float, 4> at(uint64_t i) const override
  {
    std::array<float, 4> result{0.0f, 0.0f, 0.0f, 1.0f};
    if (appMemory && i < numItems1) {
      const base_type *array = static_cast<const base_type *>(appMemory);
      props::toFloat4(result.data(), array + i * props::components);
    }
    return result;
  }

  ANARIDataType getElementType() const override
  {
    return elementType;
  }

  ANARIDataType getBufferType() const override
  {
    return bufferType;
  }

  uint64_t size() const override
  {
    return numItems1;
  }

  int dims(uint64_t *d) const override
  {
    d[0] = numItems1;
    return 1;
  }

  void declare(int index, AppendableShader &shader) override
  {
    shader.append(glsl_sample_array(bufferType, index));
  }

  void sample(int index, AppendableShader &shader) override
  {
    shader.append(ssboArraySampleFun[index]);
  }

  void drawCommand(int index, DrawCommand &command) override
  {
    auto &ssbo = command.ssbos[command.ssbocount];
    ssbo.index = index;
    ssbo.buffer = buffer;
    command.ssbocount += 1;
  }

  GLuint getTexture1D() override
  {
    if (texture == 0 && !anari::isObject(elementType)) {
      thisDevice->queue.enqueue(array1d_crate_texture, this).wait();
    }
    return texture;
  }

  GLuint getBuffer() override
  {
    return buffer;
  }

  void releasePublic() override
  {
    if (future.valid()) {
      future.wait();
    }
    if (deleter == nullptr && appMemory != nullptr) {
      void *internalized = std::malloc(numItems1 * sizeof(array_type));
      std::memcpy(internalized, appMemory, numItems1 * sizeof(array_type));
      appMemory = internalized;
      deleter = managed_deleter;
    }
  }

  ~TypedArray1D()
  {
    if (deleter) {
      deleter(userdata, appMemory);
    }
  }
};

template <int T>
class TypedArray1D<T, typename std::enable_if<anari::isObject(T)>::type>
    : public DefaultObject<Array1D, ObjectArray1D>
{
  using props = anari::ANARITypeProperties<T>;
  using base_type = typename props::base_type;
  using array_type = typename props::array_type;

  static const int elementType = T;

  const void *appMemory;
  ANARIMemoryDeleter deleter;
  const void *userdata;
  uint64_t numItems1;

  std::vector<base_type> objectArray;

 public:
  TypedArray1D(ANARIDevice d,
      ANARIObject handle,
      const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType type,
      uint64_t numItems1)
      : DefaultObject(d, handle),
        appMemory(appMemory),
        deleter(deleter),
        userdata(userdata),
        numItems1(numItems1)
  {
    if (appMemory == nullptr) {
      this->appMemory = std::calloc(numItems1, sizeof(base_type));
      this->deleter = managed_deleter;
    }
    objectArray.resize(numItems1);
    const base_type *basePtr = static_cast<const base_type *>(this->appMemory);
    for (uint64_t i = 0; i < numItems1; ++i) {
      objectArray[i] = basePtr[i];
      anariRetainInternal(device, objectArray[i], handle);
    }
  }

  void *map() override
  {
    return const_cast<void *>(appMemory);
  }
  void unmap() override
  {
    const base_type *basePtr = static_cast<const base_type *>(appMemory);
    for (uint64_t i = 0; i < numItems1; ++i) {
      anariReleaseInternal(device, objectArray[i], handle);
      objectArray[i] = basePtr[i];
      anariRetainInternal(device, objectArray[i], handle);
    }
    lastEpoch = anariIncrementEpoch(thisDevice, this);
  }
  void releasePublic() override
  {
    if (deleter) {
      deleter(userdata, appMemory);
      deleter = nullptr;
    }
    appMemory = nullptr;
  }
  void accept(ObjectVisitorBase *visitor) override
  {
    visitor->visit(this);
  }
  void traverse(ObjectVisitorBase *visitor) override
  {
    // since this is an array we visit the elements of the array instead of
    // parameters
    if (anari::isObject(elementType)) {
      for (uint64_t i = 0; i < numItems1; ++i) {
        if (auto child = handle_cast<ObjectBase *>(objectArray[i])) {
          child->accept(visitor);
        }
      }
    }
  }
  ANARIDataType getElementType() const override
  {
    return T;
  }
  uint64_t size() const override
  {
    return numItems1;
  }
  int dims(uint64_t *d) const override
  {
    d[0] = numItems1;
    return 1;
  }

  ~TypedArray1D()
  {
    for (auto handle : objectArray) {
      anariReleaseInternal(device, handle, handle);
    }
    if (deleter) {
      deleter(userdata, appMemory);
    }
  }
};

template <int T>
struct TypedAllocator
{
  ArrayObjectBase *operator()(ANARIDevice d,
      ANARIObject handle,
      const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType type,
      uint64_t numItems1)
  {
    return new TypedArray1D<T>(
        d, handle, appMemory, deleter, userdata, type, numItems1);
  }
};

ArrayObjectBase *ObjectAllocator<Array1D>::allocate(ANARIDevice d,
    ANARIObject handle,
    const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userdata,
    ANARIDataType type,
    uint64_t numItems1)
{
  return anari::anariTypeInvoke<ArrayObjectBase *, TypedAllocator>(type,
      d,
      handle,
      appMemory,
      deleter,
      userdata,
      type,
      numItems1);
}

void array2d_allocate_objects(ObjectRef<Array2D> arrayObj)
{
  auto &gl = arrayObj->thisDevice->gl;
  gl.GenTextures(1, &arrayObj->texture);
  gl.BindTexture(GL_TEXTURE_2D, arrayObj->texture);
  gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1);
  gl.TexImage2D(GL_TEXTURE_2D,
      0,
      gl_internal_format(arrayObj->elementType),
      arrayObj->numItems1,
      arrayObj->numItems2,
      0,
      gl_format(arrayObj->elementType),
      gl_type(arrayObj->elementType),
      arrayObj->appMemory);
  if (gl_mipmappable(gl_internal_format(arrayObj->elementType))) {
    gl.GenerateMipmap(GL_TEXTURE_2D);
  }
}

Object<Array2D>::Object(ANARIDevice d,
    ANARIObject handle,
    const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userdata,
    ANARIDataType elementType,
    uint64_t numItems1,
    uint64_t numItems2)
    : DefaultObject(d, handle),
      appMemory(appMemory),
      deleter(deleter),
      userdata(userdata),
      elementType(elementType),
      numItems1(numItems1),
      numItems2(numItems2)
{
  if (this->appMemory == nullptr) {
    size_t byte_size = anari::sizeOf(elementType) * numItems1 * numItems2;
    this->appMemory = std::malloc(byte_size);
    deleter = managed_deleter;
  }
}
void Object<Array2D>::init()
{
  future = thisDevice->queue.enqueue(array2d_allocate_objects, this);
}

int Object<Array2D>::dims(uint64_t *d) const
{
  d[0] = numItems1;
  d[1] = numItems2;
  return 2;
}

ANARIDataType Object<Array2D>::getElementType() const
{
  return elementType;
}

void *Object<Array2D>::map()
{
  if (future.valid()) {
    future.wait();
  }
  return const_cast<void *>(appMemory);
}
void array2d_unmap_to_buffer(ObjectRef<Array2D> arrayObj)
{
  auto &gl = arrayObj->thisDevice->gl;
  gl.BindTexture(GL_TEXTURE_2D, arrayObj->texture);
  gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  gl.TexSubImage2D(GL_TEXTURE_2D,
      0,
      0,
      0,
      arrayObj->numItems1,
      arrayObj->numItems2,
      gl_format(arrayObj->elementType),
      gl_type(arrayObj->elementType),
      arrayObj->appMemory);
  if (gl_mipmappable(gl_internal_format(arrayObj->elementType))) {
    gl.GenerateMipmap(GL_TEXTURE_2D);
  }
}
void Object<Array2D>::unmap()
{
  future = thisDevice->queue.enqueue(array2d_unmap_to_buffer, this);
  lastEpoch = anariIncrementEpoch(thisDevice, this);
}
void Object<Array2D>::releasePublic()
{
  if (future.valid()) {
    future.wait();
  }
}
GLuint Object<Array2D>::getTexture2D()
{
  return texture;
}

static void delete_texture(Object<Device> *deviceObj, GLuint texture)
{
  deviceObj->gl.DeleteTextures(1, &texture);
}

Object<Array2D>::~Object()
{
  if (deleter) {
    deleter(userdata, appMemory);
  }
  thisDevice->queue.enqueue(delete_texture, thisDevice, texture);
}

void array3d_allocate_objects(ObjectRef<Array3D> arrayObj)
{
  auto &gl = arrayObj->thisDevice->gl;
  gl.GenTextures(1, &arrayObj->texture);
  gl.BindTexture(GL_TEXTURE_3D, arrayObj->texture);
  gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  gl.TexImage3D(GL_TEXTURE_3D,
      0,
      gl_internal_format(arrayObj->elementType),
      arrayObj->numItems1,
      arrayObj->numItems2,
      arrayObj->numItems3,
      0,
      gl_format(arrayObj->elementType),
      gl_type(arrayObj->elementType),
      arrayObj->appMemory);
}

Object<Array3D>::Object(ANARIDevice d,
    ANARIObject handle,
    const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userdata,
    ANARIDataType elementType,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
    : DefaultObject(d, handle),
      appMemory(appMemory),
      deleter(deleter),
      userdata(userdata),
      elementType(elementType),
      numItems1(numItems1),
      numItems2(numItems2),
      numItems3(numItems3)
{
  if (this->appMemory == nullptr) {
    size_t byte_size =
        anari::sizeOf(elementType) * numItems1 * numItems2 * numItems3;
    this->appMemory = std::malloc(byte_size);
    deleter = managed_deleter;
  }
}
void Object<Array3D>::init()
{
  future = thisDevice->queue.enqueue(array3d_allocate_objects, this);
}

ANARIDataType Object<Array3D>::getElementType() const
{
  return elementType;
}

int Object<Array3D>::dims(uint64_t *d) const
{
  d[0] = numItems1;
  d[1] = numItems2;
  d[2] = numItems3;
  return 3;
}

void *Object<Array3D>::map()
{
  if (future.valid()) {
    future.wait();
  }
  return const_cast<void *>(appMemory);
}

void array3d_unmap_to_buffer(ObjectRef<Array3D> arrayObj)
{
  auto &gl = arrayObj->thisDevice->gl;
  gl.BindTexture(GL_TEXTURE_3D, arrayObj->texture);
  gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  gl.TexSubImage3D(GL_TEXTURE_3D,
      0,
      0,
      0,
      0,
      arrayObj->numItems1,
      arrayObj->numItems2,
      arrayObj->numItems3,
      gl_format(arrayObj->elementType),
      gl_type(arrayObj->elementType),
      arrayObj->appMemory);
}
void Object<Array3D>::unmap()
{
  future = thisDevice->queue.enqueue(array3d_unmap_to_buffer, this);
  lastEpoch = anariIncrementEpoch(thisDevice, this);
}
void Object<Array3D>::releasePublic()
{
  // internalize data if necessary
}
GLuint Object<Array3D>::getTexture3D()
{
  return texture;
}

Object<Array3D>::~Object()
{
  if (deleter) {
    deleter(userdata, appMemory);
  }
  thisDevice->queue.enqueue(delete_texture, thisDevice, texture);
}

} // namespace visgl

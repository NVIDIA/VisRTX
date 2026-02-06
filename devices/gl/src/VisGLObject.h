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

#include <atomic>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <array>
#include <type_traits>
#include "anari/anari.h"

#include "VisGLParameter.h"

#include "DrawCommand.h"
#include "AppendableShader.h"

namespace visgl {

class ParameterPack;
class ParameterBase;
class ObjectVisitorBase;

void anariDeleteInternal(ANARIDevice, ANARIObject);

template <class T>
struct base_flags
{
  static const uint32_t value = ~0u;
};

class ObjectBase
{
  template <class T>
  friend class ObjectRef;

  std::atomic<uint64_t> refcount;

 public:
  const ANARIDevice device;
  const ANARIObject handle;

  ObjectBase(ANARIDevice d, ANARIObject handle)
      : refcount(1), device(d), handle(handle)
  {}
  // init is called after the object has been constructed
  // and inserted into the handle storage
  virtual void init() {}
  virtual void retain()
  {
    refcount += 1;
  }
  virtual void release()
  {
    uint64_t c = refcount.fetch_sub(1);
    if (c == 1) {
      anariDeleteInternal(device, handle);
    } else if ((c & UINT64_C(0xFFFFFFFF)) == 1) {
      releasePublic();
    }
  }
  virtual void retainInternal(ANARIObject)
  {
    refcount += UINT64_C(0x100000000);
  }
  virtual void releaseInternal(ANARIObject)
  {
    uint64_t c = refcount.fetch_sub(UINT64_C(0x100000000));
    if (c == UINT64_C(0x100000000)) {
      anariDeleteInternal(device, handle);
    }
  }
  virtual void releasePublic() {}

  virtual bool set(
      const char *paramname, ANARIDataType type, const void *mem) = 0;
  virtual void unset(const char *paramname) = 0;
  virtual void unsetAll() = 0;

  virtual void* mapParameter1D(const char *paramname,
    ANARIDataType type,
    uint64_t numElements1,
    uint64_t *stride) = 0;
  virtual void* mapParameter2D(const char *paramname,
    ANARIDataType type,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t *stride) = 0;
  virtual void* mapParameter3D(const char *paramname,
    ANARIDataType type,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t numElements3,
    uint64_t *stride) = 0;
  virtual void unmapParameter(const char *paramname) = 0;

  virtual void commit() = 0;
  virtual int getProperty(const char *propname,
      ANARIDataType type,
      void *mem,
      uint64_t size,
      ANARIWaitMask mask) = 0;
  virtual ~ObjectBase() {}
  virtual ANARIDataType type() const = 0;
  virtual const char *subtype() const = 0;
  virtual uint32_t id() const = 0;
  virtual ParameterPack &parameters() = 0;
  template <class T>
  T handle_cast(ANARIObject h);
  template <class T>
  T acquire(ANARIObject h);
  template <class T>
  T handle_cast(ParameterBase &h);
  template <class T>
  T acquire(ParameterBase &h);

  virtual void accept(ObjectVisitorBase *) = 0;
  virtual void traverse(ObjectVisitorBase *) = 0;
  virtual uint64_t objectEpoch() const = 0;

  virtual void update() = 0;
};

template <>
struct base_flags<ObjectBase>
{
  static const uint32_t value = 0u;
};

class ArrayObjectBase : public ObjectBase
{
 public:
  ArrayObjectBase(ANARIDevice d, ANARIObject handle) : ObjectBase(d, handle) {}
  virtual void *map() = 0;
  virtual void unmap() = 0;
  virtual ANARIDataType getElementType() const = 0;
  virtual uint64_t size() const = 0;
  virtual int dims(uint64_t *) const = 0;
};

template <>
struct base_flags<ArrayObjectBase>
{
  static const uint32_t value = 0x10000000u;
};

class ObjectArray1D : public ArrayObjectBase
{
 protected:
  ObjectArray1D(ANARIDevice d, ANARIObject handle) : ArrayObjectBase(d, handle)
  {}
};

template <>
struct base_flags<ObjectArray1D>
{
  static const uint32_t value =
      0x00200000u | base_flags<ArrayObjectBase>::value;
};

class DataArray1D : public ArrayObjectBase
{
 protected:
  DataArray1D(ANARIDevice d, ANARIObject handle) : ArrayObjectBase(d, handle) {}

 public:
  virtual void declare(int index, AppendableShader &shader) = 0;
  virtual void sample(int index, AppendableShader &shader) = 0;
  virtual void drawCommand(int index, DrawCommand &command) = 0;
  virtual GLuint getTexture1D() = 0;
  virtual GLuint getBuffer() = 0;
  virtual ANARIDataType getBufferType() const = 0;
  virtual std::array<float, 6> getBounds() = 0;
  virtual std::array<float, 4> at(uint64_t) const = 0;
};

template <>
struct base_flags<DataArray1D>
{
  static const uint32_t value =
      0x00400000u | base_flags<ArrayObjectBase>::value;
};

class FrameObjectBase : public ObjectBase
{
 public:
  FrameObjectBase(ANARIDevice d, ANARIObject handle) : ObjectBase(d, handle) {}
  virtual void *mapFrame(
      const char *, uint32_t *, uint32_t *, ANARIDataType *) = 0;
  virtual void unmapFrame(const char *) = 0;
  virtual void renderFrame() = 0;
  virtual void discardFrame() = 0;
  virtual int frameReady(ANARIWaitMask) = 0;
};

template <>
struct base_flags<FrameObjectBase>
{
  static const uint32_t value = 0x20000000u;
};

class SurfaceObjectBase : public ObjectBase
{
 public:
  SurfaceObjectBase(ANARIDevice d, ANARIObject handle) : ObjectBase(d, handle)
  {}
  virtual void drawCommand(DrawCommand &) = 0;
  virtual void allocateTexture(
      int slot, GLenum target, GLuint texture, GLuint sampler) = 0;
  virtual void allocateStorageBuffer(int slot, GLuint buffer) = 0;
  virtual void allocateTransform(int slot) = 0;
  virtual int resourceIndex(int slot) = 0;
  virtual void addAttributeFlags(int attrib, uint32_t flags) = 0;
  virtual uint32_t getAttributeFlags(int attrib) = 0;
};

template <>
struct base_flags<SurfaceObjectBase>
{
  static const uint32_t value = 0x08000000u;
};

class GeometryObjectBase : public ObjectBase
{
 public:
  GeometryObjectBase(ANARIDevice d, ANARIObject handle) : ObjectBase(d, handle)
  {}
  virtual void allocateResources(SurfaceObjectBase *) = 0;
  virtual void drawCommand(SurfaceObjectBase *, DrawCommand &) = 0;
  virtual void vertexShader(SurfaceObjectBase *, AppendableShader &) = 0;
  virtual void fragmentShaderMain(SurfaceObjectBase *, AppendableShader &) = 0;

  virtual void vertexShaderShadow(SurfaceObjectBase *, AppendableShader &) = 0;
  virtual void geometryShaderShadow(
      SurfaceObjectBase *, AppendableShader &) = 0;
  virtual void fragmentShaderShadowMain(
      SurfaceObjectBase *, AppendableShader &) = 0;

  virtual void vertexShaderOcclusion(
      SurfaceObjectBase *, AppendableShader &) = 0;
  virtual std::array<float, 6> bounds() = 0;
  virtual uint32_t index() = 0;
};

template <>
struct base_flags<GeometryObjectBase>
{
  static const uint32_t value = 0x40000000u;
};

class MaterialObjectBase : public ObjectBase
{
 public:
  MaterialObjectBase(ANARIDevice d, ANARIObject handle) : ObjectBase(d, handle)
  {}
  virtual void allocateResources(SurfaceObjectBase *) = 0;
  virtual void drawCommand(SurfaceObjectBase *, DrawCommand &) = 0;
  virtual void fragmentShaderDeclarations(
      SurfaceObjectBase *, AppendableShader &) = 0;
  virtual void fragmentShaderMain(SurfaceObjectBase *, AppendableShader &) = 0;

  virtual void fragmentShaderShadowDeclarations(
      SurfaceObjectBase *, AppendableShader &) = 0;
  virtual void fragmentShaderShadowMain(
      SurfaceObjectBase *, AppendableShader &) = 0;
  virtual uint32_t index() = 0;
};

template <>
struct base_flags<MaterialObjectBase>
{
  static const uint32_t value = 0x80000000u;
};

class CameraObjectBase : public ObjectBase
{
 public:
  CameraObjectBase(ANARIDevice d, ANARIObject handle) : ObjectBase(d, handle) {}
  virtual void updateAt(size_t, float *) const = 0;
};

template <>
struct base_flags<CameraObjectBase>
{
  static const uint32_t value = 0x01000000u;
};

class LightObjectBase : public ObjectBase
{
 public:
  LightObjectBase(ANARIDevice d, ANARIObject handle) : ObjectBase(d, handle) {}
  virtual uint32_t index() = 0;
  virtual uint32_t lightType() = 0;
};

template <>
struct base_flags<LightObjectBase>
{
  static const uint32_t value = 0x02000000u;
};

class InstanceObjectBase : public ObjectBase
{
 public:
  InstanceObjectBase(ANARIDevice d, ANARIObject handle) : ObjectBase(d, handle)
  {}
  virtual const std::array<float, 16> &transform() = 0;
  virtual uint32_t index() = 0;
};

template <>
struct base_flags<InstanceObjectBase>
{
  static const uint32_t value = 0x04000000u;
};

class SamplerObjectBase : public ObjectBase
{
 public:
  SamplerObjectBase(ANARIDevice d, ANARIObject handle) : ObjectBase(d, handle)
  {}
  virtual void allocateResources(SurfaceObjectBase *, int) = 0;
  virtual void drawCommand(int, DrawCommand &) = 0;
  virtual void declare(int, AppendableShader &) = 0;
  virtual void sample(int, AppendableShader &, const char *) = 0;
  virtual std::array<uint32_t, 4> metadata() = 0;
};

template <>
struct base_flags<SamplerObjectBase>
{
  static const uint32_t value = 0x00100000u;
};

class VolumeObjectBase : public ObjectBase
{
 public:
  VolumeObjectBase(ANARIDevice d, ANARIObject handle) : ObjectBase(d, handle) {}
  virtual void drawCommand(DrawCommand &) = 0;
  virtual uint32_t index() = 0;
};

template <>
struct base_flags<VolumeObjectBase>
{
  static const uint32_t value = 0x00020000u;
};

class SpatialFieldObjectBase : public ObjectBase
{
 public:
  SpatialFieldObjectBase(ANARIDevice d, ANARIObject handle)
      : ObjectBase(d, handle)
  {}
  virtual void drawCommand(VolumeObjectBase *, DrawCommand &) = 0;
  virtual void fragmentShaderMain(VolumeObjectBase *, AppendableShader &) = 0;
  virtual void vertexShaderMain(VolumeObjectBase *, AppendableShader &) = 0;
  virtual uint32_t index() = 0;
  virtual std::array<float, 6> bounds() = 0;
};

template <>
struct base_flags<SpatialFieldObjectBase>
{
  static const uint32_t value = 0x00010000u;
};

class ObjectVisitorBase
{
 public:
  virtual void visit(ObjectBase *o)
  {
    o->traverse(this);
  }
  // forward to the base version to act as a catchall in case these are not
  // overriden
  virtual void visit(ArrayObjectBase *o)
  {
    visit(static_cast<ObjectBase *>(o));
  }
  virtual void visit(FrameObjectBase *o)
  {
    visit(static_cast<ObjectBase *>(o));
  }
  virtual void visit(GeometryObjectBase *o)
  {
    visit(static_cast<ObjectBase *>(o));
  }
  virtual void visit(MaterialObjectBase *o)
  {
    visit(static_cast<ObjectBase *>(o));
  }
  virtual void visit(CameraObjectBase *o)
  {
    visit(static_cast<ObjectBase *>(o));
  }
  virtual void visit(LightObjectBase *o)
  {
    visit(static_cast<ObjectBase *>(o));
  }
  virtual void visit(InstanceObjectBase *o)
  {
    visit(static_cast<ObjectBase *>(o));
  }
  virtual void visit(SurfaceObjectBase *o)
  {
    visit(static_cast<ObjectBase *>(o));
  }
  virtual void visit(SamplerObjectBase *o)
  {
    visit(static_cast<ObjectBase *>(o));
  }
  virtual void visit(SpatialFieldObjectBase *o)
  {
    visit(static_cast<ObjectBase *>(o));
  }
  virtual void visit(VolumeObjectBase *o)
  {
    visit(static_cast<ObjectBase *>(o));
  }
};

class Device;
template <class T>
class Object;

uint64_t anariIncrementEpoch(Object<Device> *, ObjectBase *);

template <class T, class B = ObjectBase>
class DefaultObject : public B
{
 protected:
  T staging;
  uint64_t lastEpoch;
  DefaultObject(ANARIDevice d, ANARIObject handle)
      : B(d, handle),
        staging(d, handle),
        current(d, handle),
        thisDevice(B::template acquire<Object<Device> *>(d))
  {
    lastEpoch = anariIncrementEpoch(thisDevice, this);
  }

  DefaultObject(ANARIDevice d, Object<Device> *deviceObj)
      : B(d, d), staging(d, d), current(d, d), thisDevice(deviceObj)
  {
    lastEpoch = anariIncrementEpoch(thisDevice, this);
  }

 public:
  T current;
  Object<Device> *const thisDevice;
  static const uint32_t ID = T::id | base_flags<B>::value;

  bool set(const char *paramname, ANARIDataType type, const void *mem) override
  {
    return staging.set(paramname, type, mem);
  }
  void unset(const char *paramname) override
  {
    staging.unset(paramname);
  }
  void unsetAll() override
  {
    for (size_t i = 0; i < staging.paramCount(); i++)
      staging[i].unset(staging.device, staging.object);
  }
 void* mapParameter1D(const char *paramname,
    ANARIDataType type,
    uint64_t numElements1,
    uint64_t *stride) override
  {
    ParameterBase &param = staging[paramname];
    if(param.accepts(ANARI_ARRAY1D)) {
      ANARIArray1D a = anariNewArray1D(B::device, nullptr, nullptr, nullptr, type, numElements1);
      param.set(B::device, B::handle, ANARI_ARRAY1D, &a);
      *stride = anari::sizeOf(type);
      return anariMapArray(B::device, a);
    } else {
      return nullptr;
    }
  }
  void* mapParameter2D(const char *paramname,
    ANARIDataType type,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t *stride) override
  {
    ParameterBase &param = staging[paramname];
    if(param.accepts(ANARI_ARRAY2D)) {
      ANARIArray2D a = anariNewArray2D(B::device, nullptr, nullptr, nullptr, type, numElements1, numElements2);
      param.set(B::device, B::handle, ANARI_ARRAY2D, &a);
      *stride = anari::sizeOf(type);
      return anariMapArray(B::device, a);
    } else {
      return nullptr;
    }
  }
  void* mapParameter3D(const char *paramname,
    ANARIDataType type,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t numElements3,
    uint64_t *stride) override
  {
    ParameterBase &param = staging[paramname];
    if(param.accepts(ANARI_ARRAY3D)) {
      ANARIArray3D a = anariNewArray3D(B::device, nullptr, nullptr, nullptr, type, numElements1, numElements2, numElements3);
      param.set(B::device, B::handle, ANARI_ARRAY3D, &a);
      *stride = anari::sizeOf(type);
      return anariMapArray(B::device, a);
    } else {
      return nullptr;
    }
  }
  void unmapParameter(const char *paramname) override
  {
    if(auto a = staging[paramname].getHandle()) {
      anariUnmapArray(B::device, static_cast<ANARIArray>(a));
      anariRelease(B::device, a);
    }
  }
  void commit() override
  {
    current = staging;
    lastEpoch = anariIncrementEpoch(thisDevice, this);
  }
  int getProperty(const char *propname,
      ANARIDataType type,
      void *mem,
      uint64_t size,
      ANARIWaitMask mask) override
  {
    (void)propname;
    (void)type;
    (void)mem;
    (void)size;
    (void)mask;
    return 0;
  }

  ANARIDataType type() const final
  {
    return T::type;
  }
  const char *subtype() const final
  {
    return T::subtype;
  }
  uint32_t id() const final
  {
    return T::id | base_flags<B>::value;
  }
  ParameterPack &parameters() override
  {
    return current;
  }
  void update() override {}
  void accept(ObjectVisitorBase *visitor) override
  {
    visitor->visit(this);
  }
  void traverse(ObjectVisitorBase *visitor) override
  {
    size_t count = current.paramCount();
    for (size_t i = 0; i < count; ++i) {
      if (auto child = B::template handle_cast<ObjectBase *>(current[i])) {
        child->accept(visitor);
      }
    }
  }
  uint64_t objectEpoch() const override
  {
    return lastEpoch;
  }
};

// this can be specialized in case we want to further subdivide
// object creation based on arguments
template <typename T>
struct ObjectAllocator
{
  template <typename... ARGS>
  static Object<T> *allocate(ARGS... args)
  {
    return new Object<T>(args...);
  }
};

template <class T>
class Object : public DefaultObject<T>
{
 public:
  Object(ANARIDevice d, ANARIObject handle) : DefaultObject<T>(d, handle) {}
};

template <class T>
struct is_convertible
{
  static bool check(ObjectBase *base)
  {
    uint32_t mask = base_flags<T>::value;
    return (base->id() & mask) == mask;
  }
};

template <class T>
struct is_convertible<Object<T>>
{
  static bool check(ObjectBase *base)
  {
    return (base->id() & 0x0000FFFFu) == T::id;
  }
};

void anariRetainInternal(ANARIDevice, ANARIObject, ANARIObject);
void anariReleaseInternal(ANARIDevice, ANARIObject, ANARIObject);

template <class T>
class ObjectRef
{
  using P = typename std::
      conditional<std::is_base_of<ParameterPack, T>::value, Object<T>, T>::type;

  P *ptr = nullptr;
  ANARIDevice device = nullptr;
  ANARIObject handle = nullptr;
  void acquire(P *p)
  {
    if (p) {
      device = p->device;
      handle = p->handle;
      p->retainInternal(device);
    }
  }
  void release()
  {
    if (ptr) {
      anariReleaseInternal(device, handle, device);
    }
  }

 public:
  ObjectRef() = default;

  ObjectRef(P *obj) : ptr(obj)
  {
    acquire(ptr);
  }
  ObjectRef(const ObjectRef<T> &that) : ptr(that.ptr)
  {
    acquire(ptr);
  }
  ObjectRef(ObjectRef<T> &&that)
  {
    std::swap(ptr, that.ptr);
    std::swap(device, that.device);
    std::swap(handle, that.handle);
  }
  ObjectRef &operator=(const ObjectRef<T> &that)
  {
    acquire(that.ptr);
    release();
    ptr = that.ptr;
    return *this;
  }
  ObjectRef &operator=(ObjectRef<T> &&that)
  {
    std::swap(ptr, that.ptr);
    std::swap(device, that.device);
    std::swap(handle, that.handle);
    return *this;
  }
  P *operator->()
  {
    return ptr;
  }
  P &operator*()
  {
    return *ptr;
  }
  const P *operator->() const
  {
    return ptr;
  }
  const P &operator*() const
  {
    return *ptr;
  }
  operator bool() const
  {
    return ptr;
  }
  P *get()
  {
    return ptr;
  }
  const P *get() const
  {
    return ptr;
  }
  ~ObjectRef()
  {
    release();
  }
};

template <typename T, typename T2>
bool operator==(const ObjectRef<T> &a, const T2 *b)
{
  return a.get() == b;
}
template <typename T, typename T2>
bool operator==(const T2 *a, const ObjectRef<T> &b)
{
  return a == b.get();
}
template <typename T>
bool operator==(const ObjectRef<T> &a, const ObjectRef<T> &b)
{
  return a.get() == b.get();
}

} // namespace visgl

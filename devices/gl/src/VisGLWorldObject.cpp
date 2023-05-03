// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGLSpecializations.h"
#include "math_util.h"

namespace visgl{

Object<World>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{

}

class BoundsVisitor : public ObjectVisitorBase {
  InstanceObjectBase *instance = 0;
public:
  std::array<float, 6> world_bounds{FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX};

  void visit(InstanceObjectBase *obj) override {
    obj->update();
    instance = obj;
    obj->traverse(this);
    instance = 0;
  }

  void visit(GeometryObjectBase *obj) override {
    obj->update();
    auto bounds = obj->bounds();
    if(instance) {
      transformBoundingBox(bounds.data(), instance->transform().data(), bounds.data());
    }
    foldBoundingBox(world_bounds.data(), bounds.data());
  }

  void visit(SpatialFieldObjectBase *obj) override {
    obj->update();
    auto bounds = obj->bounds();
    if(instance) {
      transformBoundingBox(bounds.data(), instance->transform().data(), bounds.data());
    }
    foldBoundingBox(world_bounds.data(), bounds.data());
  }
};

int Object<World>::getProperty(const char *propname,
  ANARIDataType type,
  void *mem,
  uint64_t size,
  ANARIWaitMask mask)
{
  if(type == ANARI_FLOAT32_BOX3 && size >= 6*sizeof(float) && std::strncmp("bounds", propname, 6) == 0) {
    BoundsVisitor bounds;
    this->accept(&bounds);
    std::memcpy(mem, bounds.world_bounds.data(), 6*sizeof(float));
    return 1;
  }
  return 0;
}

void world_free_objects(Object<Device> *deviceObj,
  GLuint occlusionbuffer)
{
  auto &gl = deviceObj->gl;
  gl.DeleteBuffers(1, &occlusionbuffer);
}

Object<World>::~Object() {
  thisDevice->queue.enqueue(world_free_objects, thisDevice,
    occlusionbuffer);
}

} //namespace visgl


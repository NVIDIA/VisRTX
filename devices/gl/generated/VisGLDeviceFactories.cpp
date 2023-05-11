// Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// This file was generated by generate_device_frontend.py
// Don't make changes to this directly

#include "VisGLDevice.h"
#include "VisGLObjects.h"
#include "VisGLSpecializations.h"
namespace visgl{
static int obj_hash(const char *str) {
   static const uint32_t table[] = {0x7a6f0012u,0x6a65002bu,0x0u,0x0u,0x0u,0x0u,0x6e6d0040u,0x0u,0x0u,0x0u,0x6261004du,0x0u,0x736d0052u,0x73650071u,0x767500a3u,0x0u,0x756300a7u,0x737200dcu,0x6f6e001du,0x0u,0x0u,0x0u,0x0u,0x0u,0x73720020u,0x0u,0x0u,0x0u,0x6d6c0024u,0x6665001eu,0x100001fu,0x80000000u,0x77760021u,0x66650022u,0x1000023u,0x80000001u,0x6a690025u,0x6f6e0026u,0x65640027u,0x66650028u,0x73720029u,0x100002au,0x80000002u,0x67660030u,0x0u,0x0u,0x0u,0x73720036u,0x62610031u,0x76750032u,0x6d6c0033u,0x75740034u,0x1000035u,0x80000003u,0x66650037u,0x64630038u,0x75740039u,0x6a69003au,0x706f003bu,0x6f6e003cu,0x6261003du,0x6d6c003eu,0x100003fu,0x80000004u,0x62610041u,0x68670042u,0x66650043u,0x34310044u,0x45440047u,0x45440049u,0x4544004bu,0x1000048u,0x80000005u,0x100004au,0x80000006u,0x100004cu,0x80000007u,0x7574004eu,0x7574004fu,0x66650050u,0x1000051u,0x80000008u,0x6f6e0058u,0x0u,0x0u,0x0u,0x0u,0x75740066u,0x6a690059u,0x6564005au,0x6a69005bu,0x7372005cu,0x6665005du,0x6463005eu,0x7574005fu,0x6a690060u,0x706f0061u,0x6f6e0062u,0x62610063u,0x6d6c0064u,0x1000065u,0x80000009u,0x69680067u,0x706f0068u,0x68670069u,0x7372006au,0x6261006bu,0x7170006cu,0x6968006du,0x6a69006eu,0x6463006fu,0x1000070u,0x8000000au,0x7372007fu,0x0u,0x0u,0x7a790089u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x6a690097u,0x0u,0x0u,0x6a69009bu,0x74730080u,0x71700081u,0x66650082u,0x64630083u,0x75740084u,0x6a690085u,0x77760086u,0x66650087u,0x1000088u,0x8000000bu,0x7473008au,0x6a69008bu,0x6463008cu,0x6261008du,0x6d6c008eu,0x6d6c008fu,0x7a790090u,0x43420091u,0x62610092u,0x74730093u,0x66650094u,0x65640095u,0x1000096u,0x8000000cu,0x6f6e0098u,0x75740099u,0x100009au,0x8000000du,0x6e6d009cu,0x6a69009du,0x7574009eu,0x6a69009fu,0x777600a0u,0x666500a1u,0x10000a2u,0x8000000eu,0x626100a4u,0x656400a5u,0x10000a6u,0x8000000fu,0x6a6900b9u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x706800beu,0x0u,0x0u,0x0u,0x737200ccu,0x777600bau,0x6a6900bbu,0x747300bcu,0x10000bdu,0x80000010u,0x666500c6u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x757400cau,0x737200c7u,0x666500c8u,0x10000c9u,0x80000011u,0x10000cbu,0x80000012u,0x767500cdu,0x646300ceu,0x757400cfu,0x767500d0u,0x737200d1u,0x666500d2u,0x656400d3u,0x535200d4u,0x666500d5u,0x686700d6u,0x767500d7u,0x6d6c00d8u,0x626100d9u,0x737200dau,0x10000dbu,0x80000013u,0x6a6100ddu,0x6f6e00e6u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x62610102u,0x747300e7u,0x716600e8u,0x706f00f3u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x0u,0x626100f7u,0x737200f4u,0x6e6d00f5u,0x10000f6u,0x80000014u,0x737200f8u,0x666500f9u,0x6f6e00fau,0x757400fbu,0x4e4d00fcu,0x626100fdu,0x757400feu,0x757400ffu,0x66650100u,0x1000101u,0x80000015u,0x6f6e0103u,0x68670104u,0x6d6c0105u,0x66650106u,0x1000107u,0x80000016u};
   uint32_t cur = 0x75630000u;
   for(int i = 0;cur!=0;++i) {
      uint32_t idx = cur&0xFFFFu;
      uint32_t low = (cur>>16u)&0xFFu;
      uint32_t high = (cur>>24u)&0xFFu;
      uint32_t c = (uint32_t)str[i];
      if(c>=low && c<high) {
         cur = table[idx+c-low];
      } else {
         break;
      }
      if(cur&0x80000000u) {
         return cur&0xFFFFu;
      }
      if(str[i]==0) {
         break;
      }
   }
   return -1;
}
ANARIArray1D VisGLDevice::newArray1D(const void* appMemory, ANARIMemoryDeleter deleter, const void* userdata, ANARIDataType type, uint64_t numItems1, uint64_t byteStride1) {
   return allocate<ANARIArray1D, Array1D>(appMemory, deleter, userdata, type, numItems1, byteStride1);
}
ANARIArray2D VisGLDevice::newArray2D(const void* appMemory, ANARIMemoryDeleter deleter, const void* userdata, ANARIDataType type, uint64_t numItems1, uint64_t numItems2, uint64_t byteStride1, uint64_t byteStride2) {
   return allocate<ANARIArray2D, Array2D>(appMemory, deleter, userdata, type, numItems1, numItems2, byteStride1, byteStride2);
}
ANARIArray3D VisGLDevice::newArray3D(const void* appMemory, ANARIMemoryDeleter deleter, const void* userdata, ANARIDataType type, uint64_t numItems1, uint64_t numItems2, uint64_t numItems3, uint64_t byteStride1, uint64_t byteStride2, uint64_t byteStride3) {
   return allocate<ANARIArray3D, Array3D>(appMemory, deleter, userdata, type, numItems1, numItems2, numItems3, byteStride1, byteStride2, byteStride3);
}
ANARIFrame VisGLDevice::newFrame() {
   return allocate<ANARIFrame, Frame>();
}
ANARIGroup VisGLDevice::newGroup() {
   return allocate<ANARIGroup, Group>();
}
ANARIInstance VisGLDevice::newInstance() {
   return allocate<ANARIInstance, Instance>();
}
ANARIWorld VisGLDevice::newWorld() {
   return allocate<ANARIWorld, World>();
}
ANARIRenderer VisGLDevice::newRenderer(const char *type) {
   int idx = obj_hash(type);
   switch(idx) {
      case 3: //default
         return allocate<ANARIRenderer, RendererDefault>();
      default: // unknown object
         return 0;
   }
   return 0;
}
ANARISurface VisGLDevice::newSurface() {
   return allocate<ANARISurface, Surface>();
}
ANARICamera VisGLDevice::newCamera(const char *type) {
   int idx = obj_hash(type);
   switch(idx) {
      case 9: //omnidirectional
         return allocate<ANARICamera, CameraOmnidirectional>();
      case 10: //orthographic
         return allocate<ANARICamera, CameraOrthographic>();
      case 11: //perspective
         return allocate<ANARICamera, CameraPerspective>();
      default: // unknown object
         return 0;
   }
   return 0;
}
ANARIGeometry VisGLDevice::newGeometry(const char *type) {
   int idx = obj_hash(type);
   switch(idx) {
      case 0: //cone
         return allocate<ANARIGeometry, GeometryCone>();
      case 1: //curve
         return allocate<ANARIGeometry, GeometryCurve>();
      case 2: //cylinder
         return allocate<ANARIGeometry, GeometryCylinder>();
      case 15: //quad
         return allocate<ANARIGeometry, GeometryQuad>();
      case 17: //sphere
         return allocate<ANARIGeometry, GeometrySphere>();
      case 22: //triangle
         return allocate<ANARIGeometry, GeometryTriangle>();
      default: // unknown object
         return 0;
   }
   return 0;
}
ANARILight VisGLDevice::newLight(const char *type) {
   int idx = obj_hash(type);
   switch(idx) {
      case 4: //directional
         return allocate<ANARILight, LightDirectional>();
      case 13: //point
         return allocate<ANARILight, LightPoint>();
      case 18: //spot
         return allocate<ANARILight, LightSpot>();
      default: // unknown object
         return 0;
   }
   return 0;
}
ANARIMaterial VisGLDevice::newMaterial(const char *type) {
   int idx = obj_hash(type);
   switch(idx) {
      case 8: //matte
         return allocate<ANARIMaterial, MaterialMatte>();
      case 21: //transparentMatte
         return allocate<ANARIMaterial, MaterialTransparentMatte>();
      case 12: //physicallyBased
         return allocate<ANARIMaterial, MaterialPhysicallyBased>();
      default: // unknown object
         return 0;
   }
   return 0;
}
ANARISampler VisGLDevice::newSampler(const char *type) {
   int idx = obj_hash(type);
   switch(idx) {
      case 5: //image1D
         return allocate<ANARISampler, SamplerImage1D>();
      case 6: //image2D
         return allocate<ANARISampler, SamplerImage2D>();
      case 7: //image3D
         return allocate<ANARISampler, SamplerImage3D>();
      case 14: //primitive
         return allocate<ANARISampler, SamplerPrimitive>();
      case 20: //transform
         return allocate<ANARISampler, SamplerTransform>();
      default: // unknown object
         return 0;
   }
   return 0;
}
ANARISpatialField VisGLDevice::newSpatialField(const char *type) {
   int idx = obj_hash(type);
   switch(idx) {
      case 19: //structuredRegular
         return allocate<ANARISpatialField, Spatial_FieldStructuredRegular>();
      default: // unknown object
         return 0;
   }
   return 0;
}
ANARIVolume VisGLDevice::newVolume(const char *type) {
   int idx = obj_hash(type);
   switch(idx) {
      case 16: //scivis
         return allocate<ANARIVolume, VolumeScivis>();
      default: // unknown object
         return 0;
   }
   return 0;
}
} //namespace visgl
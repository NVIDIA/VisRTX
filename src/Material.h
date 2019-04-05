/*
* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#pragma warning( push )
#pragma warning( disable : 4250 ) // C4250 - 'class1' : inherits 'class2::member' via dominance

#include "VisRTX.h"
#include "OptiX.h"
#include "Object.h"
#include "Pathtracer/Common.h"

#include "MDL/MDL.h"

#include <queue>
#include <vector>
#include <set>
#include <algorithm>
#include <cassert>


namespace VisRTX
{
    namespace Impl
    {
        /*
         * Contiguous pool template for material parameter buffers
         */
        template<typename T>
        class ContiguousPool
        {
        public:
            ContiguousPool(uint32_t initialCapacity = 512)
            {
                uint32_t capacity = std::max(1u, initialCapacity);
                this->data.resize(capacity);

                for (uint32_t i = 0; i < capacity; ++i)
                    this->freeElements.push(i);
            }

            T& operator[](uint32_t index)
            {
                assert(index < this->data.size());
                return this->data[index];
            }

            const T* GetData() const
            {
                return this->data.data();
            }

            uint32_t GetNumBytes() const
            {
                return this->numElements * sizeof(T);
            }

            uint32_t GetNumElements() const
            {
                return this->numElements;
            }

            uint32_t GetElementSize() const
            {
                return sizeof(T);
            }

            uint32_t Allocate()
            {
                if (this->numElements >= this->data.size())
                {
                    size_t newCapacity = 2 * this->data.size();
                    this->data.resize(newCapacity);

                    for (uint32_t i = this->numElements; i < newCapacity; ++i)
                        this->freeElements.push(i);
                }

                uint32_t index = this->freeElements.top();
                this->freeElements.pop();

                this->usedElements.insert(index);
                this->numElements = this->usedElements.empty() ? 0 : *this->usedElements.rbegin() + 1;

                return index;
            }

            void Release(uint32_t index)
            {
                assert(index < this->numElements);
                this->freeElements.push(index);

                this->usedElements.erase(index);
                this->numElements = this->usedElements.empty() ? 0 : *this->usedElements.rbegin() + 1;
            }

        private:
            std::vector<T> data;
            std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> freeElements;
            std::set<uint32_t> usedElements;
            uint32_t numElements = 0;
        };


        /*
         * Base class
         */
        class Material : public virtual Object, public virtual VisRTX::Material
        {
            friend class Geometry;
            friend class TriangleGeometry;
            friend class SphereGeometry;
            friend class CylinderGeometry;

        public:
            Material() = default;
            virtual ~Material() = default;

        protected:
            MaterialId material = MATERIAL_NULL;
        };


        /*
         * Basic material
         */
        class BasicMaterial : public VisRTX::BasicMaterial, public Material
        {
            friend class Renderer;

        public:
            BasicMaterial();
            virtual ~BasicMaterial();

        public:
            void SetDiffuse(const Vec3f& Kd) override;
            void SetDiffuseTexture(VisRTX::Texture* diffuseTex) override;

            void SetSpecular(const Vec3f& Ks) override;
            void SetSpecularTexture(VisRTX::Texture* specularTex) override;
            void SetShininess(float Ns) override;
            void SetShininessTexture(VisRTX::Texture* shininessTex) override;

            void SetEmissive(const Vec3f& Ke) override;
            void SetEmissiveTexture(VisRTX::Texture* emissiveTex) override;
            void SetLuminosity(float luminosity) override;

            void SetOpacity(float opacity) override;
            void SetOpacityTexture(VisRTX::Texture* opacityTex) override;
            
            void SetTransparencyFilter(const Vec3f& Tf) override;
            
            void SetBumpMapTexture(VisRTX::Texture* bumpMapTex) override;

        private:
            static ContiguousPool<BasicMaterialParameters> parameters;
            static bool parametersDirty;

        private:
            uint32_t index;

            VisRTX::Texture* diffuseTex = nullptr;
            VisRTX::Texture* specularTex = nullptr;
            VisRTX::Texture* shininessTex = nullptr;
            VisRTX::Texture* emissiveTex = nullptr;
            VisRTX::Texture* opacityTex = nullptr;
            VisRTX::Texture* bumpMapTex = nullptr;
        };


        /*
         * MDL material
         */
        class MDLMaterial : public VisRTX::MDLMaterial, public Material
        {
            friend class Renderer;
        public:
            MDLMaterial(CompilationType compilationType = CompilationType::CLASS);
            MDLMaterial(const char* material, const char* source, uint32_t sourceBytes, uint32_t numModulePaths, const char** modulePaths, CompilationType compilationType);
            virtual ~MDLMaterial();

        public:
            void Load(const char* material, const char* source = nullptr, uint32_t sourceBytes = 0, uint32_t numModulePaths = 0, const char** modulePaths = nullptr) override;

            void Compile() override;

            virtual const char* GetName() const override;

            virtual CompilationType GetCompilationType() const override;

            virtual uint32_t GetParameterCount() override;
            virtual const char* GetParameterName(uint32_t i) override;

            virtual ParameterType GetParameterType(const char* name) override;

            virtual const char* GetParameterDescription(const char* name) override;

            virtual bool GetParameterBool(const char* name) override;
            virtual bool SetParameterBool(const char* name, bool value) override;

            virtual int GetParameterInt(const char* name) override;
            virtual bool SetParameterInt(const char* name, int value) override;

            virtual float GetParameterFloat(const char* name) override;
            virtual bool SetParameterFloat(const char* name, float value) override;

            virtual double GetParameterDouble(const char* name) override;
            virtual bool SetParameterDouble(const char* name, double value) override;

            virtual Vec3f GetParameterColor(const char* name) override;
            virtual bool SetParameterColor(const char* name, const Vec3f &value) override;

            virtual void GetParameterValueOther(const char* name, void *data, size_t size);
            virtual bool SetParameterValueOther(const char* name, const void *data, size_t size);

            virtual bool SetParameterTexture(const char* name, VisRTX::Texture* texture) override;

        private:
            static ContiguousPool<MDLMaterialParameters> parameters;
            static bool parametersDirty;
            void UpdateArgumentBlock();

        private:
            MDL* mdl;

            uint32_t index;
            uint32_t materialCandidate;
           
            MDL::Material loadedMaterial;
            std::vector<std::string> availableParameters;

            MDL::CompileArguments compileArguments;
            CompilationType compilationType;
            MDL::CompiledMaterial compiledMaterial;   
                                    
            std::set<uint32_t> freeTextureSlots;
            std::map<std::string, uint32_t> usedTextureSlots;

            std::map<std::string, VisRTX::Texture*> textureHandles;
        };
    }
}

#pragma warning( pop ) // C4250
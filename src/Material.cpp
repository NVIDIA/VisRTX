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

#include "Material.h"
#include "Texture.h"
#include "ProgramLoader.h"
#include "OptiX.h"

#include "Pathtracer/Common.h"


namespace VisRTX
{
    namespace Impl
    {
        // ------------------------------------------------------------------------------------
        // Basic material
        // ------------------------------------------------------------------------------------
        ContiguousPool<BasicMaterialParameters> BasicMaterial::parameters;
        bool BasicMaterial::parametersDirty = true;


        BasicMaterial::BasicMaterial()
        {
            // Acquire element in parameters buffer
            this->index = BasicMaterial::parameters.Allocate();
            this->material = (BASIC_MATERIAL_BIT | this->index);
            BasicMaterial::parametersDirty = true;

            // Default material: White (not too bright)
            this->SetDiffuse(Vec3f(0.8f, 0.8f, 0.8f));
            this->SetSpecular(Vec3f(0.0f, 0.0f, 0.0f));
            this->SetShininess(10.0f);

            this->SetEmissive(Vec3f(0.0f, 0.0f, 0.0f));
            this->SetLuminosity(0.0f);

            this->SetDiffuseTexture(nullptr);
            this->SetSpecularTexture(nullptr);
            this->SetShininessTexture(nullptr);
            this->SetEmissiveTexture(nullptr);
            this->SetOpacityTexture(nullptr);
            this->SetBumpMapTexture(nullptr);

            this->SetOpacity(1.0f);
            this->SetTransparencyFilter(Vec3f(0.0f, 0.0f, 0.0f));
        }

        BasicMaterial::~BasicMaterial()
        {
            BasicMaterial::parameters.Release(this->index);
            BasicMaterial::parametersDirty = true;

            if (this->diffuseTex)
                this->diffuseTex->Release();

            if (this->specularTex)
                this->specularTex->Release();

            if (this->shininessTex)
                this->shininessTex->Release();

            if (this->emissiveTex)
                this->emissiveTex->Release();

            if (this->opacityTex)
                this->opacityTex->Release();

            if (this->bumpMapTex)
                this->bumpMapTex->Release();
        }

        void BasicMaterial::SetDiffuse(const Vec3f& Kd)
        {
            BasicMaterial::parameters[this->index].diffuseColor = optix::make_float3(Kd.r, Kd.g, Kd.b);
            BasicMaterial::parametersDirty = true;
        }

        void BasicMaterial::SetSpecular(const Vec3f& Ks)
        {
            BasicMaterial::parameters[this->index].specularColor = optix::make_float3(Ks.r, Ks.g, Ks.b);
            BasicMaterial::parametersDirty = true;
        }

        void BasicMaterial::SetShininess(float Ns)
        {
            BasicMaterial::parameters[this->index].specularExponent = Ns;
            BasicMaterial::parametersDirty = true;
        }

        void BasicMaterial::SetEmissive(const Vec3f& Ke)
        {
            BasicMaterial::parameters[this->index].emissiveColor = optix::make_float3(Ke.r, Ke.g, Ke.b);
            BasicMaterial::parametersDirty = true;
        }

        void BasicMaterial::SetLuminosity(float luminosity)
        {
            BasicMaterial::parameters[this->index].luminosity = luminosity;
            BasicMaterial::parametersDirty = true;
        }

        void BasicMaterial::SetOpacity(float opacity)
        {
            BasicMaterial::parameters[this->index].opacity = opacity;
            BasicMaterial::parametersDirty = true;
        }

        void BasicMaterial::SetTransparencyFilter(const Vec3f& Tf)
        {
            BasicMaterial::parameters[this->index].transparencyFilterColor = optix::make_float3(Tf.r, Tf.g, Tf.b);
            BasicMaterial::parametersDirty = true;
        }

        void BasicMaterial::SetDiffuseTexture(VisRTX::Texture* diffuseTex)
        {
            VisRTX::Impl::Texture* tex = dynamic_cast<VisRTX::Impl::Texture*>(diffuseTex);
            BasicMaterial::parameters[this->index].diffuseTexture = tex ? tex->sampler->getId() : RT_TEXTURE_ID_NULL;
            BasicMaterial::parametersDirty = true;

            if (this->diffuseTex)
                this->diffuseTex->Release();
            this->diffuseTex = diffuseTex;
            if (this->diffuseTex)
                this->diffuseTex->Retain();
        }

        void BasicMaterial::SetSpecularTexture(VisRTX::Texture* specularTex)
        {
            VisRTX::Impl::Texture* tex = dynamic_cast<VisRTX::Impl::Texture*>(specularTex);
            BasicMaterial::parameters[this->index].specularTexture = tex ? tex->sampler->getId() : RT_TEXTURE_ID_NULL;
            BasicMaterial::parametersDirty = true;

            if (this->specularTex)
                this->specularTex->Release();
            this->specularTex = specularTex;
            if (this->specularTex)
                this->specularTex->Retain();
        }

        void BasicMaterial::SetShininessTexture(VisRTX::Texture* shininessTex)
        {
            VisRTX::Impl::Texture* tex = dynamic_cast<VisRTX::Impl::Texture*>(shininessTex);
            BasicMaterial::parameters[this->index].specularExponentTexture = tex ? tex->sampler->getId() : RT_TEXTURE_ID_NULL;
            BasicMaterial::parametersDirty = true;

            if (this->shininessTex)
                this->shininessTex->Release();
            this->shininessTex = shininessTex;
            if (this->shininessTex)
                this->shininessTex->Retain();
        }

        void BasicMaterial::SetEmissiveTexture(VisRTX::Texture* emissiveTex)
        {
            VisRTX::Impl::Texture* tex = dynamic_cast<VisRTX::Impl::Texture*>(emissiveTex);
            BasicMaterial::parameters[this->index].emissiveTexture = tex ? tex->sampler->getId() : RT_TEXTURE_ID_NULL;
            BasicMaterial::parametersDirty = true;

            if (this->emissiveTex)
                this->emissiveTex->Release();
            this->emissiveTex = emissiveTex;
            if (this->emissiveTex)
                this->emissiveTex->Retain();
        }

        void BasicMaterial::SetOpacityTexture(VisRTX::Texture* opacityTex)
        {
            VisRTX::Impl::Texture* tex = dynamic_cast<VisRTX::Impl::Texture*>(opacityTex);
            BasicMaterial::parameters[this->index].opacityTexture = tex ? tex->sampler->getId() : RT_TEXTURE_ID_NULL;
            BasicMaterial::parametersDirty = true;

            if (this->opacityTex)
                this->opacityTex->Release();
            this->opacityTex = opacityTex;
            if (this->opacityTex)
                this->opacityTex->Retain();
        }

        void BasicMaterial::SetBumpMapTexture(VisRTX::Texture* bumpMapTex)
        {
            VisRTX::Impl::Texture* tex = dynamic_cast<VisRTX::Impl::Texture*>(bumpMapTex);
            BasicMaterial::parameters[this->index].bumpMapTexture = tex ? tex->sampler->getId() : RT_TEXTURE_ID_NULL;
            BasicMaterial::parametersDirty = true;

            if (this->bumpMapTex)
                this->bumpMapTex->Release();
            this->bumpMapTex = bumpMapTex;
            if (this->bumpMapTex)
                this->bumpMapTex->Retain();
        }


        // ------------------------------------------------------------------------------------
        // MDL material
        // ------------------------------------------------------------------------------------
        ContiguousPool<MDLMaterialParameters> MDLMaterial::parameters;
        bool MDLMaterial::parametersDirty = true;

        // Helper function to extract the module name from a fully-qualified material name.
        std::string get_module_name(const std::string& material_name)
        {
            std::size_t p = material_name.rfind("::");
            return material_name.substr(0, p);
        }


        // Helper function to extract the material name from a fully-qualified material name.
        std::string get_material_name(const std::string& material_name)
        {
            std::size_t p = material_name.rfind("::");
            if (p == std::string::npos)
                return material_name;
            return material_name.substr(p + 2, material_name.size() - p);
        }

        MDLMaterial::MDLMaterial(CompilationType compilationType) : compilationType(compilationType)
        {
            // Initial state: Not ready to use
            this->material = 0;

            // Acquire element in parameters buffer
            this->index = MDLMaterial::parameters.Allocate();
            this->materialCandidate = (MDL_MATERIAL_BIT | this->index);
            MDLMaterial::parametersDirty = true;

            static MDL* instance = nullptr;
            if (!instance)
            {
                try
                {
                    instance = new MDL(OptiXContext::Get(), ProgramLoader::Get().GetMDLTexturesPTX());
            }
                catch (optix::Exception& e)
                {
                    throw VisRTX::Exception(VisRTX::Error::MDL_ERROR, e.getErrorString().c_str());
                }
        }

            this->mdl = instance;
    }

        MDLMaterial::MDLMaterial(const char* material, const char* source, uint32_t sourceBytes, uint32_t numModulePaths, const char** modulePaths, CompilationType compilationType) : MDLMaterial(compilationType)
        {
            this->Load(material, source, sourceBytes, numModulePaths, modulePaths);
        }

        MDLMaterial::~MDLMaterial()
        {
            MDLMaterial::parameters.Release(this->index);
            MDLMaterial::parametersDirty = true;

            for (auto it : this->textureHandles)
                if (it.second)
                    it.second->Release();

            // Clean up OptiX stuff
            this->compiledMaterial.Destroy();
        }

        void MDLMaterial::UpdateArgumentBlock()
        {
            MDLMaterialParameters& params = MDLMaterial::parameters[this->index];
            memcpy(params.argBlock, this->compiledMaterial.argumentBlock, MDL_ARGUMENT_BLOCK_SIZE);
            MDLMaterial::parametersDirty = true;
        }

        void MDLMaterial::Load(const char* material, const char* source, uint32_t sourceBytes, uint32_t numModulePaths, const char** modulePaths)
        {
            // Module name
            std::string materialStr(material);
            const std::size_t p = materialStr.rfind("::");
            std::string moduleName = materialStr.substr(0, p);

            // Material name
            std::string materialName = (p == std::string::npos) ? materialStr : materialStr.substr(p + 2, materialStr.size() - p);

            // Add search paths
            for (uint32_t i = 0; i < numModulePaths; ++i)
                this->mdl->AddModulePath(std::string(modulePaths[i]));

            // Load material
            try
            {
                this->loadedMaterial = this->mdl->Load(materialName, moduleName, std::string(source, sourceBytes));
            }
            catch (optix::Exception& e)
            {
#ifdef MDL_VERBOSE_LOGGING
                std::cerr << e.getErrorString() << std::endl;
#endif
                throw VisRTX::Exception(VisRTX::Error::MDL_ERROR, e.getErrorString().c_str());
            }

            if (this->loadedMaterial.IsLoaded())
            {
                // Update list of available parameters
                this->availableParameters.clear();
                for (auto entry : this->loadedMaterial.parameters)
                {
                    const std::string& name = entry.first;

                    if (name.find(MDL_USE_TEXTURE_PREFIX) == 0)
                        continue;

                    this->availableParameters.push_back(name);
                }

                // Automatically compile if class compilation
                if (this->compilationType == CompilationType::CLASS)
                    this->Compile();
            }
        }

        void MDLMaterial::Compile()
        {
            // Class-compiled materials don't need to be recompiled
            if (this->compiledMaterial.IsCompiled() && this->compilationType == CompilationType::CLASS)
                return;

            // Compile
            try
            {
                // Clean up existing
                this->compiledMaterial.Destroy();

                // Compile new
                this->compiledMaterial = this->mdl->Compile(this->loadedMaterial, this->compilationType == CompilationType::CLASS, this->compileArguments);
            }
            catch (optix::Exception& e)
            {
#ifdef MDL_VERBOSE_LOGGING
                std::cerr << e.getErrorString() << std::endl;
#endif
                throw VisRTX::Exception(VisRTX::Error::MDL_ERROR, e.getErrorString().c_str());
            }

            if (this->compiledMaterial.IsCompiled())
            {
                // Assign callable programs
                MDLMaterialParameters& params = MDLMaterial::parameters[this->index];
                params.init = this->compiledMaterial.initProg->getId();
                params.sample = this->compiledMaterial.sampleProg->getId();
                params.evaluate = this->compiledMaterial.evaluateProg->getId();
                params.opacity = this->compiledMaterial.opacityProg->getId();

                params.hasArgBlock = this->compilationType == CompilationType::CLASS;
                if (this->compilationType == CompilationType::INSTANCE)
                {
                    params.hasArgBlock = 0;
                }
                else //this->compilationType == CompilationType::CLASS
                {
                    if (this->compiledMaterial.argumentBlock)
                    {
                        memcpy(params.argBlock, this->compiledMaterial.argumentBlock, MDL_ARGUMENT_BLOCK_SIZE);
                        params.hasArgBlock = 1;
                    }
                    else
                    {
                        params.hasArgBlock = 0;
                    }

                }
                this->compiledMaterial.pdfProg->getId(); // currently not used

                // Initialize set of free texture slots
                for (uint32_t i = this->compiledMaterial.numTextures; i < MDL_MAX_TEXTURES; ++i)
                    this->freeTextureSlots.insert(this->compiledMaterial.texturesOffset + i);

                MDLMaterial::parametersDirty = true;

                // Mark material as ready to use
                this->material = this->materialCandidate;
            }
        }

        const char* MDLMaterial::GetName() const
        {
            return this->loadedMaterial.material.c_str();
        }

        CompilationType MDLMaterial::GetCompilationType() const
        {
            return this->compilationType;
        }

        uint32_t MDLMaterial::GetParameterCount()
        {
            return (uint32_t)this->availableParameters.size();
        }

        const char* MDLMaterial::GetParameterName(uint32_t i)
        {
            return this->availableParameters[i].c_str();
        }

        ParameterType MDLMaterial::GetParameterType(const char* name)
        {
            auto entry = this->loadedMaterial.parameters.find(name);
            if (entry == this->loadedMaterial.parameters.end())
                return ParameterType::NONE;

            const MDL::Material::Parameter& info = entry->second;
            switch (info.kind)
            {
            case mi::neuraylib::IType::Kind::TK_BOOL:
                return ParameterType::BOOL;
            case mi::neuraylib::IType::Kind::TK_COLOR:
                return ParameterType::COLOR;
            case mi::neuraylib::IType::Kind::TK_DOUBLE:
                return ParameterType::DOUBLE;
            case mi::neuraylib::IType::Kind::TK_FLOAT:
                return ParameterType::FLOAT;
            case mi::neuraylib::IType::Kind::TK_INT:
                return ParameterType::INT;
            case mi::neuraylib::IType::Kind::TK_TEXTURE:
                return ParameterType::TEXTURE;
            }

            return ParameterType::NONE;
        }

        inline void *GetParameterPointer(MDL::CompiledMaterial& material, const char* name, size_t expected_size)
        {
            if (material.layout.empty())
                return nullptr;

            auto entry = material.layout.find(name);
            if (entry == material.layout.end())
            {
#ifdef MDL_VERBOSE_LOGGING
                std::cerr << "Warning: Failed to access material parameter \"" << name << "\" because it does not exist" << std::endl;
#endif
                return nullptr;
            }

            const MDL::CompiledMaterial::Layout& parameter_info = entry->second;
            if (expected_size != parameter_info.size)
            {
#ifdef MDL_VERBOSE_LOGGING
                std::cerr << "Warning: Failed to access material parameter \"" << name << "\" due to type size mismatch (expected: " << expected_size << "; found: " << parameter_info.size << ")" << std::endl;
#endif
                return nullptr;
            }

            if (parameter_info.offset + parameter_info.size > MDL_ARGUMENT_BLOCK_SIZE)
            {
#ifdef MDL_VERBOSE_LOGGING
                std::cerr << "Warning: Failed to access material parameter \"" << name << "\" because it would exceed the " + std::to_string(MDL_ARGUMENT_BLOCK_SIZE) + " byte argument block" << std::endl;
#endif
                return nullptr;
            }

            return (void*)(material.argumentBlock + parameter_info.offset);
        }

        void MDLMaterial::GetParameterValueOther(const char* name, void *data, size_t size)
        {
            if (!data) return;
            const void *ptr = GetParameterPointer(this->compiledMaterial, name, size);
            if (ptr) { memcpy(data, ptr, size); }
        }

        bool MDLMaterial::SetParameterValueOther(const char* name, const void *data, size_t size)
        {
            if (this->compilationType == CompilationType::CLASS)
            {
                if (!data) return false;
                void *ptr = GetParameterPointer(this->compiledMaterial, name, size);
                if (ptr) { memcpy(ptr, data, size); }
                return nullptr != ptr;
            }
            else
            {
                // TODO
                return false;
            }
        }

        template<typename T>
        inline T *GetParameter(MDL::CompiledMaterial& material, const char* name)
        {
            return (T*)GetParameterPointer(material, name, sizeof(T));
        }

        template<typename T>
        inline T GetDefault(MDL::Material& material, const char* name)
        {
            T result;

            auto it = material.parameters.find(name);
            if (it != material.parameters.end())
            {
                memcpy(&result, it->second.defaultValue, std::min(sizeof(T), (size_t)32));
            }

            return result;
        }

        bool MDLMaterial::GetParameterBool(const char* name)
        {
            if (this->compilationType == CompilationType::CLASS)
            {
                bool *ptr = GetParameter<bool>(this->compiledMaterial, name);
                return ptr ? *ptr : false;
            }
            else
            {
                auto it = this->compileArguments.bools.find(name);
                if (it != this->compileArguments.bools.end())
                    return it->second;

                return GetDefault<bool>(this->loadedMaterial, name);
            }
        }

        bool MDLMaterial::SetParameterBool(const char* name, bool value)
        {
            if (this->compilationType == CompilationType::CLASS)
            {
                bool *ptr = GetParameter<bool>(this->compiledMaterial, name);
                if (ptr) { *ptr = value; UpdateArgumentBlock(); }
                return nullptr != ptr;
            }
            else
            {
                this->compileArguments.bools[name] = value;
                return true;
            }
        }

        int MDLMaterial::GetParameterInt(const char* name)
        {
            if (this->compilationType == CompilationType::CLASS)
            {
                int *ptr = GetParameter<int>(this->compiledMaterial, name);
                return ptr ? *ptr : 0;
            }
            else
            {
                auto it = this->compileArguments.ints.find(name);
                if (it != this->compileArguments.ints.end())
                    return it->second;

                return GetDefault<int>(this->loadedMaterial, name);
            }
        }

        bool MDLMaterial::SetParameterInt(const char* name, int value)
        {
            if (this->compilationType == CompilationType::CLASS)
            {
                int *ptr = GetParameter<int>(this->compiledMaterial, name);
                if (ptr) { *ptr = value; UpdateArgumentBlock(); }
                return nullptr != ptr;
            }
            else
            {
                this->compileArguments.ints[name] = value;
                return true;
            }
        }

        float MDLMaterial::GetParameterFloat(const char* name)
        {
            if (this->compilationType == CompilationType::CLASS)
            {
                float *ptr = GetParameter<float>(this->compiledMaterial, name);
                return ptr ? *ptr : 0.0f;
            }
            else
            {
                auto it = this->compileArguments.floats.find(name);
                if (it != this->compileArguments.floats.end())
                    return it->second;

                return GetDefault<float>(this->loadedMaterial, name);
            }
        }

        bool MDLMaterial::SetParameterFloat(const char* name, float value)
        {
            if (this->compilationType == CompilationType::CLASS)
            {
                float *ptr = GetParameter<float>(this->compiledMaterial, name);
                if (ptr) { *ptr = value; UpdateArgumentBlock(); }
                return nullptr != ptr;
            }
            else
            {
                this->compileArguments.floats[name] = value;
                return true;
            }
        }

        double MDLMaterial::GetParameterDouble(const char* name)
        {
            if (this->compilationType == CompilationType::CLASS)
            {
                double *ptr = GetParameter<double>(this->compiledMaterial, name);
                return ptr ? *ptr : 0.0;
            }
            else
            {
                auto it = this->compileArguments.doubles.find(name);
                if (it != this->compileArguments.doubles.end())
                    return it->second;

                return GetDefault<double>(this->loadedMaterial, name);
            }
        }

        bool MDLMaterial::SetParameterDouble(const char* name, double value)
        {
            if (this->compilationType == CompilationType::CLASS)
            {
                double *ptr = GetParameter<double>(this->compiledMaterial, name);
                if (ptr) { *ptr = value; UpdateArgumentBlock(); }
                return nullptr != ptr;
            }
            else
            {
                this->compileArguments.doubles[name] = value;
                return true;
            }
        }

        Vec3f MDLMaterial::GetParameterColor(const char* name)
        {
            if (this->compilationType == CompilationType::CLASS)
            {
                Vec3f *ptr = GetParameter<Vec3f>(this->compiledMaterial, name);
                return ptr ? *ptr : Vec3f(0.0f, 0.0f, 0.0f);
            }
            else
            {
                auto it = this->compileArguments.colors.find(name);
                if (it != this->compileArguments.colors.end())
                    return Vec3f(it->second.r, it->second.g, it->second.b);

                return GetDefault<Vec3f>(this->loadedMaterial, name);
            }
        }

        bool MDLMaterial::SetParameterColor(const char* name, const Vec3f &value)
        {
            if (this->compilationType == CompilationType::CLASS)
            {
                Vec3f *ptr = GetParameter<Vec3f>(this->compiledMaterial, name);
                if (ptr) { *ptr = value; UpdateArgumentBlock(); }
                return nullptr != ptr;
            }
            else
            {
                this->compileArguments.colors[name] = mi::math::Color(value.x, value.y, value.z);
                return true;
            }
        }

        bool MDLMaterial::SetParameterTexture(const char* name, VisRTX::Texture* texture)
        {
            bool changed = false;

            const std::string prefixedName = MDL_USE_TEXTURE_PREFIX + std::string(name);

            if (this->compilationType == CompilationType::CLASS)
            {
                // Assign texture
                if (texture)
                {
                    if (this->freeTextureSlots.empty())
                        throw VisRTX::Exception(VisRTX::Error::MDL_ERROR, "Out of texture slots");

                    // Get free slot
                    auto it = this->freeTextureSlots.begin();
                    const uint32_t slot = *it;

                    // Set slot index in arg block for parameter (returns false if parameter does not exist)
                    if (this->SetParameterInt(name, slot + 1)) // 0 is invalid texture
                    {
                        // TODO Workaround: Enforce usage of texture
                        this->SetParameterBool(prefixedName.c_str(), true);

                        // Mark slot as used and store texture handle
                        this->freeTextureSlots.erase(it);
                        this->usedTextureSlots[name] = slot;

                        // Update bindless sampler id at slot position
                        int* ids = static_cast<int *>(this->compiledMaterial.textures->map());
                        ids[slot] = dynamic_cast<VisRTX::Impl::Texture*>(texture)->sampler->getId();
                        this->compiledMaterial.textures->unmap();

                        changed = true;
                    }
                }

                // Release texture
                else
                {
                    auto it = this->usedTextureSlots.find(name);
                    if (it != this->usedTextureSlots.end())
                    {
                        // Set texture index in arg block (no need to update sampler buffer)
                        this->SetParameterInt(name, 0);

                        // TODO Workaround: Disable usage of texture
                        this->SetParameterBool(prefixedName.c_str(), false);

                        // Mark slot as free again
                        this->freeTextureSlots.insert(it->second);
                        this->usedTextureSlots.erase(it);

                        changed = true;
                    }
                }
            }

            // Instance compilation
            else
            {
                if (texture)
                {
                    this->compileArguments.textures[name] = dynamic_cast<VisRTX::Impl::Texture*>(texture)->sampler->getId();
                    this->compileArguments.bools[prefixedName] = true;
                }
                else
                {
                    this->compileArguments.textures.erase(name);
                    this->compileArguments.textures.erase(prefixedName);
                }

                changed = true;
            }

            // Update stored handle
            if (changed)
            {
                // Clear current handle
                auto it = this->textureHandles.find(name);
                if (it != this->textureHandles.end())
                {
                    if (it->second != nullptr)
                        it->second->Release();
                }

                // Store and increase ref count
                if (texture)
                {
                    this->textureHandles[name] = texture;
                    texture->Retain();
                }
            }

            return changed;
        }

        const char* MDLMaterial::GetParameterDescription(const char* name)
        {
            return ""; // TODO
        }

}
}

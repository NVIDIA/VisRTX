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

#include "Config.h"

#include <optixu/optixpp_namespace.h>
#include <mi/mdl_sdk.h>

#include <map>
#include <set>
#include <string>
#include <stdint.h>


const std::string MDL_USE_TEXTURE_PREFIX = "VisRTX_use_";


class MDL
{
    class Logger;

public:
    class Material
    {
        friend class MDL;

    public:
        static void Copy(Material& dst, Material& src);

        bool IsLoaded() const
        {
            return !this->instanceName.empty();
        }

    public:
        struct Parameter
        {
            std::string name;
            mi::neuraylib::IType::Kind kind;
            uint8_t defaultValue[32];
        };

        std::string material;
        std::string module;
        std::map<std::string, Parameter> parameters;

    private:
        std::string instanceName;
    };

    class CompiledMaterial
    {
    public:
        static void Copy(CompiledMaterial& dst, CompiledMaterial& src);

        bool IsCompiled() const
        {
            return this->initProg;
        }

        void Destroy()
        {
            for (optix::Program p : this->generatedPrograms)
                p->destroy();

            for (optix::TextureSampler s : this->generatedSamplers)
                s->destroy();

            for (optix::Buffer b : this->generatedBuffers)
                b->destroy();
        }

    public:
        struct Layout
        {
            size_t offset;
            size_t size;
        };

        // BSDF
        optix::Program initProg;
        optix::Program sampleProg;
        optix::Program evaluateProg;
        optix::Program pdfProg;

        // Cut-out opacity
        optix::Program opacityProg;

		// Refraction index
		optix::Program iorProg;

		// Thin-walled
		optix::Program thinwalledProg;

		// Volume absorption coefficient
		optix::Program absorbProg;

        // Argument block
        char argumentBlock[MDL_ARGUMENT_BLOCK_SIZE];
        std::map<std::string, Layout> layout;

        // Textures
        uint32_t numTextures; // number of texture slots used by hardcoded textures in material
        optix::Buffer textures; // bindless texture sampler ids
        uint32_t texturesOffset;

        // All OptiX handles required for cleanup
        std::set<optix::Program> generatedPrograms;
        std::set<optix::Buffer> generatedBuffers; // does not contain the pixel data buffers as these are cached for reuse
        std::set<optix::TextureSampler> generatedSamplers;
    };

    struct CompileArguments
    {
        std::map<std::string, bool> bools;
        std::map<std::string, int> ints;
        std::map<std::string, float> floats;
        std::map<std::string, double> doubles;
        std::map<std::string, mi::math::Color> colors;
        std::map<std::string, int> textures; // OptiX sampler ids
    };

public:
    MDL(optix::Context context, const std::string& texturesPTX);
    virtual ~MDL();

    void AddModulePath(const std::string& modulePath);

    Material Load(const std::string& material, const std::string& moduleName, const std::string& source);
    CompiledMaterial Compile(const Material& material, bool classCompilation, const CompileArguments& arguments);

    static std::string TypeToString(mi::neuraylib::IType::Kind kind);

private:
    std::string PreprocessSource(const std::string& source) const;

private:
    optix::Context context;
    std::string texturesPTX;

    mi::base::Handle<mi::neuraylib::INeuray> neuray;
    mi::base::Handle<mi::neuraylib::IMdl_compiler> compiler;
    mi::base::Handle<mi::neuraylib::IDatabase> database;
    mi::base::Handle<mi::neuraylib::IScope> globalScope;
    mi::base::Handle<mi::neuraylib::IMdl_factory> factory;
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> executionContext;
    mi::base::Handle<mi::neuraylib::IMdl_backend> cudaBackend;
    mi::base::Handle<mi::neuraylib::IImage_api> imageAPI;

    std::set<std::string> modulePaths;

    Logger* logger;

    static std::map<std::string, CompiledMaterial> compiledCache;
};

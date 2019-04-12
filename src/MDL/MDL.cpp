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

#include "MDL.h"
#include "neuray_loader.h"

#include <iostream>
#include <regex>
#include <fstream>


 // Regex replace with lambda from https://stackoverflow.com/questions/22617209/regex-replace-with-callback-in-c11
namespace std
{
    template<class BidirIt, class Traits, class CharT, class UnaryFunction>
    std::basic_string<CharT> regex_replace(BidirIt first, BidirIt last, const std::basic_regex<CharT, Traits>& re, UnaryFunction f)
    {
        std::basic_string<CharT> s;

        typename std::match_results<BidirIt>::difference_type positionOfLastMatch = 0;
        auto endOfLastMatch = first;

        auto callback = [&](const std::match_results<BidirIt>& match)
        {
            auto positionOfThisMatch = match.position(0);
            auto diff = positionOfThisMatch - positionOfLastMatch;

            auto startOfThisMatch = endOfLastMatch;
            std::advance(startOfThisMatch, diff);

            s.append(endOfLastMatch, startOfThisMatch);
            s.append(f(match));

            auto lengthOfMatch = match.length(0);

            positionOfLastMatch = positionOfThisMatch + lengthOfMatch;

            endOfLastMatch = startOfThisMatch;
            std::advance(endOfLastMatch, lengthOfMatch);
        };

        std::regex_iterator<BidirIt> begin(first, last, re), end;
        std::for_each(begin, end, callback);

        s.append(endOfLastMatch, last);

        return s;
    }

    template<class Traits, class CharT, class UnaryFunction>
    std::string regex_replace(const std::string& s, const std::basic_regex<CharT, Traits>& re, UnaryFunction f)
    {
        return regex_replace(s.cbegin(), s.cend(), re, f);
    }
}


// -------------- Utility functions adapted from MDL helper --------------

template<typename T>
std::string to_string(T val)
{
    std::ostringstream stream;
    stream << val;
    return stream.str();
}

// Throws an exception with the given error message, when expr is false.
void check_success(bool expr, const char *errMsg)
{
    if (expr) return;

    throw optix::Exception(errMsg);
}

// Throws an exception with the given error message, when expr is false.
void check_success(bool expr, const std::string &errMsg)
{
    if (expr) return;

    throw optix::Exception(errMsg);
}

// Throws an exception with the given error message, when expr is negative.
void check_success(mi::Sint32 errVal, const char *errMsg)
{
    if (errVal >= 0) return;

    throw optix::Exception(std::string(errMsg) + " (" + to_string(errVal) + ")");
}

// Throws an exception with the given error message, when expr is negative.
void check_success(mi::Sint32 errVal, const std::string &errMsg)
{
    if (errVal >= 0) return;

    throw optix::Exception(errMsg + " (" + to_string(errVal) + ")");
}

// Returns a string-representation of the given message severity.
const char* message_severity_to_string(mi::base::Message_severity severity)
{
    switch (severity)
    {
    case mi::base::MESSAGE_SEVERITY_ERROR:
        return "Error";
    case mi::base::MESSAGE_SEVERITY_WARNING:
        return "Warning";
    case mi::base::MESSAGE_SEVERITY_INFO:
        return "Info";
    case mi::base::MESSAGE_SEVERITY_VERBOSE:
        return "Verbose";
    case mi::base::MESSAGE_SEVERITY_DEBUG:
        return "Debug";
    default:
        break;
    }
    return "";
}

// Returns a string-representation of the given message category
const char* message_kind_to_string(mi::neuraylib::IMessage::Kind message_kind)
{
    switch (message_kind)
    {
    case mi::neuraylib::IMessage::MSG_INTEGRATION:
        return "MDL SDK";
    case mi::neuraylib::IMessage::MSG_IMP_EXP:
        return "Importer/Exporter";
    case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
        return "Compiler Backend";
    case mi::neuraylib::IMessage::MSG_COMILER_CORE:
        return "Compiler Core";
    case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
        return "Compiler Archive Tool";
    case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
        return "Compiler DAG generator";
    default:
        break;
    }
    return "";
}

// Prints the messages of the given context.
void check_success(mi::neuraylib::IMdl_execution_context* context)
{
    for (mi::Size i = 0; i < context->get_messages_count(); ++i)
    {
        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));

        switch (message->get_severity())
        {
        case mi::base::MESSAGE_SEVERITY_ERROR:
        case mi::base::MESSAGE_SEVERITY_FATAL:
        {
            std::string severity = message_severity_to_string(message->get_severity());
            std::string body = message->get_string();
            std::string kind = message_kind_to_string(message->get_kind());
            throw optix::Exception(severity + ": " + body + "(" + kind + ")");
        }
        default:
            fprintf(stderr, "%s: %s (%s)\n",
                message_severity_to_string(message->get_severity()),
                message->get_string(),
                message_kind_to_string(message->get_kind()));
            break;
        }
    }
}

template<typename I> struct Alpha_type_trait {};

template<> struct Alpha_type_trait<mi::Uint8>
{
    static mi::Uint8 get_alpha_max() { return 255; }
};

template<> struct Alpha_type_trait<mi::Sint8>
{
    static mi::Sint8 get_alpha_max() { return 127; }
};

template<> struct Alpha_type_trait<mi::Uint16>
{
    static mi::Uint16 get_alpha_max() { return 0xffff; }
};

template<> struct Alpha_type_trait<mi::Sint32>
{
    static mi::Sint32 get_alpha_max() { return 0x7fffffff; }
};

template<> struct Alpha_type_trait<mi::Float32>
{
    static mi::Float32 get_alpha_max() { return 1.0f; }
};


typedef enum
{
    CONV_IMG_1_TO_1,  ///< source and destination both have one component
    CONV_IMG_2_TO_2,  ///< source and destination both have two components
    CONV_IMG_3_TO_4,  ///< source has three and destination has four components, alpha is set to maximum alpha value for the according type
    CONV_IMG_4_TO_4   ///< source and destination both have four components
} Convertmode;

// Creates an OptiX buffer and fills it with image data converting it on the fly.
template <typename data_type, RTformat format, Convertmode conv_mode>
optix::Buffer LoadImageData(optix::Context context, mi::base::Handle<mi::neuraylib::IImage_api> imageAPI, mi::base::Handle<const mi::neuraylib::ICanvas> canvas, mi::neuraylib::ITarget_code::Texture_shape texture_shape)
{
    mi::Uint32 res_x = canvas->get_resolution_x();
    mi::Uint32 res_y = canvas->get_resolution_y();
    mi::Uint32 num_layers = canvas->get_layers_size();
    mi::Uint32 tiles_x = canvas->get_tiles_size_x();
    mi::Uint32 tiles_y = canvas->get_tiles_size_y();
    mi::Uint32 tile_res_x = canvas->get_tile_resolution_x();
    mi::Uint32 tile_res_y = canvas->get_tile_resolution_y();

    optix::Buffer buffer;

    if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_invalid)
        throw optix::Exception("ERROR: Invalid texture shape used");
    else if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_ptex)
        throw optix::Exception("ERROR: texture_ptex not supported yet");
    else if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube && num_layers != 6)
        throw optix::Exception("ERROR: texture_cube must be used with a texture with 6 layers");

    // We need to differentiate between createBuffer with 4 and 5 parameters here,
    // because the 5 parameter version always uses rtSetBufferSize3D
    // which prevents rtTex2D* functions from working.
    if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_2d)
    {
        buffer = context->createBuffer(RT_BUFFER_INPUT, format, res_x, res_y);

        // When using a texture as texture_2d, always enforce using only the first layer.
        num_layers = 1;
    }
    else if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_3d)
    {
        buffer = context->createBuffer(RT_BUFFER_INPUT, format, res_x, res_y, num_layers);
    }
    else
    {
        buffer = context->createBuffer(
            RT_BUFFER_INPUT |
            (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube ? RT_BUFFER_CUBEMAP : 0),
            format, res_x, res_y, num_layers);
    }

    data_type *buffer_data = static_cast<data_type *>(buffer->map());

    mi::Uint32 num_src_components =
        conv_mode == CONV_IMG_1_TO_1 ? 1
        : conv_mode == CONV_IMG_2_TO_2 ? 2
        : conv_mode == CONV_IMG_3_TO_4 ? 3
        : conv_mode == CONV_IMG_4_TO_4 ? 4
        : 0;

    for (mi::Uint32 layer = 0; layer < num_layers; ++layer)
    {
        for (mi::Uint32 tile_y = 0, tile_ypos = 0; tile_y < tiles_y;
            ++tile_y, tile_ypos += tile_res_y)
        {
            for (mi::Uint32 tile_x = 0, tile_xpos = 0; tile_x < tiles_x;
                ++tile_x, tile_xpos += tile_res_x)
            {
                mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(tile_xpos, tile_ypos, layer));
                const data_type *tile_data =
                    static_cast<const data_type *>(tile->get_data());

                mi::Uint32 x_end = std::min(tile_xpos + tile_res_x, res_x);
                mi::Uint32 x_pad = (tile_xpos + tile_res_x - x_end) * num_src_components;
                mi::Uint32 y_end = std::min(tile_ypos + tile_res_y, res_y);
                for (mi::Uint32 y = tile_y; y < y_end; ++y)
                {
                    for (mi::Uint32 x = tile_x; x < x_end; ++x)
                    {
                        *buffer_data++ = *tile_data++;                                     // R
                        if (conv_mode != CONV_IMG_1_TO_1)
                        {
                            *buffer_data++ = *tile_data++;                                 // G
                            if (conv_mode != CONV_IMG_2_TO_2)
                            {
                                *buffer_data++ = *tile_data++;                             // B
                                data_type alpha;
                                if (conv_mode == CONV_IMG_3_TO_4)
                                    alpha = Alpha_type_trait<data_type>::get_alpha_max();
                                else  // conv_mode == CONV_IMG_4_TO_4
                                    alpha = *tile_data++;
                                *buffer_data++ = alpha;                                    // A
                            }
                        }
                    }
                    tile_data += x_pad;  // possible padding due to tile size
                }
            }
        }
    }

    buffer->unmap();

    return buffer;
}

// Loads the texture given by the database name into an OptiX buffer converting it as necessary.
optix::Buffer LoadTexture(optix::Context context, mi::base::Handle<mi::neuraylib::IImage_api> imageAPI, mi::base::Handle<mi::neuraylib::ITransaction> transaction, const char *texture_name, mi::neuraylib::ITarget_code::Texture_shape texture_shape)
{
    /// Maps a texture name and a texture shape to an OptiX buffer to avoid the texture
    /// being loaded and converted multiple times.
    typedef std::map<std::string, optix::Buffer> TextureCache;
    static TextureCache m_texture_cache;

    // First check the texture cache
    std::string entry_name = std::string(texture_name) + "_" +
        to_string(unsigned(texture_shape));
    TextureCache::iterator it = m_texture_cache.find(entry_name);
    if (it != m_texture_cache.end())
        return it->second;

    mi::base::Handle<const mi::neuraylib::ITexture> texture(transaction->access<mi::neuraylib::ITexture>(texture_name));
    mi::base::Handle<const mi::neuraylib::IImage> image(transaction->access<mi::neuraylib::IImage>(texture->get_image()));
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas());

#ifdef MDL_VERBOSE_LOGGING
    std::cout << "   Image type: " << image->get_type()
        << "\n   Resolution: " << image->resolution_x() << " * " << image->resolution_y()
        << "\n   Layers: " << canvas->get_layers_size()
        << "\n   Canvas Resolution: "
        << canvas->get_resolution_x() << " * " << canvas->get_resolution_y()
        << "\n   Tile Resolution: "
        << canvas->get_tile_resolution_x() << " * " << canvas->get_tile_resolution_y()
        << "\n   Tile Size: "
        << canvas->get_tiles_size_x() << " * " << canvas->get_tiles_size_y()
        << "\n   Canvas Gamma: "
        << canvas->get_gamma()
        << "\n   Texture Gamma: " << texture->get_gamma() << " (effective: "
        << texture->get_effective_gamma() << ")"
        << std::endl;
#endif

    const char *image_type = image->get_type();

    // Convert to linear color space if necessary
    if (texture->get_effective_gamma() != 1.0f)
    {
        // Copy canvas and adjust gamma from "effective gamma" to 1
        mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(imageAPI->convert(canvas.get(), image->get_type()));
        gamma_canvas->set_gamma(texture->get_effective_gamma());
        imageAPI->adjust_gamma(gamma_canvas.get(), 1.0f);
        canvas = gamma_canvas;
    }

    // Handle the different image types (see \ref mi_neuray_types for the available pixel types)

    optix::Buffer buffer;
    if (!strcmp(image_type, "Sint8"))
    {
        buffer = LoadImageData<mi::Sint8, RT_FORMAT_BYTE, CONV_IMG_1_TO_1>(
            context, imageAPI, canvas, texture_shape);
    }
    else if (!strcmp(image_type, "Sint32"))
    {
        buffer = LoadImageData<mi::Sint32, RT_FORMAT_INT, CONV_IMG_1_TO_1>(
            context, imageAPI, canvas, texture_shape);
    }
    else if (!strcmp(image_type, "Float32"))
    {
        buffer = LoadImageData<mi::Float32, RT_FORMAT_FLOAT, CONV_IMG_1_TO_1>(
            context, imageAPI, canvas, texture_shape);
    }
    else if (!strcmp(image_type, "Float32<2>"))
    {
        buffer = LoadImageData<mi::Float32, RT_FORMAT_FLOAT2, CONV_IMG_2_TO_2>(
            context, imageAPI, canvas, texture_shape);
    }
    else if (!strcmp(image_type, "Rgb"))
    {
        // Note: OptiX does not support RT_FORMAT_UNSIGNED_BYTE3
        buffer = LoadImageData<mi::Uint8, RT_FORMAT_UNSIGNED_BYTE4, CONV_IMG_3_TO_4>(
            context, imageAPI, canvas, texture_shape);
    }
    else if (!strcmp(image_type, "Rgba"))
    {
        buffer = LoadImageData<mi::Uint8, RT_FORMAT_UNSIGNED_BYTE4, CONV_IMG_4_TO_4>(
            context, imageAPI, canvas, texture_shape);
    }
    else if (!strcmp(image_type, "Rgbe"))
    {
        // Convert to Rgb_fp first
        canvas = imageAPI->convert(canvas.get(), "Rgb_fp");
        buffer = LoadImageData<mi::Float32, RT_FORMAT_FLOAT4, CONV_IMG_3_TO_4>(
            context, imageAPI, canvas, texture_shape);
    }
    else if (!strcmp(image_type, "Rgbea"))
    {
        // Convert to Color first
        canvas = imageAPI->convert(canvas.get(), "Color");
        buffer = LoadImageData<mi::Float32, RT_FORMAT_FLOAT4, CONV_IMG_4_TO_4>(
            context, imageAPI, canvas, texture_shape);
    }
    else if (!strcmp(image_type, "Rgb_16"))
    {
        // Note: OptiX does not support RT_FORMAT_UNSIGNED_SHORT3
        buffer = LoadImageData<mi::Uint16, RT_FORMAT_UNSIGNED_SHORT4, CONV_IMG_3_TO_4>(
            context, imageAPI, canvas, texture_shape);
    }
    else if (!strcmp(image_type, "Rgba_16"))
    {
        buffer = LoadImageData<mi::Uint16, RT_FORMAT_UNSIGNED_SHORT4, CONV_IMG_4_TO_4>(
            context, imageAPI, canvas, texture_shape);
    }
    else if (!strcmp(image_type, "Rgb_fp") || !strcmp(image_type, "Float32<3>"))
    {
        // Note: OptiX does not support RT_FORMAT_FLOAT3
        buffer = LoadImageData<mi::Float32, RT_FORMAT_FLOAT4, CONV_IMG_3_TO_4>(
            context, imageAPI, canvas, texture_shape);
    }
    else if (!strcmp(image_type, "Color") || !strcmp(image_type, "Float32<4>"))
    {
        buffer = LoadImageData<mi::Float32, RT_FORMAT_FLOAT4, CONV_IMG_4_TO_4>(
            context, imageAPI, canvas, texture_shape);
    }
    else
        throw optix::Exception(
            std::string("ERROR: Image type \"") + image_type + "\" not supported, yet!");

    m_texture_cache[entry_name] = buffer;
    return buffer;
}

// ----------------------------------------------------------------------

class MDL::Logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    Logger(bool writeLogFile)
    {
        if (writeLogFile)
            this->logFile.open("mdl.log");
    }

    virtual void message(mi::base::Message_severity level, const char* module_category, const char* message)
    {
#ifdef MDL_VERBOSE_LOGGING
        const mi::base::Message_severity minLevel = mi::base::Message_severity::MESSAGE_SEVERITY_DEBUG;
#else
        const mi::base::Message_severity minLevel = mi::base::Message_severity::MESSAGE_SEVERITY_WARNING;
#endif

        if (level <= minLevel)
        {
            std::cout << "[" << module_category << "] " << message_severity_to_string(level) << ": " << message << std::endl;
        }

        if (this->logFile.is_open())
            this->logFile << "[" << module_category << "] " << message_severity_to_string(level) << ": " << message << std::endl;
    }

private:
    std::ofstream logFile;
};


std::map<std::string, MDL::CompiledMaterial> MDL::compiledCache;

void MDL::CompiledMaterial::Copy(MDL::CompiledMaterial& dst, MDL::CompiledMaterial& src)
{
    dst.initProg = src.initProg;
    dst.sampleProg = src.sampleProg;
    dst.evaluateProg = src.evaluateProg;
    dst.pdfProg = src.pdfProg;

    dst.opacityProg = src.opacityProg;
	dst.thinwalledProg = src.thinwalledProg;
	dst.iorProg = src.iorProg;
	dst.absorbProg = src.absorbProg;

    dst.layout = src.layout;
    memcpy(dst.argumentBlock, src.argumentBlock, MDL_ARGUMENT_BLOCK_SIZE);

    dst.numTextures = src.numTextures;
    dst.textures = src.textures;
    dst.texturesOffset = src.texturesOffset;
};


/**
 * Constructor.
 */
MDL::MDL(optix::Context context, const std::string& texturesPTX) : context(context), texturesPTX(texturesPTX)
{
#ifdef MI_PLATFORM_WINDOWS
    const std::string lib_mdl_sdk = "libmdl_sdk.dll";
    const std::string lib_nv_freeimage = "nv_freeimage.dll";
    const std::string lib_dds = "dds.dll";
#else
    const std::string lib_mdl_sdk = "libmdl_sdk.so";
    const std::string lib_nv_freeimage = "nv_freeimage.so";
    const std::string lib_dds = "dds.so";
#endif

    this->neuray = load_and_get_ineuray();
    check_success(neuray.is_valid_interface(), "Initialization of MDL SDK failed: " + lib_mdl_sdk + " not found or wrong version");

    this->compiler = this->neuray->get_api_component<mi::neuraylib::IMdl_compiler>();
    check_success(this->compiler, "Initialization of MDL compiler failed");

    this->logger = new Logger(false);
    this->compiler->set_logger(this->logger);

    // Add current working dir as default module path
    this->AddModulePath(".");

    // Load plugins for texture support
    check_success(this->compiler->load_plugin_library(lib_nv_freeimage.c_str()), "Failed to load plugin " + lib_nv_freeimage);
    check_success(this->compiler->load_plugin_library(lib_dds.c_str()), "Failed to load plugin " + lib_dds);

    check_success(this->neuray->start(), "Starting MDL SDK failed");

    this->database = this->neuray->get_api_component<mi::neuraylib::IDatabase>();
    this->globalScope = this->database->get_global_scope();
    this->factory = this->neuray->get_api_component<mi::neuraylib::IMdl_factory>();
    this->executionContext = this->factory->create_execution_context();

    this->executionContext->set_option("internal_space", "coordinate_world");  // equals default
    this->executionContext->set_option("bundle_resources", false);             // equals default
    this->executionContext->set_option("mdl_meters_per_scene_unit", 1.0f);     // equals default
    this->executionContext->set_option("mdl_wavelength_min", 380.0f);          // equals default
    this->executionContext->set_option("mdl_wavelength_max", 780.0f);          // equals default
    this->executionContext->set_option("include_geometry_normal", true);       // equals default

    this->cudaBackend = this->compiler->get_backend(mi::neuraylib::IMdl_compiler::MB_CUDA_PTX);

    check_success(this->cudaBackend->set_option("num_texture_spaces", "1") == 0, "Setting PTX option num_texture_spaces failed");
    check_success(this->cudaBackend->set_option("num_texture_results", to_string(MDL_MAX_TEXTURES).c_str()) == 0, "Setting PTX option num_texture_results failed");
    check_success(this->cudaBackend->set_option("sm_version", "30") == 0, "Setting PTX option sm_version failed");
    check_success(this->cudaBackend->set_option("tex_lookup_call_mode", "optix_cp") == 0, "Setting PTX option tex_lookup_call_mode failed");

    this->imageAPI = this->neuray->get_api_component<mi::neuraylib::IImage_api>();

    mi::base::Handle<mi::neuraylib::ITransaction> transaction = mi::base::make_handle(this->globalScope->create_transaction());
    {
        mi::base::Handle<mi::neuraylib::IImage> image(transaction->create<mi::neuraylib::IImage>("Image"));
        transaction->store(image.get(), "VisRTX_dummyImage");
    }
    transaction->commit();
}


/**
 * Destructor.
 */
MDL::~MDL()
{
    this->imageAPI.reset();
    this->cudaBackend.reset();
    this->executionContext.reset();
    this->factory.reset();
    this->globalScope.reset();
    this->database.reset();
    this->compiler.reset();

    this->neuray->shutdown();

    delete this->logger;
}


/**
 * Loads the given material either from a module file or given source code.
 */
MDL::Material MDL::Load(const std::string& material, const std::string& moduleName, const std::string& source)
{
    Material result;
    result.material = material;
    result.module = moduleName;

    /*
     * Load material
     */
    const std::string mdlModuleName = (moduleName.find("::") == 0) ? moduleName : "::" + moduleName;
    const std::string mdlMaterialName = "mdl" + mdlModuleName + "::" + material;

    static int id = 0;
    result.instanceName = std::to_string(id++) + "-" + mdlMaterialName;

    mi::base::Handle<mi::neuraylib::ITransaction> transaction = mi::base::make_handle(this->globalScope->create_transaction());
    {
        // Check if material definition is already in database
        mi::base::Handle<const mi::neuraylib::IMaterial_definition> materialDefinition = mi::base::make_handle(transaction->access<mi::neuraylib::IMaterial_definition>(mdlMaterialName.c_str()));
        if (!materialDefinition)
        {
            // Load material
#ifdef MDL_VERBOSE_LOGGING
            std::cout << "[MDL] Loading material definition for \"" << material << "\" from module \"" << moduleName << "\"" << std::endl;
#endif

            if (source.empty())
                check_success(this->compiler->load_module(transaction.get(), mdlModuleName.c_str()), "Loading \"" + moduleName + ".mdl\" failed");
            else
                check_success(this->compiler->load_module_from_string(transaction.get(), mdlModuleName.c_str(), this->PreprocessSource(source).c_str()), "Loading \"" + moduleName + "\" from memory failed");

            materialDefinition = mi::base::make_handle(transaction->access<mi::neuraylib::IMaterial_definition>(mdlMaterialName.c_str()));
        }
        else
        {
#ifdef MDL_VERBOSE_LOGGING
            std::cout << "[MDL] Loaded material definition for \"" << material << "\" from database" << std::endl;
#endif
        }

        check_success(materialDefinition, "Material " + mdlMaterialName + " not found");

        // Instantiate material
        mi::base::Handle<const mi::neuraylib::IExpression_list> arguments;
        mi::Sint32 errors;
        mi::base::Handle<mi::neuraylib::IMaterial_instance> materialInstance = mi::base::make_handle(materialDefinition->create_material_instance(arguments.get(), &errors));
        check_success(errors, "Creating material instance failed");

        transaction->store(materialInstance.get(), result.instanceName.c_str());
    }
    transaction->commit();


    /*
     * Extract parameter info
     */
    result.parameters.clear();
    transaction = mi::base::make_handle(this->globalScope->create_transaction());
    {
        mi::neuraylib::Argument_editor editor(transaction.get(), result.instanceName.c_str(), this->factory.get());

#ifdef MDL_VERBOSE_LOGGING
        std::cout << "[MDL]  - Parameters: " << editor.get_parameter_count() << std::endl;
#endif
        for (int i = 0; i < editor.get_parameter_count(); ++i)
        {
            Material::Parameter p;

            // Name
            const char* name = editor.get_parameter_name(i);
            p.name = std::string(name);

            // Kind
            auto types = editor.get_parameter_types();
            auto type = types->get_type(name);
            auto resolvedType = type->skip_all_type_aliases();
            p.kind = resolvedType->get_kind();
            resolvedType->release();
            type->release();
            types->release();

            // Value
            if (p.kind == mi::neuraylib::IType::Kind::TK_BOOL)
                editor.get_value(name, (bool&)(*p.defaultValue));
            else if (p.kind == mi::neuraylib::IType::Kind::TK_FLOAT)
                editor.get_value(name, (float&)(*p.defaultValue));
            else if (p.kind == mi::neuraylib::IType::Kind::TK_DOUBLE)
                editor.get_value(name, (double&)(*p.defaultValue));
            else if (p.kind == mi::neuraylib::IType::Kind::TK_INT)
                editor.get_value(name, (int&)(*p.defaultValue));
            else if (p.kind == mi::neuraylib::IType::Kind::TK_TEXTURE)
                editor.get_value(name, (int&)(*p.defaultValue));
            else if (p.kind == mi::neuraylib::IType::Kind::TK_COLOR)
                editor.get_value(name, (mi::math::Color&)(*p.defaultValue));
#ifdef MDL_VERBOSE_LOGGING
            else
                std::cout << "[MDL] Warning: Unhandled parameter type: " << (int)p.kind << std::endl;
#endif

#ifdef MDL_VERBOSE_LOGGING
            std::cout << "[MDL]      " << name << " (type: " << MDL::TypeToString(p.kind) << ")" << std::endl;
#endif

            result.parameters[p.name] = p;
        }
    }
    transaction->commit();

    return result;
}


/**
 * Compiles the given material.
 */
MDL::CompiledMaterial MDL::Compile(const MDL::Material& material, bool classCompilation, const MDL::CompileArguments& arguments)
{
    CompiledMaterial result;

    if (!material.IsLoaded())
        return result;

    /*
     * Check cache (class compilation only)
     */
    bool cacheHit = false;
    const std::string cacheKey = material.module + material.material;
    if (classCompilation)
    {
        auto it = compiledCache.find(cacheKey);
        if (it != compiledCache.end())
        {
            CompiledMaterial::Copy(result, it->second);
            cacheHit = true;

            // Increase texture buffer size
            RTsize size;
            result.textures->getSize(size);

            std::vector<int> tmp(size);

            int* ids = (int*)result.textures->map();
            memcpy(tmp.data(), ids, size * sizeof(int));
            result.textures->unmap();

            // TODO: Why doesn't this preserve contents? (so the two memcpys should not be necessary)
            result.textures->setSize(size + MDL_MAX_TEXTURES);

            ids = (int*)result.textures->map();
            memcpy(ids, tmp.data(), size * sizeof(int));

            for (int i = 0; i < MDL_MAX_TEXTURES; ++i)
                ids[size + i] = 0;

            result.textures->unmap();

            // Update texture offset for this material
            result.texturesOffset = (uint32_t)size;

#ifdef MDL_VERBOSE_LOGGING
            std::cout << "[MDL] Loaded compiled material \"" << material.material << "\" from cache" << std::endl;
#endif
        }
    }

    /*
     * Compile (cache miss)
     */
    if (!cacheHit)
    {
#ifdef MDL_VERBOSE_LOGGING
        std::cout << "[MDL] Compiling material \"" << material.material << "\"" << std::endl;
#endif

        /*
         * Set parameters/arguments
         */
        std::map<std::string, int> textureOverrides;

        if (!classCompilation)
        {
            mi::base::Handle<mi::neuraylib::ITransaction> transaction = mi::base::make_handle(this->globalScope->create_transaction());
            {
                mi::neuraylib::Argument_editor editor(transaction.get(), material.instanceName.c_str(), this->factory.get());

                // Standard values
                for (auto it : arguments.bools)
                    editor.set_value(it.first.c_str(), it.second);

                for (auto it : arguments.ints)
                    editor.set_value(it.first.c_str(), it.second);

                for (auto it : arguments.floats)
                    editor.set_value(it.first.c_str(), it.second);

                for (auto it : arguments.doubles)
                    editor.set_value(it.first.c_str(), it.second);

                for (auto it : arguments.colors)
                    editor.set_value(it.first.c_str(), it.second);

                // Textures
                mi::base::Handle<mi::neuraylib::IMaterial_instance> materialInstance = mi::base::make_handle(transaction->edit<mi::neuraylib::IMaterial_instance>(material.instanceName.c_str()));

                for (auto it : arguments.textures)
                {
                    const std::string name = it.first;
                    const int sampler = it.second;

                    if (sampler > 0)
                    {
                        // Create dummy texture object and assign expression to argument
                        static int counter = 0;
                        const int id = counter++;
                        const std::string textureName = "VisRTX_texture" + std::to_string(id);

                        mi::base::Handle<mi::neuraylib::ITexture> texture = mi::base::make_handle(transaction->create<mi::neuraylib::ITexture>("Texture"));
                        texture->set_image("VisRTX_dummyImage");
                        transaction->store(texture.get(), textureName.c_str());

                        mi::base::Handle<mi::neuraylib::IValue_factory> valueFactory(editor.get_value_factory());
                        mi::base::Handle<mi::neuraylib::IExpression_factory> expressionFactory(editor.get_expression_factory());

                        mi::base::Handle<const mi::neuraylib::IType_list> types(editor.get_parameter_types());
                        mi::base::Handle<const mi::neuraylib::IType> arg_type(types->get_type(name.c_str()));

                        mi::base::Handle<mi::neuraylib::IValue_texture> arg_value(valueFactory->create<mi::neuraylib::IValue_texture>(arg_type.get()));
                        arg_value->set_value(textureName.c_str());

                        mi::base::Handle<mi::neuraylib::IExpression> arg_expr(expressionFactory->create_constant(arg_value.get()));
                        materialInstance->set_argument(name.c_str(), arg_expr.get());

                        // Save texture name for sampler id replacement at later stage
                        textureOverrides[textureName] = sampler;
                    }
                }
            }
            transaction->commit();
        }


        /*
         * Compile
         */
        mi::base::Handle<mi::neuraylib::ITransaction> transaction = mi::base::make_handle(this->globalScope->create_transaction());
        {
            mi::base::Handle<const mi::neuraylib::IMaterial_instance> materialInstance = mi::base::make_handle(transaction->access<mi::neuraylib::IMaterial_instance>(material.instanceName.c_str()));

            mi::neuraylib::IMaterial_instance::Compilation_options compilationType = classCompilation ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
            mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial = mi::base::make_handle(materialInstance->create_compiled_material(compilationType, this->executionContext.get()));
            check_success(this->executionContext.get());

            mi::base::Handle<mi::neuraylib::ILink_unit> linkUnit = mi::base::make_handle(this->cudaBackend->create_link_unit(transaction.get(), this->executionContext.get()));

            std::vector<mi::neuraylib::Target_function_description> functionDescriptions =
            {
                mi::neuraylib::Target_function_description("surface.scattering", "surface_scattering"),
                mi::neuraylib::Target_function_description("geometry.cutout_opacity", "geometry_cutout_opacity"),
				mi::neuraylib::Target_function_description("thin_walled", "thin_walled"),
				mi::neuraylib::Target_function_description("ior", "ior"),
				mi::neuraylib::Target_function_description("volume.absorption_coefficient", "volume_absorption_coefficient"),
            };

            linkUnit->add_material(compiledMaterial.get(), functionDescriptions.data(), functionDescriptions.size(), this->executionContext.get());

            mi::base::Handle<const mi::neuraylib::ITarget_code> code = mi::base::make_handle(this->cudaBackend->translate_link_unit(linkUnit.get(), this->executionContext.get()));
            check_success(this->executionContext.get());
            check_success(code, "Failed to translate link unit to target code");

#ifdef MDL_DUMP_PTX
            static int prog = 0;
            std::string fname = material + "_" + to_string(prog++) + ".ptx";
            FILE *file = fopen(fname.c_str(), "wt");
            fwrite(code->get_code(), 1, code->get_code_size(), file);
            fclose(file);
#endif

            std::vector<optix::Program*> programs =
            {
                &result.initProg,
                &result.sampleProg,
                &result.evaluateProg,
                &result.pdfProg,
                &result.opacityProg,
				&result.thinwalledProg,
				&result.iorProg,				
				&result.absorbProg
            };

            std::vector<std::string> functionNames(programs.size());
            for (size_t i = 0; i < programs.size(); ++i)
                functionNames[i] = code->get_callable_function(i);

            const std::string ptx = std::string(code->get_code(), code->get_code_size());

            for (size_t i = 0; i < programs.size(); ++i)
                *programs[i] = this->context->createProgramFromPTXString(ptx, functionNames[i]);


            /*
             * Argument block
             */
            if (code->get_argument_block_count() != 0)
            {
                check_success(code->get_argument_block_count() == 1, "More than one argument block encountered");

                mi::base::Handle<mi::neuraylib::ITarget_argument_block const> argumentBlock = mi::base::make_handle(code->get_argument_block(0));
                mi::base::Handle<mi::neuraylib::ITarget_value_layout const> layout = mi::base::make_handle(code->get_argument_block_layout(0));

#ifdef MDL_VERBOSE_LOGGING
                std::cout << "[MDL]  - Arg block: " << argumentBlock->get_size() << " bytes" << std::endl;
                std::cout << "[MDL]  - Parameters: " << compiledMaterial->get_parameter_count() << std::endl;
#endif

                for (unsigned i = 0u; i < compiledMaterial->get_parameter_count(); i++)
                {
                    std::string name = compiledMaterial->get_parameter_name(i);
                    mi::neuraylib::Target_value_layout_state nestedState = layout->get_nested_state(i);
                    mi::neuraylib::IValue::Kind kind;
                    mi::Size size;
                    mi::Size offset = layout->get_layout(kind, size, nestedState);

                    CompiledMaterial::Layout info = {};
                    info.offset = offset;
                    info.size = size;

                    result.layout[name] = info;

#ifdef MDL_VERBOSE_LOGGING
                    std::cout << "[MDL]      " << name << " (size: " << size << ", offset: " << offset << ")" << std::endl;
#endif
                }

                check_success(argumentBlock->get_size() <= MDL_ARGUMENT_BLOCK_SIZE, "Argument blocks larger than " + std::to_string(MDL_ARGUMENT_BLOCK_SIZE) + " are not supported");
                memcpy(result.argumentBlock, argumentBlock->get_data(), argumentBlock->get_size());
            }


            /*
             * Textures
             */
            result.textures = this->context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, MDL_MAX_TEXTURES);
            result.texturesOffset = 0;
            result.numTextures = 0;

            result.generatedBuffers.insert(result.textures);

#ifdef MDL_VERBOSE_LOGGING
            std::cout << "[MDL]  - Texture slots: " << code->get_texture_count() << std::endl;
#endif

            int* ids = static_cast<int *>(result.textures->map());

            for (int i = 0; i < MDL_MAX_TEXTURES; ++i)
                ids[i] = RT_TEXTURE_ID_NULL;

            // Load textures from disk
            if (code->get_texture_count() > 1) // 0 is invalid texture (if material potentially uses a texture)
            {
                // i = 0 is the invalid texture, so we skip it
                for (mi::Size i = 0; i < code->get_texture_count(); ++i)
                {
                    const char* texture = (i > 0) ? code->get_texture(i) : "null";

                    auto texIt = textureOverrides.find(texture);
                    const bool hasOverride = texIt != textureOverrides.end();

#ifdef MDL_VERBOSE_LOGGING
                    std::string overrideString = hasOverride ? ("OptiX bindless sampler ID: " + std::to_string(texIt->second)) : "";
                    const char* printTexture = hasOverride ? overrideString.c_str() : texture;
                    std::cout << "[MDL]      [" << i << "] " << printTexture << std::endl;
#endif
                    if (i <= 0)
                        continue;

                    if (hasOverride)
                    {
                        ids[i - 1] = texIt->second;
                    }
                    else
                    {
                        optix::TextureSampler sampler = this->context->createTextureSampler();

                        // For cube maps use clamped address mode to avoid artifacts in the corners
                        mi::neuraylib::ITarget_code::Texture_shape shape = code->get_texture_shape(i);
                        if (shape == mi::neuraylib::ITarget_code::Texture_shape_cube)
                        {
                            sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
                            sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
                            sampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
                        }

                        optix::Buffer buffer = LoadTexture(this->context, this->imageAPI, transaction, texture, shape);
                        sampler->setBuffer(buffer);

                        ids[i - 1] = sampler->getId();

                        result.generatedSamplers.insert(sampler);
                    }

                    ++result.numTextures;
                }
            }

            result.textures->unmap();


            // Set up the texture access functions
            static const char *tex_prog_names[] = {
                "tex_lookup_float4_2d",
                "tex_lookup_float3_2d",
                "tex_lookup_float3_cube",
                "tex_texel_float4_2d",
                "tex_lookup_float4_3d",
                "tex_lookup_float3_3d",
                "tex_texel_float4_3d",
                "tex_resolution_2d"
            };

            std::vector<optix::Program*> allPrograms = {
                &result.initProg,
                &result.sampleProg,
                &result.evaluateProg,
                &result.pdfProg,
                &result.opacityProg,
				&result.thinwalledProg,
				&result.iorProg,				
				&result.absorbProg
            };

            for (size_t i = 0; i < sizeof(tex_prog_names) / sizeof(*tex_prog_names); ++i)
            {
                optix::Program texprog = this->context->createProgramFromPTXString(this->texturesPTX, tex_prog_names[i]);
                check_success(texprog, std::string("Compiling ") + tex_prog_names[i] + "failed");
                texprog["texture_sampler_ids"]->setInt(result.textures->getId());

                for (size_t j = 0, n = allPrograms.size(); j < n; ++j)
                    (*allPrograms[j])[tex_prog_names[i]]->setProgramId(texprog);

                result.generatedPrograms.insert(texprog);
            }

            for (optix::Program* p : allPrograms)
                result.generatedPrograms.insert(*p);
        }
        transaction->commit();

        // Store in cache
        if (classCompilation)
            CompiledMaterial::Copy(compiledCache[cacheKey], result);
    }

    return result;
}


/**
 * Adds a module search path.
 */
void MDL::AddModulePath(const std::string& modulePath)
{
    if (modulePath.empty())
        return;

    if (this->modulePaths.find(modulePath) == this->modulePaths.end())
    {
        this->compiler->add_module_path(modulePath.c_str());
        this->modulePaths.insert(modulePath);
    }
}


/**
 * Source code preprocessing.
 */
std::string MDL::PreprocessSource(const std::string& source) const
{
    std::set<std::string> textureVariables;

    /*
     * (1) Replace material signature
     */
     // Find all texture material parameters with invalid texture as default, i.e., no image path specified
    std::regex textureDeclareRegex("uniform.*?texture_2d.*?((?:[a-zA-Z][a-zA-Z0-9_]*)).*?=.*?texture_2d.*?\\(\\).*?(\\[\\[anno::unused\\(\\)\\]\\])?", std::regex::icase);

    std::string phase1 = std::regex_replace(source, textureDeclareRegex, [&](const std::smatch& match)
    {
        // 2 matches: (full string, texture variable name)
        // (0) uniform texture_2d map_Kd = texture_2d()
        // (1) map_Kd
        // (2) [[anno::unused()]]  // optional
        if (match.size() >= 2)
        {
            // Save variable name
            const std::string var = match[1];
            textureVariables.insert(var);

            // Check if [[anno::unused()]]
            bool unused = false;
            if (match.size() == 3)
            {
                const std::string anno = match[2];
                if (anno.find("anno::unused") != std::string::npos)
                    unused = true;
            }

            // Introduce bool VisRTX_use_* variable before texture
            if (unused)
                return "uniform bool " + MDL_USE_TEXTURE_PREFIX + var + " = false [[anno::unused()]], " + match.str();
            else
                return "uniform bool " + MDL_USE_TEXTURE_PREFIX + var + " = false, " + match.str();
        }

        return match.str();
    });

    /*
     * (2) Replace tex::isvalid calls
     */
     // Find all texture_isvalid calls
    std::regex textureIsValidRegex("tex::texture_isvalid.*?\\(.*?((?:[a-zA-Z][a-zA-Z0-9_]*)).*?\\)", std::regex::icase);

    std::string phase2 = std::regex_replace(phase1, textureIsValidRegex, [&](const std::smatch& match)
    {
        // 2 matches: (full string, texture variable name)
        // (0) tex::texture_isvalid(map_Kd)
        // (1) map_Kd
        if (match.size() == 2)
        {
            // Check if variable is in set of invalid default textures
            const std::string var = match[1];

            if (textureVariables.find(var) != textureVariables.end())
            {
                // Replace with bool variable
                return MDL_USE_TEXTURE_PREFIX + var;

                // This would also be possible! But it's slower than above bool
                //return "(tex::width(" + var + ") > 0)";
            }
        }

        return match.str();
    });

    return phase2;
}

std::string MDL::TypeToString(mi::neuraylib::IType::Kind kind)
{
    switch (kind)
    {
    case mi::neuraylib::IType::Kind::TK_BOOL:
        return "bool";
    case mi::neuraylib::IType::Kind::TK_COLOR:
        return "color";
    case mi::neuraylib::IType::Kind::TK_DOUBLE:
        return "double";
    case mi::neuraylib::IType::Kind::TK_FLOAT:
        return "float";
    case mi::neuraylib::IType::Kind::TK_INT:
        return "int";
    case mi::neuraylib::IType::Kind::TK_TEXTURE:
        return "texture";
    default:
        return "other";
    }
}

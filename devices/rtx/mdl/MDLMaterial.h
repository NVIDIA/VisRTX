/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// #include "gpu/material.cuh"

#include <filesystem>
#include <unordered_map>
#include "mdl/materialSbtData.cuh"

namespace visrtx {

class MDLMaterial
{
 public:
  OptixProgramGroup getMaterialProgramGroup() const;
  MaterialSbtData getMaterialSbt() const;

 private:
  // The compiled material.
  mi::base::Handle<mi::neuraylib::ICompiled_material const> compiled_material;

  /// The generated target code object.
  mi::base::Handle<mi::neuraylib::ITarget_code const> target_code;

  /// The argument block for the compiled material.
  mi::base::Handle<mi::neuraylib::ITarget_argument_block const> argument_block;

  /// Information required to load a texture.
  struct Texture_info
  {
    std::string db_name;
    mi::neuraylib::ITarget_code::Texture_shape shape;

    Texture_info() : shape(mi::neuraylib::ITarget_code::Texture_shape_invalid)
    {}

    Texture_info(
        char const *db_name, mi::neuraylib::ITarget_code::Texture_shape shape)
        : db_name(db_name), shape(shape)
    {}
  };

  /// Information required to load a light profile.
  struct Light_profile_info
  {
    std::string db_name;

    Light_profile_info() {}

    Light_profile_info(char const *db_name) : db_name(db_name) {}
  };

  /// Information required to load a BSDF measurement.
  struct Bsdf_measurement_info
  {
    std::string db_name;

    Bsdf_measurement_info() {}

    Bsdf_measurement_info(char const *db_name) : db_name(db_name) {}
  };

  /// Textures used by the compile result.
  std::vector<Texture_info> textures;

  /// Textures used by the compile result.
  std::vector<Light_profile_info> light_profiles;

  /// Textures used by the compile result.
  std::vector<Bsdf_measurement_info> bsdf_measurements;

  /// Constructor.
  MDLMaterial()
  {
    // add invalid resources
    textures.emplace_back();
    light_profiles.emplace_back();
    bsdf_measurements.emplace_back();
  }
};

} // namespace visrtx

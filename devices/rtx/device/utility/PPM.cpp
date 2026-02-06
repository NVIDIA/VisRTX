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
#include <anari/frontend/anari_enums.h>
#include <anari/frontend/type_utility.h>
#include <cuda_runtime.h>

#include <glm/fwd.hpp>
#include <iostream>
#include <fstream>

#include <glm/glm.hpp>

namespace visrtx {

bool saveToPXm(const char* filename, const char* data, ANARIDataType dataType, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    int itemByteSize = anari::sizeOf(dataType);

    cudaPointerAttributes attr{};
    cudaPointerGetAttributes(&attr, data);
    if (attr.type == cudaMemoryTypeDevice) {
        // Copy device memory to host
        auto hostData = new char[itemByteSize * width * height];
        cudaMemcpy(hostData, data, itemByteSize * width * height, cudaMemcpyDeviceToHost);
        data = hostData;
    }

    // Write PFM header
    switch (dataType) {
        case ANARI_UFIXED8:
            file << "P5\n" << width << " " << height << "\n255\n";
            break;
        case ANARI_UFIXED16:
            file << "P5\n" << width << " " << height << "\n65535\n";
            break;
        case ANARI_UFIXED8_VEC3:
            file << "P6\n" << width << " " << height << "\n255\n";
            break;
        case ANARI_UFIXED16_VEC3:
            file << "P6\n" << width << " " << height << "\n65535\n";
            break;
        case ANARI_FLOAT32_VEC3:
            file << "PF\n" << width << " " << height << "\n-1.0\n";
            break;
        case ANARI_FLOAT32:
            file << "Pf\n" << width << " " << height << "\n-1.0\n";
            break;
        default:
            std::cerr << "Unsupported data type for PFM: " << anari::toString(dataType) << "\n";
            return false;
    }

    // Write pixel data
    file.write(data, width * height * itemByteSize);
    file.close();

    if (attr.type == cudaMemoryTypeDevice) {
        // Free the temporary host memory if it was allocated
        delete[] data;
    }

    return true;
}

bool saveToPfm(const char* filename, const float* data, int width, int height) {
    return saveToPXm(filename, reinterpret_cast<const char*>(data), ANARI_FLOAT32, width, height);
}

bool saveToPfm(const char* filename, const glm::vec3* data, int width, int height) {
    return saveToPXm(filename, reinterpret_cast<const char*>(data), ANARI_FLOAT32_VEC3, width, height);
}

bool saveToPgm(const char* filename, const unsigned char* data, int width, int height) {
    return saveToPXm(filename, reinterpret_cast<const char*>(data), ANARI_UFIXED8, width, height);
}

bool saveToPpm(const char* filename, const glm::u8vec3* data, int width, int height) {
    return saveToPXm(filename, reinterpret_cast<const char*>(data), ANARI_UFIXED8_VEC3, width, height);
}


} // namespace visrtx

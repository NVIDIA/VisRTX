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

#include "Sample.h"
#include "OSPRayMDL.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <vector>
#include <limits>
#include <algorithm>



 /*
  * Example to test MDL-based implementations of the OSPRay material definitions. This sample supports both instance and class compilation for MDL materials.
  */

const VisRTX::CompilationType compilationType = CompilationType::INSTANCE;
//const VisRTX::CompilationType compilationType = CompilationType::CLASS;


class SampleMDL : public Sample
{
public:
    bool Init(int argc, char **argv) override
    {
        // ---------------------------------------------------------------------------------------------------------------
        // Resource paths
        // ---------------------------------------------------------------------------------------------------------------

        std::string objPath = "cow.obj"; // your mesh here

        for (int i = 0; i < argc - 1; ++i)
        {
            if (std::string(argv[i]) == "-m")
                objPath = std::string(argv[i + 1]);
        }


        const std::vector<std::string> mdlModulePaths =
        {
            //// vMaterials (https://www.nvidia.com/en-us/design-visualization/technologies/vmaterials/)
            //"C:/mdl-vmaterials/mdl",

            //// MDL SDK (for NVIDIA core definitions)
            //"C:/mdl-sdk-312200.1281/examples/mdl",

            //// Ground plane material from MDL example in Optix SDK
            //"C:/OptiX-SDK-6.0.0/SDK-precompiled-samples/media/data",
        };

        // Load OSPRay materials from memory
        const std::string osprayMDLSource((const char*)OSPRay_mdl, sizeof(OSPRay_mdl));

        std::vector<std::string> meshMaterials =
        {
            "::OSPRay::Alloy",
            "::OSPRay::Glass",
            "::OSPRay::ThinGlass",
            "::OSPRay::MetallicPaint",
            "::OSPRay::OBJMaterial",
            "::OSPRay::CarPaint",
            "::OSPRay::Principled"
        };

        std::string planeMaterial = "::OSPRay::Metal";

        // ---------------------------------------------------------------------------------------------------------------

        this->ambientColor.r = 0.95f;
        this->ambientColor.g = 0.95f;
        this->ambientColor.b = 0.95f;

        this->rotationHorizontal = 0.0f;
        this->rotationVertical = 0.0f;
        this->distance = 7.5f;

        this->depthOfField = true;
        this->focalDistance = 6.7f;
        this->apertureRadius = 0.05f;

        this->numBouncesMin = 1;
        this->numBouncesMax = 15;


        VisRTX::Context* context = VisRTX_GetContext();

        std::cout << "Loading resources..." << std::endl;

        /*
         * Load model
         */
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string warn;
        std::string err;
        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objPath.c_str());

        if (!err.empty())
        {
            std::cerr << err << std::endl;
            return false;
        }

        std::vector<VisRTX::Vec3f> vertices;
        std::vector<VisRTX::Vec3f> normals;
        std::vector<VisRTX::Vec2f> texcoords;
        std::vector<VisRTX::Vec3ui> triangles;

        for (size_t s = 0; s < shapes.size(); s++)
        {
            size_t index_offset = 0;
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
            {
                int fv = shapes[s].mesh.num_face_vertices[f];

                if (fv == 3)
                {
                    uint32_t vertexOffset = (uint32_t)vertices.size();

                    for (size_t v = 0; v < fv; v++)
                    {
                        // access to vertex
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                        if (idx.vertex_index >= 0)
                        {
                            tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                            tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                            tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

                            vertices.push_back(VisRTX::Vec3f(vx, vy, vz));
                        }

                        if (idx.normal_index >= 0)
                        {
                            tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                            tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                            tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

                            normals.push_back(VisRTX::Vec3f(nx, ny, nz));
                        }

                        if (idx.texcoord_index >= 0)
                        {
                            tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                            tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
                        }
                    }
                    index_offset += fv;

                    triangles.push_back(VisRTX::Vec3ui(vertexOffset, vertexOffset + 1, vertexOffset + 2));
                }
            }
        }

        // Normalize model
        VisRTX::Vec3f min, max;
        min.x = min.y = min.z = std::numeric_limits<float>::max();
        max.x = max.y = max.z = -std::numeric_limits<float>::max();

        for (const VisRTX::Vec3f& v : vertices)
        {
            min.x = std::min(min.x, v.x);
            min.y = std::min(min.y, v.y);
            min.z = std::min(min.z, v.z);

            max.x = std::max(max.x, v.x);
            max.y = std::max(max.y, v.y);
            max.z = std::max(max.z, v.z);
        }

        const float size = std::max(max.x - min.x, std::max(max.y - min.y, max.z - min.z));
        const float sx = (max.x - min.x) / size;
        const float sy = (max.y - min.y) / size;
        const float sz = (max.z - min.z) / size;


        /*
         * Create instances with materials
         */
         // Determine grid resolution
        size_t numMaterials = meshMaterials.size();

        const float cellSize = 10.0f / numMaterials;

        const float normalizedSize = 0.9f * cellSize;

        bool computeTexcoords = texcoords.size() <= 0;
        texcoords.reserve(vertices.size());

        for (VisRTX::Vec3f& v : vertices)
        {
            const float nx = (v.x - min.x) / size;
            const float ny = (v.y - min.y) / size;
            const float nz = (v.z - min.z) / size;

            v.x = -0.5f * sx * normalizedSize + normalizedSize * nx;
            v.y = -0.5f * sy * normalizedSize + normalizedSize * ny;
            v.z = -0.5f * sz * normalizedSize + normalizedSize * nz;

            // Use XY plane as tex coords if none were provided
            if (computeTexcoords)
                texcoords.push_back(VisRTX::Vec2f(nx, ny));
        }

        for (size_t i = 0; i < numMaterials; ++i)
        {
            std::vector<VisRTX::Vec3f> vertTrans(vertices.size());;

            const float px = -5.0f + (i + 0.5f) * cellSize;

            for (int j = 0; j < vertices.size(); ++j)
            {
                vertTrans[j].x = vertices[j].x + px;
                vertTrans[j].y = vertices[j].y;
                vertTrans[j].z = vertices[j].z;
            }

            VisRTX::TriangleGeometry* mesh = context->CreateTriangleGeometry((uint32_t)triangles.size(), triangles.data(), (uint32_t)vertTrans.size(), vertTrans.data(), normals.empty() ? nullptr : normals.data());
            mesh->SetTexCoords(texcoords.data());

            VisRTX::MDLMaterial* meshMaterial = this->LoadMDL(meshMaterials[i], osprayMDLSource, mdlModulePaths, compilationType, 0, "");
            mesh->SetMaterial(meshMaterial);
            meshMaterial->Release();

            model->AddGeometry(mesh);
            this->releaseLater.insert(mesh);
        }


        /*
         * Ground plane
         */
        const float planeSize = 6.0f;
        Vec3f planeVertices[4] = {
            Vec3f(-planeSize, -0.5f * normalizedSize * sy, -planeSize),
            Vec3f(-planeSize, -0.5f * normalizedSize * sy,  planeSize),
            Vec3f(planeSize, -0.5f * normalizedSize * sy,  planeSize),
            Vec3f(planeSize, -0.5f * normalizedSize * sy, -planeSize)
        };

        Vec2f planeTexCoords[4] = {
            Vec2f(0.0f, 0.0f),
            Vec2f(0.0f, 1.0f),
            Vec2f(1.0f, 1.0f),
            Vec2f(1.0f, 0.0f)
        };

        Vec3ui planeTriangles[2] = {
            Vec3ui(0, 1, 2),
            Vec3ui(0, 2, 3)
        };

        TriangleGeometry* plane = context->CreateTriangleGeometry();
        plane->SetTriangles(2, planeTriangles, 4, planeVertices);
        plane->SetTexCoords(planeTexCoords);

        MDLMaterial* planeMat = this->LoadMDL(planeMaterial, osprayMDLSource, mdlModulePaths, compilationType, 0, "Plane");
        plane->SetMaterial(planeMat);
        planeMat->Release();

        model->AddGeometry(plane);
        this->releaseLater.insert(plane);


        /*
         * Lights
         */
        VisRTX::DirectionalLight* dirLight1 = context->CreateDirectionalLight();
        dirLight1->SetDirection(Vec3f(-1.0f, -1.0f, -1.0f));
        dirLight1->SetIntensity(1.0f);
        dirLight1->SetVisible(true);
        dirLight1->SetAngularDiameter(5.0f);
        renderer->AddLight(dirLight1);
        dirLight1->Release();


        VisRTX::DirectionalLight* dirLight2 = context->CreateDirectionalLight();
        dirLight2->SetDirection(Vec3f(-1.0f, -1.0f, 1.0f));
        dirLight2->SetIntensity(1.0f);
        dirLight2->SetVisible(true);
        dirLight2->SetAngularDiameter(5.0f);
        renderer->AddLight(dirLight2);
        dirLight2->Release();

        return true;
    }

    void UpdateScene(bool benchmark, BenchmarkPhase benchmarkPhase, float benchmarkTimeDelta, bool pauseAllAnimations, bool& reset) override
    {
    }

    void UpdateGUI(bool& reset) override
    {
    }
};

int main(int argc, char **argv)
{
    SampleMDL mdl;
    mdl.Run("VisRTX Sample: MDL Materials", argc, argv);
    return 0;
}

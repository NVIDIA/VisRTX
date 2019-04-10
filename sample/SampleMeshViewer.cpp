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
 * Example loading a set of OBJ meshes from disk with given MDL materials. This sample supports both instance and class compilation for MDL materials.
 *
 * Usage:  .\visRtxSampleMeshViewer.exe ::OBJMaterial cow.obj cow2.obj ::Principled cow3.obj ::MetallicPaint --plane
 */

const VisRTX::CompilationType compilationType = CompilationType::INSTANCE;
//const VisRTX::CompilationType compilationType = CompilationType::CLASS;


class SampleMeshViewer : public Sample
{
public:
    bool Init(int argc, char **argv) override
    {
        bool printHelp = false;

        bool swapXY = false;

        struct Mesh
        {
            std::string path;
            std::string material;

            std::vector<VisRTX::Vec3f> vertices;
            std::vector<VisRTX::Vec3f> normals;
            std::vector<VisRTX::Vec2f> texcoords;
            std::vector<VisRTX::Vec3ui> triangles;

            VisRTX::MDLMaterial* mdl;
        };

        std::vector<Mesh> meshes;

        std::string currentMaterial = "::OBJMaterial";

        std::string planeMaterial;

        // Parse args
        for (int i = 1; i < argc; ++i)
        {
            std::string arg(argv[i]);
            std::string argNext(i < argc - 1 ? argv[i + 1] : "");

            if (arg == "-h" || arg == "--help")
            {
                printHelp = true;
            }
            else if (arg.find("::") == 0)
            {
                currentMaterial = arg;
            }
            else if (arg == "--plane")
            {
                planeMaterial = currentMaterial;
            }
            else if (arg == "--swapXY")
            {
                swapXY = true;
            }
            else
            {
                Mesh m;
                m.path = arg;
                m.material = currentMaterial;
                meshes.push_back(m);
            }
        }

        // Print help
        if (printHelp || meshes.empty())
        {
            std::cout << "VisRTX mesh loader" << std::endl;
            std::cout << "Usage: visRtxSampleMeshViewer [params]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << " ::$material : Use MDL material with given name for the following meshes (Available: ::OBJMaterial, ::Alloy, ::Metal, ::MetallicPaint, ::CarPaint, ::Principled, ::ThinGlass, ::Glass)" << std::endl;
            std::cout << " $path.obj : Any number of OBJ meshes" << std::endl;
            std::cout << " --plane : Add a ground plane with the current material" << std::endl;
            std::cout << " --swapXY : Swap X/Y axes" << std::endl;
            return false;
        }




        // ---------------------------------------------------------------------------------------------------------------

        this->ambientColor.r = 0.95f;
        this->ambientColor.g = 0.95f;
        this->ambientColor.b = 0.95f;

        this->rotationHorizontal = 0.0f;
        this->rotationVertical = 0.0f;
        this->distance = 2.5f;

        this->numBouncesMin = 1;
        this->numBouncesMax = 8;


        VisRTX::Context* context = VisRTX_GetContext();

        std::cout << "Loading resources..." << std::endl;

        // Load OSPRay materials from memory
        const std::string osprayMDLSource((const char*)OSPRay_mdl, sizeof(OSPRay_mdl));

        VisRTX::Vec3f min, max;
        min.x = min.y = min.z = std::numeric_limits<float>::max();
        max.x = max.y = max.z = -std::numeric_limits<float>::max();

        for (Mesh& mesh : meshes)
        {
            Timer timer;

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;

            std::string warn;
            std::string err;
            bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, mesh.path.c_str());

            if (!err.empty())
            {
                std::cerr << err << std::endl;
                return false;
            }

            for (size_t s = 0; s < shapes.size(); s++)
            {
                size_t index_offset = 0;
                for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
                {
                    int fv = shapes[s].mesh.num_face_vertices[f];

                    if (fv == 3)
                    {
                        uint32_t vertexOffset = (uint32_t)mesh.vertices.size();

                        for (size_t v = 0; v < fv; v++)
                        {
                            // access to vertex
                            tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                            if (idx.vertex_index >= 0)
                            {
                                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

                                mesh.vertices.push_back(VisRTX::Vec3f(vx, vy, vz));

                                // Update aabb
                                min.x = std::min(min.x, vx);
                                min.y = std::min(min.y, vy);
                                min.z = std::min(min.z, vz);

                                max.x = std::max(max.x, vx);
                                max.y = std::max(max.y, vy);
                                max.z = std::max(max.z, vz);
                            }

                            if (idx.normal_index >= 0)
                            {
                                tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                                tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                                tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

                                mesh.normals.push_back(VisRTX::Vec3f(nx, ny, nz));
                            }

                            if (idx.texcoord_index >= 0)
                            {
                                tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                                tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];

                                mesh.texcoords.push_back(VisRTX::Vec2f(tx, ty));
                            }
                        }
                        index_offset += fv;

                        mesh.triangles.push_back(VisRTX::Vec3ui(vertexOffset, vertexOffset + 1, vertexOffset + 2));
                    }
                }
            }

            std::cout << "Load OBJ: " << mesh.path << ": " << timer.GetElapsedMilliseconds() << " ms" << std::endl;

            size_t i = mesh.path.find_last_of("/\\");
            std::string file = (i == std::string::npos) ? mesh.path : mesh.path.substr(i + 1);

            mesh.mdl = this->LoadMDL("::ospray" + mesh.material, osprayMDLSource, {}, compilationType, file);
        }

        // Normalize all vertices
        const float size = std::max(max.x - min.x, std::max(max.y - min.y, max.z - min.z));
        const float sx = (max.x - min.x) / size;
        const float sy = (max.y - min.y) / size;
        const float sz = (max.z - min.z) / size;

        for (Mesh& mesh : meshes)
        {
            bool computeTexCoords = mesh.texcoords.empty();

            for (Vec3f& v : mesh.vertices)
            {
                v.x = (v.x - min.x) / size;
                v.y = (v.y - min.y) / size;
                v.z = (v.z - min.z) / size;

                if (swapXY)
                    std::swap(v.x, v.y);

                if (computeTexCoords)
                {
                    // Just use normalized (XY) coordinate
                    mesh.texcoords.push_back(Vec2f(v.x, v.y));
                }

                // Shift to center
                v.x -= 0.5f * sx;
                v.y -= 0.5f * sy;
                v.z -= 0.5f * sz;
            }
        }

        // Create geometries
        for (Mesh& mesh : meshes)
        {
            VisRTX::TriangleGeometry* geo = context->CreateTriangleGeometry((uint32_t)mesh.triangles.size(), mesh.triangles.data(), (uint32_t)mesh.vertices.size(), mesh.vertices.data(), mesh.normals.empty() ? nullptr : mesh.normals.data());
            geo->SetTexCoords(mesh.texcoords.empty() ? nullptr : mesh.texcoords.data());
            geo->SetMaterial(mesh.mdl);
            mesh.mdl->Release();

            model->AddGeometry(geo);
            this->releaseLater.insert(geo);
        }


        /*
         * Ground plane
         */
        if (!planeMaterial.empty())
        {
            const float planeSize = 2.0f;
            Vec3f planeVertices[4] = {
                Vec3f(-planeSize, -0.5f * sy, -planeSize),
                Vec3f(-planeSize, -0.5f * sy,  planeSize),
                Vec3f(planeSize, -0.5f * sy,  planeSize),
                Vec3f(planeSize, -0.5f * sy, -planeSize)
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

            MDLMaterial* planeMat = this->LoadMDL("::ospray" + planeMaterial, osprayMDLSource, {}, compilationType, "Plane");
            plane->SetMaterial(planeMat);
            planeMat->Release();

            model->AddGeometry(plane);
            this->releaseLater.insert(plane);
        }


        /*
         * Lights
         */
        dirLight1 = context->CreateDirectionalLight();
        dirLight1->SetDirection(Vec3f(-1.0f, -1.0f, -1.0f));
        dirLight1->SetIntensity(this->lightsIntensity);
        dirLight1->SetVisible(true);
        dirLight1->SetAngularDiameter(this->lightsDiameter);
        renderer->AddLight(dirLight1);

        dirLight2 = context->CreateDirectionalLight();
        dirLight2->SetDirection(Vec3f(-1.0f, -1.0f, 1.0f));
        dirLight2->SetIntensity(this->lightsIntensity);
        dirLight2->SetVisible(true);
        dirLight2->SetAngularDiameter(this->lightsDiameter);
        renderer->AddLight(dirLight2);

        dirLight1->Release();
        dirLight2->Release();

        return true;
    }

    void UpdateScene(bool benchmark, BenchmarkPhase benchmarkPhase, float benchmarkTimeDelta, bool pauseAllAnimations, bool& reset) override
    {
        dirLight1->SetIntensity(this->lightsIntensity);
        dirLight1->SetAngularDiameter(this->lightsDiameter);

        dirLight2->SetIntensity(this->lightsIntensity);
        dirLight2->SetAngularDiameter(this->lightsDiameter);
    }

    void UpdateGUI(bool& reset) override
    {
#ifdef VISRTX_SAMPLE_WITH_GLFW
        if (ImGui::CollapsingHeader("Directional Lights"))
        {
            reset |= ImGui::SliderFloat("Intensity##Directional", &lightsIntensity, 0.0f, 10.0f, "%.1f");
            reset |= ImGui::SliderFloat("Angular Diameter##Directional", &lightsDiameter, 0.0f, 50.0f, "%.1f");
        }
#endif
    }

    VisRTX::DirectionalLight* dirLight1;
    VisRTX::DirectionalLight* dirLight2;

    float lightsDiameter = 20.0f;
    float lightsIntensity = 1.0f;
};

int main(int argc, char **argv)
{
    SampleMeshViewer mdl;
    mdl.Run("VisRTX Sample: Mesh Viewer", argc, argv);
    return 0;
}

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


/*
 * Test application to verify the different light sources. Used in conjunction with #define TEST_DIRECT_ONLY/TEST_NEE_ONLY/TEST_MIS in Common.h
 */

class SampleLightTest : public Sample
{
public:
    /*
     * Init
     */
    bool Init(int argc, char **argv) override
    {
        const std::string source = "mdl 1.0;\
            import df::*;\
            export material diffuse()\
            = material(\
                surface: material_surface(\
                    scattering : df::diffuse_reflection_bsdf()\
                )\
            ); ";


        this->numBouncesMin = 1;
        this->numBouncesMax = 3;

        VisRTX::Context* context = VisRTX_GetContext();

        // Spheres
        Vec3f sphereVertices[1] = {
            Vec3f(0.0f, 0.0f, 0.0f)
        };

        float sphereRadii[1] = {
            1.0f,
        };

        SphereGeometry* sphereGeo = context->CreateSphereGeometry();
        sphereGeo->SetSpheres(1, sphereVertices, sphereRadii);

        MDLMaterial* sphereMaterial = context->CreateMDLMaterial("::test::diffuse", source.c_str(), (uint32_t) source.length(), 0, nullptr, CompilationType::INSTANCE);
        sphereMaterial->Compile();

        sphereGeo->SetMaterial(sphereMaterial);
        sphereMaterial->Release();


        // Plane
        Vec3f planeVertices[4] = {
            Vec3f(-2.0f, -1.0f, -2.0f),
            Vec3f(-2.0f, -1.0f,  2.0f),
            Vec3f(2.0f, -1.0f,  2.0f),
            Vec3f(2.0f, -1.0f, -2.0f)
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

        TriangleGeometry* triangleGeo = context->CreateTriangleGeometry();
        triangleGeo->SetTriangles(2, planeTriangles, 4, planeVertices);
        triangleGeo->SetTexCoords(planeTexCoords);

        MDLMaterial* triangleMaterial = context->CreateMDLMaterial("::test::diffuse", source.c_str(), (uint32_t)source.length(), 0, nullptr, CompilationType::INSTANCE);
        triangleMaterial->Compile();
        triangleGeo->SetMaterial(triangleMaterial);
        triangleMaterial->Release();

        model->AddGeometry(sphereGeo);
        model->AddGeometry(triangleGeo);

        this->releaseLater.insert(sphereGeo);
        this->releaseLater.insert(triangleGeo);

        dirLight = context->CreateDirectionalLight();
        dirLight->SetDirection(Vec3f(-1.0f, -1.0f, -1.0f));
        dirLight->SetIntensity(0.7f);

        pointLight = context->CreateSphericalLight();
        pointLight->SetColor(Vec3f(0.0f, 1.0f, 0.0f));

        quadLight = context->CreateQuadLight();
        quadLight->SetColor(Vec3f(1.0f, 0.0f, 0.0f));

        spotLight = context->CreateSpotLight();
        spotLight->SetColor(Vec3f(0.8f, 0.8f, 1.0f));
        spotLight->SetPosition(Vec3f(-1.8f, 1.0f, 1.6f));
        spotLight->SetDirection(Vec3f(0.5f, -1.0f, -0.5f));

        renderer->AddLight(pointLight);
        renderer->AddLight(dirLight);
        renderer->AddLight(quadLight);
        renderer->AddLight(spotLight);

        pointLight->Release();
        dirLight->Release();
        quadLight->Release();
        spotLight->Release();

        // HDRI light
        const uint32_t hdriRes = 1024;
        std::vector<float> tmp(hdriRes * hdriRes * 3);

        for (uint32_t y = 0; y < hdriRes; ++y)
        {
            for (uint32_t x = 0; x < hdriRes; ++x)
            {
                float r = 1.0f;
                float g = 1.0f;
                float b = 1.0f;

                float fx = (float)x;
                float fy = (float)y;
                float halfRes = 0.5f * hdriRes;

                float dx = fx - halfRes;
                float dy = fy - halfRes;

                if (fy >= halfRes && fabs(dx) <= 0.01f * hdriRes)
                {
                    r = 1.0f;
                    g = 0.0f;
                    b = 0.0f;
                }

                if (sqrtf(dx * dx + dy * dy) < 0.03f * hdriRes)
                {
                    r = 0.0f;
                    g = 1.0f;
                    b = 0.0f;
                }

                tmp[3 * (y * hdriRes + x) + 0] = r;
                tmp[3 * (y * hdriRes + x) + 1] = g;
                tmp[3 * (y * hdriRes + x) + 2] = b;
            }
        }

        VisRTX::Texture* hdriTex = context->CreateTexture(VisRTX::Vec2ui(hdriRes, hdriRes), VisRTX::TextureFormat::RGB32F, tmp.data());

        hdriLight = context->CreateHDRILight(hdriTex);
        renderer->AddLight(hdriLight);

        hdriTex->Release();
        hdriLight->Release();


        // Command line args
        this->directionalLightEnabled = false;
        this->pointLightEnabled = false;
        this->quadLightEnabled = false;
        this->spotLightEnabled = false;
        this->hdriLightEnabled = false;

        for (int i = 0; i < argc; ++i)
        {
            std::string param(argv[i]);

            if (param == "point" || param == "sphere" || param == "spherical")
                this->pointLightEnabled = true;
            else if (param == "dir" || param == "directional")
                this->directionalLightEnabled = true;
            else if (param == "quad")
                this->quadLightEnabled = true;
            else if (param == "spot")
                this->spotLightEnabled = true;
            else if (param == "hdri")
                this->hdriLightEnabled = true;
        }

        // Default: all
        if (argc <= 1)
        {
            this->directionalLightEnabled = true;
            this->pointLightEnabled = true;
            //this->quadLightEnabled = true;
            //this->spotLightEnabled = true;
        }

        return true;
    }

    /*
    * Update scene
    */
    void UpdateScene(bool benchmark, BenchmarkPhase benchmarkPhase, float benchmarkTimeDelta, bool pauseAllAnimations, bool& reset) override
    {
        // Point light
        pointLight->SetPosition(Vec3f(0.5f, 0.5f, 1.6f));
        pointLight->SetRadius(pointLightRadius);
        pointLight->SetEnabled(pointLightEnabled);
        pointLight->SetIntensity(pointLightIntensity);
        pointLight->SetVisible(pointLightVisible);

        // Directional light
        dirLight->SetEnabled(directionalLightEnabled);
        dirLight->SetVisible(directionalLightVisible);
        dirLight->SetIntensity(directionalLightIntensity);
        dirLight->SetAngularDiameter(directionalLightAngularDiameter);

        // Quad light
        Vec3f center(0.0f, 0.0f, 1.6f);
        float phi = 20.0f;
        Vec3f U(quadLightSize * cos(phi), 0.0f, quadLightSize * sin(phi));
        Vec3f V(0.0f, quadLightSize, 0.0f);

        Vec3f quadPos;
        quadPos.x = center.x - 0.5f * U.x - 0.5f * V.x;
        quadPos.y = center.y - 0.5f * U.y - 0.5f * V.y;
        quadPos.z = center.z - 0.5f * U.z - 0.5f * V.z;

        Vec3f edge1 = U;
        Vec3f edge2 = V;

        quadLight->SetRect(quadPos, edge1, edge2);
        quadLight->SetTwoSided(quadLightTwoSided);
        quadLight->SetEnabled(quadLightEnabled);
        quadLight->SetVisible(quadLightVisible);
        quadLight->SetIntensity(quadLightIntensity);

        // Spot light
        spotLight->SetEnabled(spotLightEnabled);
        spotLight->SetVisible(spotLightVisible);
        spotLight->SetIntensity(spotLightIntensity);
        spotLight->SetRadius(spotLightRadius);
        spotLight->SetOpeningAngle(spotLightOpeningAngle);
        spotLight->SetPenumbraAngle(spotLightPenumbraAngle);

        // HDRI light
        hdriLight->SetEnabled(hdriLightEnabled);
        hdriLight->SetVisible(hdriLightVisible);
        hdriLight->SetIntensity(hdriLightIntensity);
        hdriLight->SetDirection(hdriLightDir);
        hdriLight->SetUp(hdriLightUp);
    }

    /*
    * Update GUI
    */
    void UpdateGUI(bool& reset) override
    {
        if (ImGui::CollapsingHeader("Directional Light"))
        {
            reset |= ImGui::Checkbox("Enabled##Directional", &directionalLightEnabled);
            if (directionalLightEnabled)
            {
                reset |= ImGui::Checkbox("Visible##Directional", &directionalLightVisible);
                reset |= ImGui::SliderFloat("Intensity##Directional", &directionalLightIntensity, 0.0f, 10.0f, "%.1f");
                reset |= ImGui::SliderFloat("Angular Diameter##Directional", &directionalLightAngularDiameter, 0.0f, 20.0f, "%.1f");
            }
        }

        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Spherical Light"))
        {
            reset |= ImGui::Checkbox("Enabled##Point", &pointLightEnabled);
            if (pointLightEnabled)
            {
                reset |= ImGui::Checkbox("Visible##Point", &pointLightVisible);
                reset |= ImGui::SliderFloat("Radius##Point", &pointLightRadius, 0.0f, 1.0f, "%.3f");
                reset |= ImGui::SliderFloat("Intensity##Point", &pointLightIntensity, 0.0f, 10.0f, "%.1f");
            }
        }

        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Quad Light"))
        {
            reset |= ImGui::Checkbox("Enabled##Quad", &quadLightEnabled);
            if (quadLightEnabled)
            {
                reset |= ImGui::Checkbox("Visible##Quad", &quadLightVisible);
                reset |= ImGui::Checkbox("Two Sided", &quadLightTwoSided);
                reset |= ImGui::SliderFloat("Intensity##Quad", &quadLightIntensity, 0.0f, 10.0f, "%.1f");
                reset |= ImGui::SliderFloat("Size##Quad", &quadLightSize, 0.0f, 1.0f, "%.2f");
            }
        }

        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Spot Light"))
        {
            reset |= ImGui::Checkbox("Enabled##Spot", &spotLightEnabled);
            if (spotLightEnabled)
            {
                reset |= ImGui::Checkbox("Visible##Spot", &spotLightVisible);
                reset |= ImGui::SliderFloat("Intensity##Spot", &spotLightIntensity, 0.0f, 10.0f, "%.1f");
                reset |= ImGui::SliderFloat("Radius##Spot", &spotLightRadius, 0.0f, 1.0f, "%.2f");
                reset |= ImGui::SliderFloat("Opening Angle##Spot", &spotLightOpeningAngle, 0.0f, 180.0f, "%.2f");
                reset |= ImGui::SliderFloat("Penumbra Angle##Spot", &spotLightPenumbraAngle, 0.0f, 0.5f * spotLightOpeningAngle, "%.2f");
            }
        }

        ImGui::Spacing();

        if (ImGui::CollapsingHeader("HDRI Light"))
        {
            reset |= ImGui::Checkbox("Enabled##HDRI", &hdriLightEnabled);
            if (hdriLightEnabled)
            {
                reset |= ImGui::Checkbox("Visible##HDRI", &hdriLightVisible);
                reset |= ImGui::SliderFloat("Intensity##HDRI", &hdriLightIntensity, 0.0f, 10.0f, "%.1f");
                reset |= ImGui::SliderFloat3("Direction##HDRI", &hdriLightDir.x, -1.0f, 1.0f, "%.2f");
                reset |= ImGui::SliderFloat3("Up##HDRI", &hdriLightUp.x, -1.0f, 1.0f, "%.2f");
            }
        }
    }

public:

    // Directional light
    DirectionalLight* dirLight;
    bool directionalLightEnabled = false;
    bool directionalLightVisible = true;
    float directionalLightIntensity = 3.0f;
    float directionalLightAngularDiameter = 5.0f;

    // Point light
    SphericalLight* pointLight;
    bool pointLightEnabled = true;
    bool pointLightVisible = true;
    float pointLightIntensity = 1.0f;
    float pointLightRadius = 0.07f;

    // Quad light
    QuadLight* quadLight;
    bool quadLightEnabled = false;
    bool quadLightVisible = true;
    float quadLightIntensity = 1.0f;
    float quadLightSize = 0.5f;
    bool quadLightTwoSided = true;

    // Spot light
    SpotLight* spotLight;
    bool spotLightEnabled = false;
    bool spotLightVisible = true;
    float spotLightIntensity = 2.0f;
    float spotLightRadius = 0.25f;
    float spotLightOpeningAngle = 40.0f;
    float spotLightPenumbraAngle = 3.0f;

    // HDRI light
    HDRILight* hdriLight;
    bool hdriLightEnabled = false;
    bool hdriLightVisible = true;
    float hdriLightIntensity = 0.2f;
    VisRTX::Vec3f hdriLightDir = VisRTX::Vec3f(0.0f, 0.0f, -1.0f);
    VisRTX::Vec3f hdriLightUp = VisRTX::Vec3f(0.0f, 1.0f, 0.0f);
};

int main(int argc, char **argv)
{
    SampleLightTest basic;
    basic.Run("VisRTX Sample: Light Tests", argc, argv);
    return 0;
}

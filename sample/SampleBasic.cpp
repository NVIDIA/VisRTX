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
 * Example for testing the BasicMaterial in conjunction with various light sources and geometry types.
 */

class SampleBasic : public Sample
{
public:
    /*
     * Init
     */
    bool Init(int argc, char **argv) override
    {
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

        Vec4f sphereColors[1] =
        {
            Vec4f(255.0f / 255.0f, 47.0f / 255.0f, 154.0f / 255.0f, 1.0f)
        };

        SphereGeometry* sphereGeo = context->CreateSphereGeometry();
        sphereGeo->SetSpheres(1, sphereVertices, sphereRadii);
        sphereGeo->SetColors(sphereColors);

        sphereMaterial = context->CreateBasicMaterial();
        sphereGeo->SetMaterial(sphereMaterial);


        // Second smaller spheres
        const uint32_t numSmall = 9;
        Vec3f sphereVertices2[1 + numSmall];
        float sphereRadii2[1 + numSmall];
        Vec4f sphereColors2[1 + numSmall];

        sphereVertices2[0] = Vec3f(1.1f, 0.8f, 1.1f);
        sphereRadii2[0] = 0.4f;
        sphereColors2[0] = Vec4f(115 / 255.0f, 194 / 255.0f, 251 / 255.0f, 1.0f);

        for (uint32_t i = 0; i < numSmall; ++i)
        {
            float angle = (float)i / (float)(numSmall - 1) * 3.1415926535f;

            const float r = 0.3f;
            float x = -1.0f;
            float y = -0.9f + 2.0f * r * sin(angle);
            float z = 1.6f + r * cos(angle);

            sphereVertices2[1 + i] = Vec3f(x, y, z);
            sphereRadii2[1 + i] = 0.1f;
            sphereColors2[1 + i] = Vec4f(1.0f, 1.0f, 1.0f, 1.0f);
        }



        SphereGeometry* sphereGeo2 = context->CreateSphereGeometry();
        sphereGeo2->SetSpheres(1 + numSmall, sphereVertices2, sphereRadii2);
        sphereGeo2->SetColors(sphereColors2);

        VisRTX::Material* materials[1 + numSmall];
        for (int i = 0; i <= numSmall; ++i)
        {
            BasicMaterial* sphereMaterial2 = context->CreateBasicMaterial();
            sphereMaterial2->SetDiffuse(Vec3f(0.4f, 0.4f, 0.4f));

            if (i == 0 || i % 3 != 0)
            {
                sphereMaterial2->SetSpecular(Vec3f(1.0f, 1.0f, 1.0f));
                sphereMaterial2->SetShininess(100.0f);
            }
            else
            {
                sphereMaterial2->SetEmissive(Vec3f(1.0f, 0.65f, 0.0f));
                sphereMaterial2->SetLuminosity(2.0f);
            }

            materials[i] = sphereMaterial2;
        }

        sphereGeo2->SetMaterials(materials);
        for (VisRTX::Material* m : materials)
            m->Release();

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

        BasicMaterial* triangleMaterial = context->CreateBasicMaterial();
        triangleGeo->SetMaterial(triangleMaterial);
        triangleMaterial->Release();


        const uint32_t checkerboardRes = 512;
        const uint32_t tiles = 8;
        const uint32_t tileSize = checkerboardRes / tiles;

        Vec4b* checkerboardPixels = new Vec4b[checkerboardRes * checkerboardRes];

        for (uint32_t y = 0; y < checkerboardRes; ++y)
        {
            for (uint32_t x = 0; x < checkerboardRes; ++x)
            {
                uint32_t tx = x / tileSize;
                uint32_t ty = y / tileSize;

                checkerboardPixels[y * checkerboardRes + x] = ((tx + ty) % 2 == 0) ? Vec4b(20, 20, 20) : Vec4b(255, 255, 255);
            }
        }

        Texture* checkerboard = context->CreateTexture(Vec2ui(checkerboardRes, checkerboardRes), TextureFormat::RGBA8, checkerboardPixels);
        triangleMaterial->SetDiffuseTexture(checkerboard);
        checkerboard->Release();

        delete[] checkerboardPixels;


        // Cylinders
        Vec3f cylinderVertices[4] = {
            Vec3f(1.4f, -1.0f, 1.2f),
            Vec3f(1.4f,  0.3f, 1.2f),
            Vec3f(-0.5f, -0.7f, 1.6f),
            Vec3f(0.5f, -0.7f, 1.6f),
        };

        float cylinderRadii[4] = {
            0.1f, 0.1f,
            0.3f, 0.29f
        };

        Vec4f cylinderColors[4] =
        {
            Vec4f(255 / 255.0f, 0, 0, 1.0f),
            Vec4f(255 / 255.0f, 255 / 255.0f, 0, 1.0f),
            Vec4f(20 / 255.0f, 255 / 255.0f, 20 / 255.0f, 1.0f),
            Vec4f(20 / 255.0f, 220 / 255.0f, 255 / 255.0f, 1.0f)
        };

        Vec2ui cylinders[2] = {
            Vec2ui(0, 1),
            Vec2ui(2, 3)
        };

        cylinderGeo = context->CreateCylinderGeometry();
        cylinderGeo->SetCylinders(2, cylinders, 4, cylinderVertices, cylinderRadii);
        cylinderGeo->SetColors(cylinderColors);

        cylinderMaterial = context->CreateBasicMaterial();
        cylinderGeo->SetMaterial(cylinderMaterial);
        cylinderMaterial->Release();


        // Parameterized spring
        const uint32_t numVertices = 200;
        uint32_t windings = 3;
        float yMin = -1.0f;
        float yMax = 1.0f;
        float radius = 1.2f;
        float thickness = 0.1f;

        Vec2ui springCylinders[numVertices - 1];
        Vec3f springVertices[numVertices];
        float springRadii[numVertices];
        Vec4f springColors[numVertices];
        float springParameterization[numVertices];

        for (uint32_t i = 0; i < numVertices; ++i)
        {
            float s = (float)i / (float)(numVertices - 1);
            float angle = s * 360.0f * windings * piOver180;

            float x = radius * cos(angle);
            float z = radius * sin(angle);
            float y = yMin + s * (yMax - yMin);

            springVertices[i] = Vec3f(x, y, z);

            springRadii[i] = 0.5f * thickness;

            springColors[i] = Vec4f(1.0f, 1.0f, 1.0f, 1.0f);

            springParameterization[i] = s;

            if (i < numVertices - 1)
                springCylinders[i] = Vec2ui(i, i + 1);
        }

        springGeo = context->CreateCylinderGeometry();
        springGeo->SetCylinders(numVertices - 1, springCylinders, numVertices, springVertices, springRadii);
        springGeo->SetColors(springColors);
        springGeo->SetParameterization(springParameterization);

        springMaterial = context->CreateBasicMaterial();
        springMaterial->SetDiffuse(Vec3f(0.0f, 0.0f, 0.0f));
        springMaterial->SetEmissive(Vec3f(0.0f, 0.0f, 1.0f));
        springGeo->SetMaterial(springMaterial);
        springMaterial->Release();


        model->AddGeometry(sphereGeo);
        model->AddGeometry(sphereGeo2);
        model->AddGeometry(triangleGeo);
        model->AddGeometry(cylinderGeo, GeometryFlag::DYNAMIC);
        model->AddGeometry(springGeo);

        this->releaseLater.insert(sphereGeo);
        this->releaseLater.insert(sphereGeo2);
        this->releaseLater.insert(triangleGeo);
        this->releaseLater.insert(cylinderGeo);
        this->releaseLater.insert(springGeo);


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
        spotLight->SetDirection(Vec3f(0.5f, -1.0f, 0.0f));

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
        hdriLight->Release();

        return true;
    }

    /*
    * Update scene
    */
    void UpdateScene(bool benchmark, BenchmarkPhase benchmarkPhase, float benchmarkTimeDelta, bool pauseAllAnimations, bool& reset) override
    {
        // Update lights
        // Point light
        if (benchmark)
        {
            if (benchmarkPhase != STATIC)
            {
                pointLightAngle += pointLightSpeed * benchmarkTimeDelta;
                reset = true;
            }
        }
        else if (animatePointLight && !pauseAllAnimations)
        {
            pointLightAngle += pointLightSpeed * (float)pointLightTimer.GetElapsedSeconds();
            reset = true;
        }
        pointLightTimer.Reset();

        float phi = pointLightAngle * piOver180;
        float x = 1.9f * sin(phi);
        float y = -0.7f;
        float z = 1.6f;

        pointLight->SetPosition(Vec3f(x, y, z));
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
        if (benchmark)
        {
            if (benchmarkPhase != STATIC)
            {
                quadLightAngle += quadLightSpeed * benchmarkTimeDelta;
                reset = true;
            }
        }
        else if (animateQuadLight && !pauseAllAnimations)
        {
            quadLightAngle += quadLightSpeed * (float)quadLightTimer.GetElapsedSeconds();
            reset = true;
        }
        quadLightTimer.Reset();

        Vec3f center(0.0f, 0.0f, 1.6f);

        phi = quadLightAngle * piOver180;
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

        // Sphere material
        sphereMaterial->SetDiffuse(Vec3f(sphereDiffuse, sphereDiffuse, sphereDiffuse));
        sphereMaterial->SetSpecular(Vec3f(sphereSpecular, sphereSpecular, sphereSpecular));
        sphereMaterial->SetShininess(sphereShininess);
        sphereMaterial->SetOpacity(sphereOpacity);

        // Cylinders
        if (!showCylinders && cylindersAdded)
        {
            model->RemoveGeometry(cylinderGeo);
            cylindersAdded = false;
        }
        else if (showCylinders && !cylindersAdded)
        {
            model->AddGeometry(cylinderGeo, GeometryFlag::DYNAMIC);
            cylindersAdded = true;
        }

        cylinderMaterial->SetOpacity(cylinderOpacity);

        // Spring
        springMaterial->SetLuminosity(springLuminosity);

        if (benchmark)
        {
            if (benchmarkPhase != STATIC)
            {
                springTime += benchmarkTimeDelta;
                reset = true;
            }

            springGeo->SetAnimatedParameterization(springAnimation, springTime, springFrequency, springScale);
        }
        else
        {
            if (!pauseAllAnimations)
                springTime += (float)springTimer.GetElapsedSeconds();
            springTimer.Reset();

            springGeo->SetAnimatedParameterization(springAnimation, springTime, springFrequency, springScale);
            if (springAnimation && !pauseAllAnimations)
                reset = true;
        }
    }

    /*
    * Update GUI
    */
    void UpdateGUI(bool& reset) override
    {
        if (ImGui::CollapsingHeader("Sphere"))
        {
            reset |= ImGui::SliderFloat("Diffuse##Sphere", &sphereDiffuse, 0.0f, 1.0f, "%.1f");
            reset |= ImGui::SliderFloat("Specular##Sphere", &sphereSpecular, 0.0f, 1.0f, "%.1f");
            reset |= ImGui::SliderFloat("Shininess##Sphere", &sphereShininess, 0.0f, 100.0f, "%.1f", 1.0f);
            reset |= ImGui::SliderFloat("Opacity##Sphere", &sphereOpacity, 0.0f, 1.0f, "%.1f");
        }

        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Cylinders"))
        {
            reset |= ImGui::Checkbox("Show Cylinders", &showCylinders);
            reset |= ImGui::SliderFloat("Opacity##Cylinders", &cylinderOpacity, 0.0f, 1.0f, "%.1f");
        }

        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Spring"))
        {
            reset |= ImGui::SliderFloat("Luminosity##Spring", &springLuminosity, 0.0f, 10.0f, "%.1f");
            reset |= ImGui::Checkbox("Animated Parameterization", &springAnimation);
            if (springAnimation)
            {
                reset |= ImGui::SliderFloat("Frequency##Spring", &springFrequency, 0.0f, 10.0f, "%.1f");
                reset |= ImGui::SliderFloat("Scale##Spring", &springScale, 0.0f, 10.0f, "%.1f");
            }
        }

        ImGui::Spacing();

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
                reset |= ImGui::SliderFloat("Radius##Point", &pointLightRadius, 0.0f, 0.2f, "%.3f");
                reset |= ImGui::SliderFloat("Intensity##Point", &pointLightIntensity, 0.0f, 10.0f, "%.1f");
                reset |= ImGui::Checkbox("Animate##Point", &animatePointLight);
                if (animatePointLight)
                {
                    ImGui::SliderFloat("Speed##SphericalLight", &pointLightSpeed, 0.0f, 1000.0f, "%.1f deg/s");
                }
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
                reset |= ImGui::Checkbox("Animate Quad Light", &animateQuadLight);
                if (animateQuadLight)
                {
                    ImGui::SliderFloat("Speed##QuadLight", &quadLightSpeed, 0.0f, 1000.0f, "%.1f deg/s");
                }
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
    bool directionalLightEnabled = true;
    bool directionalLightVisible = true;
    float directionalLightIntensity = 0.4f;
    float directionalLightAngularDiameter = 5.0f;

    // Point light
    SphericalLight* pointLight;
    bool pointLightEnabled = true;
    bool pointLightVisible = true;
    float pointLightIntensity = 5.0f;
    float pointLightRadius = 0.07f;
    bool animatePointLight = true;
    float pointLightAngle = 120.0f;
    float pointLightSpeed = 90.0f; // deg/sec
    Timer pointLightTimer;

    // Quad light
    QuadLight* quadLight;
    bool quadLightEnabled = true;
    bool quadLightVisible = true;
    float quadLightIntensity = 5.0f;
    float quadLightSize = 0.5f;
    bool quadLightTwoSided = true;
    bool animateQuadLight = true;
    float quadLightAngle = 0.0f;
    float quadLightSpeed = 180.0f; // deg/sec
    Timer quadLightTimer;

    // Spot light
    SpotLight* spotLight;
    bool spotLightEnabled = true;
    bool spotLightVisible = true;
    float spotLightIntensity = 10.0f;
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

    // Sphere
    BasicMaterial* sphereMaterial;
    float sphereDiffuse = 1.0f;
    float sphereSpecular = 1.0f;
    float sphereShininess = 15.0f;
    float sphereOpacity = 1.0f;

    // Cylinders
    CylinderGeometry* cylinderGeo;
    BasicMaterial* cylinderMaterial;
    bool cylindersAdded = true;
    bool showCylinders = true;
    float cylinderOpacity = 0.8f;

    // Parameterized spring
    CylinderGeometry* springGeo;
    BasicMaterial* springMaterial;
    bool springAnimation = true;
    float springLuminosity = 1.0f;
    float springFrequency = 0.5f;
    float springScale = 5.0f;
    Timer springTimer;
    float springTime = 0.0f;
};

int main(int argc, char **argv)
{
    SampleBasic basic;
    basic.Run("VisRTX Sample: Basic Materials", argc, argv);
    return 0;
}

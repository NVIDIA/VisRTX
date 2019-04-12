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
 * Example volumes. This sample supports both instance and class compilation for MDL materials.
 */

const VisRTX::CompilationType compilationType = CompilationType::INSTANCE;
//const VisRTX::CompilationType compilationType = CompilationType::CLASS;


class SampleVolume : public Sample
{
public:
	bool Init(int argc, char** argv) override
	{
		this->ambientColor.r = 1.0f;
		this->ambientColor.g = 1.0f;
		this->ambientColor.b = 1.0f;

		this->rotationHorizontal = 0.0f;
		this->rotationVertical = 20.0f;
		this->distance = 2.9f;

		this->numBouncesMin = 2;
		this->numBouncesMax = 50;


		VisRTX::Context* context = VisRTX_GetContext();

		std::cout << "Loading resources..." << std::endl;

		// Load OSPRay materials from memory
		const std::string osprayMDLSource((const char*)OSPRay_mdl, sizeof(OSPRay_mdl));

		// Spheres
		VisRTX::Vec3f centers[] =
		{
			Vec3f(-0.5f, -0.3f, 0.0f),
			Vec3f(0.5f, 0.0f, 0.0f),			
			Vec3f(1.3f, -0.1f, 0.0f)
		};
		float radii[] =
		{
			0.4f,
			0.375f,			
			0.3f
		};
		SphereGeometry* sphere = context->CreateSphereGeometry();
		sphere->SetSpheres(sizeof(centers) / sizeof(Vec3f), centers, radii);

		// Assign different materials
		MDLMaterial* mat1 = this->LoadMDL("::ospray::Glass", osprayMDLSource, {}, compilationType, 2, "Sphere 1"); // <-- higher priority than box
		mat1->SetParameterColor("attenuationColor", Vec3f(0.15f, 1.0f, 0.0f));
		mat1->SetParameterFloat("attenuationDistance", 0.5f);
		mat1->SetParameterFloat("eta", 3.0f);
		mat1->Compile();

		MDLMaterial* mat2 = this->LoadMDL("::ospray::Glass", osprayMDLSource, {}, compilationType, 2, "Sphere 2"); // <-- lower priority than box
		mat2->SetParameterColor("attenuationColor", Vec3f(0.15f, 1.0f, 0.0f));
		mat2->SetParameterFloat("attenuationDistance", 0.5f);
		mat2->SetParameterFloat("eta", 1.5f);
		mat2->Compile();

		MDLMaterial* mat3 = this->LoadMDL("::ospray::Glass", osprayMDLSource, {}, compilationType, 0, "Sphere 3");
		mat3->SetParameterColor("attenuationColor", Vec3f(0.15f, 1.0f, 0.0f));
		mat3->SetParameterFloat("attenuationDistance", 0.5f);
		mat3->SetParameterFloat("eta", 1.5f);
		mat3->Compile();

		VisRTX::Material* sphereMaterials[] =
		{
			mat1,
			mat2,
			mat3
		};
		
		sphere->SetMaterials(sphereMaterials);
		mat1->Release();
		mat2->Release();
		mat3->Release();

		model->AddGeometry(sphere);
		this->releaseLater.insert(sphere);


		// Box
		VisRTX::Vec3f vertices[8] =
		{
			VisRTX::Vec3f(-0.5f, -0.5f, 0.5f),
				VisRTX::Vec3f(0.5f, -0.5f, 0.5f),
				VisRTX::Vec3f(0.5f, -0.5f, -0.5f),
				VisRTX::Vec3f(-0.5f, -0.5f, -0.5f),
				VisRTX::Vec3f(-0.5f, 0.5f, 0.5f),
				VisRTX::Vec3f(0.5f, 0.5f, 0.5f),
				VisRTX::Vec3f(0.5f, 0.5f, -0.5f),
				VisRTX::Vec3f(-0.5f, 0.5f, -0.5f),
		};

		VisRTX::Vec3ui triangles[12] =
		{
			VisRTX::Vec3ui(0, 2, 1),
			VisRTX::Vec3ui(0, 3, 2),
			VisRTX::Vec3ui(1,2,6),
			VisRTX::Vec3ui(1,6,5),
			VisRTX::Vec3ui(0,1,5),
			VisRTX::Vec3ui(0, 5, 4),
			VisRTX::Vec3ui(0, 4, 3),
			VisRTX::Vec3ui(3,4,7),
			VisRTX::Vec3ui(3,7,2),
			VisRTX::Vec3ui(2,7,6),
			VisRTX::Vec3ui(4, 5, 7),
			VisRTX::Vec3ui(7,5,6),
		};

		VisRTX::TriangleGeometry * box = context->CreateTriangleGeometry(12, triangles, 8, vertices, nullptr);

		MDLMaterial* boxMat = this->LoadMDL("::ospray::Glass", osprayMDLSource, {}, compilationType, 1, "Box"); // <- priority 1
		boxMat->SetParameterColor("attenuationColor", Vec3f(0.0f, 0.559f, 1.0f));
		boxMat->SetParameterFloat("attenuationDistance", 0.528f);
		boxMat->SetParameterFloat("eta", 1.33f);
		boxMat->Compile();

		box->SetMaterial(boxMat);
		boxMat->Release();

		model->AddGeometry(box);
		this->releaseLater.insert(box);

		// Ground plane
		const float y = -0.6f;

		const float planeSize = 2.0f;
		Vec3f planeVertices[4] = {
			Vec3f(-planeSize, y, planeSize),
			Vec3f(planeSize, y,  planeSize),
			Vec3f(planeSize, y,  -planeSize),
			Vec3f(-planeSize, y, -planeSize)
		};

		Vec2f planeTexCoords[4] = {
			Vec2f(0.0f, 0.0f),
			Vec2f(1.0f, 0.0f),
			Vec2f(1.0f, 1.0f),
			Vec2f(0.0f, 1.0f)
		};

		Vec3ui planeTriangles[2] = {
			Vec3ui(0, 1, 2),
			Vec3ui(0, 2, 3)
		};

		TriangleGeometry * plane = context->CreateTriangleGeometry();
		plane->SetTriangles(2, planeTriangles, 4, planeVertices);
		plane->SetTexCoords(planeTexCoords);

		std::string texMatSrc = "mdl 1.0;\
			import df::*;\
			import tex::*;\
			import state::*;\
			export material Plane()\
				= material(\
					surface: material_surface(\
						scattering : df::diffuse_reflection_bsdf(tint : tex::lookup_color(texture_2d(\"NVIDIA_Logo.jpg\"), float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y)))\
					)\
				);";

		MDLMaterial * planeMat = this->LoadMDL("::test::Plane", texMatSrc, {}, compilationType, 0, "Plane");
		plane->SetMaterial(planeMat);
		planeMat->Release();

		model->AddGeometry(plane);
		this->releaseLater.insert(plane);


		// Lights
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

		// Make sure GUI shows the values that we've set manually above
		for (MaterialGUI& mat : this->materials)
			mat.LoadCurrentValues();

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

	VisRTX::DirectionalLight * dirLight1;
	VisRTX::DirectionalLight * dirLight2;

	float lightsDiameter = 8.0f;
	float lightsIntensity = 1.0f;
	};

int main(int argc, char** argv)
{
	SampleVolume vol;
	vol.Run("VisRTX Sample: Volumes", argc, argv);
	return 0;
}

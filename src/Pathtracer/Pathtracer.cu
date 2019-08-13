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


#include "Common.h"
#include "Config.h"

#include "Light.h"
#include "PackNormals.h"
#include "Sphere.h"
#include "BasicMaterial.h"
#include "Pick.h"
#include "Sample.h"



rtDeclareVariable(optix::uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(optix::uint2, launchDim, rtLaunchDim, );

rtBuffer<LaunchParameters> launchParameters;

rtBuffer<BasicMaterialParameters> basicMaterialParameters;
rtBuffer<MDLMaterialParameters> mdlMaterialParameters;

rtBuffer<Light> lights;
rtBuffer<ClippingPlane> clippingPlanes;

rtDeclareVariable(rtObject, topObject, , );

rtDeclareVariable(PathtracePRD, prd, rtPayload, );
rtDeclareVariable(OcclusionPRD, shadowPrd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );


RT_FUNCTION constexpr float origin() { return 1.0f / 32.0f; }
RT_FUNCTION constexpr float float_scale() { return 1.0f / 65536.0f; }
RT_FUNCTION constexpr float int_scale() { return 256.0f; }

// Normal points outward for rays exiting the surface, else is flipped.
RT_FUNCTION optix::float3 offsetRay(const optix::float3& p, const optix::float3& n)
{
    optix::int3 of_i = optix::make_int3(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z);

    optix::float3 p_i = optix::make_float3(
        int_as_float(float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        int_as_float(float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        int_as_float(float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return optix::make_float3(fabsf(p.x) < origin() ? p.x + float_scale() * n.x : p_i.x,
        fabsf(p.y) < origin() ? p.y + float_scale() * n.y : p_i.y,
        fabsf(p.z) < origin() ? p.z + float_scale() * n.z : p_i.z);
}


RT_FUNCTION bool SampleLight(const Light& light, PathtracePRD& prd, optix::Ray& ray, optix::float3& L, optix::float3& edf_over_pdf, float& pdf)
{
	const optix::float3 normal = prd.normal;

	float  Ldist; // distance towards light source
	float weight = 1.0f;
	pdf = PDF_DIRAC; // Default

	// Directional light
	if (light.type == Light::DIRECTIONAL)
	{
		Ldist = 1e8f; // infinitely far awar
		L = -light.dir;

		// Area light
		if (light.angularDiameter > 0.0f)
		{
			const optix::float2 sampleAreaLight = Sample2D(prd.randState);
			L = SampleCone(L, light.angularDiameter, sampleAreaLight);
		}

		edf_over_pdf = light.color;
		pdf = light.pdf; // Precomputed light.pdf = (this->angularDiameter > 0.0f) ? (1.0f / (2.0f * M_PIf * (1.0f - cosAngle))) : PDF_DIRAC;
	}

	// Point light
	else if (light.type == Light::POSITIONAL)
	{
		optix::float3 lightPos = light.pos;
		Ldist = optix::length(lightPos - ray.origin);
		float invDist = 1.0f / Ldist;
		L = (lightPos - ray.origin) * invDist;

		const float sinTheta = light.radius * invDist;

		if (light.radius > 0.0f && sinTheta > 0.005f)
		{
			// sample surface of sphere as seen by hit point -> cone of directions
			// for very small cones treat as point light, because float precision is not good enough
			if (sinTheta < 1.0f)
			{
				// Compute angular diameter of sphere cap visible from point
				const optix::float2 sampleAreaLight = Sample2D(prd.randState);
				const float angle = asinf(light.radius * invDist);
				L = SampleCone(L, angle, sampleAreaLight);

				const float cosAngle = cos(angle);
				pdf = 1.0f / (2.0f * M_PIf * (1.0f - cosAngle));

				// Compute distance to actual surface of light source
				float t0, t1;
				if (intersectSphere(lightPos, light.radius, ray.origin, ray.direction, t0, t1))
				{
					Ldist = t0;
					invDist = 1.0f / Ldist;
				}
			}
			else
			{
				// Inside
				pdf = M_1_PIf;
			}
		}

		edf_over_pdf = light.color * invDist * invDist;
	}

	// Quad light
	else if (light.type == Light::QUAD)
	{
		const optix::float2 sampleAreaLight = Sample2D(prd.randState);
		optix::float3 lightPos = light.pos + sampleAreaLight.x * light.edge1 + sampleAreaLight.y * light.edge2;
		Ldist = optix::length(lightPos - ray.origin);
		L = (lightPos - ray.origin) / Ldist;

		const optix::float3 emissionDir = optix::normalize(optix::cross(light.edge1, light.edge2));

		// One sided
		if (light.twoSided == 0)
		{
			if (optix::dot(emissionDir, L) >= 0.0f)
				weight = 0.0f;
		}

		if (light.pdf > 0.0f)
		{
			const float cosTheta = fabs(optix::dot(-L, emissionDir));
			pdf = Ldist * Ldist * light.pdf / cosTheta; // Precomputed light.pdf = area > 0.0f ? (1.0f / area) : PDF_DIRAC;
		}

		edf_over_pdf = weight * light.color;
		if (pdf > 0.0f)
			edf_over_pdf /= pdf;
	}

	// Spot light
	else if (light.type == Light::SPOT)
	{
		// Determine light position
		optix::float3 lightPos = light.pos;
		if (light.radius > 0.0f)
		{
			const optix::float2 sampleAreaLight = Sample2D(prd.randState);
			lightPos = SampleDisk(light.pos, light.dir, light.radius, sampleAreaLight);
		}

		Ldist = optix::length(lightPos - ray.origin);
		L = (lightPos - ray.origin) / Ldist;

		// Compute intensity based on angle between light vector and light direction
		const float angle = acosf(optix::dot(-L, light.dir));

		if (light.pdf > 0.0f)
		{
			const float cosAngle = cos(angle);
			pdf = light.pdf * Ldist * Ldist * fabs(cosAngle); // Precomputed light.pdf = area > 0.0f ? (1.0f / area) : PDF_DIRAC;
		}

		if (angle > light.outerAngle)
		{
			weight = 0.0f;
		}
		else
		{
			if (angle >= light.innerAngle)
			{
				weight = 1.0f - (angle - light.innerAngle) / (light.outerAngle - light.innerAngle);
			}
		}

		edf_over_pdf = light.color * weight / (Ldist * Ldist);
	}

	// Ambient light (never sampled)
	//else if (light.type == Light::AMBIENT)
	//{
	//    Ldist = 1e8f; // infinitely far awar

	//    const optix::float2 sampleAreaLight = Sample2D(prd.randState);
	//    optix::cosine_sample_hemisphere(sampleAreaLight.x, sampleAreaLight.y, L);

	//    pdf = L.z * M_1_PIf;

	//    optix::Onb onb(normal);
	//    onb.inverse_transform(L);
	//    L = optix::normalize(L);
	//}

	// HDRI light (never sampled)
	//else if (light.type == Light::HDRI)
	//{
	//    Ldist = 1e8f; // infinitely far awar

	//    const optix::float2 sampleAreaLight = Sample2D(prd.randState);
	//    optix::cosine_sample_hemisphere(sampleAreaLight.x, sampleAreaLight.y, L);

	//    pdf = L.z * M_1_PIf;

	//    optix::Onb onb(normal);
	//    onb.inverse_transform(L);
	//    L = optix::normalize(L);

	//    const optix::float3 Z = light.dir;
	//    const optix::float3 X = optix::cross(light.up, Z);
	//    const optix::float3 Y = optix::cross(Z, X);

	//    const optix::float3 transformedL = L.x * X + L.y * Y + L.z * Z;

	//    float theta = atan2f(transformedL.x, transformedL.z);
	//    float phi = M_PIf * 0.5f - acosf(transformedL.y);
	//    float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	//    float v = 0.5f * (1.0f + sinf(phi));
	//    Lcolor *= make_float3(optix::rtTex2D<float4>(light.texture, u, v));
	//}

	// Check if front-facing
	const float ndl = optix::dot(normal, L);
	if (ndl <= 0.0f)
		return false;

	// Cast shadow ray
	OcclusionPRD shadowPrd;
	shadowPrd.occlusion = optix::make_float3(1.0f);
	shadowPrd.randState = prd.randState;

    // Calculate shadow ray origin with offset to avoid self intersection
	optix::Ray shadow_ray = optix::make_Ray(offsetRay(prd.hitPoint, prd.geometricNormal), L, OCCLUSION_RAY_TYPE, 0.0f, Ldist);
	rtTrace(/*launchParameters[0].*/topObject, shadow_ray, shadowPrd);

	if (fmaxf(shadowPrd.occlusion) <= 0.0f)
		return false;

	edf_over_pdf *= shadowPrd.occlusion;

	return true;
}

RT_FUNCTION bool EvaluateLight(const Light & light, PathtracePRD & prd, optix::Ray & ray, optix::float3 & edf, float& pdf)
{
	edf = optix::make_float3(0.0f);
	pdf = -1.0f;

	const float Ldist = prd.tHit; // distance from ray.origin to hit on light surface (assuming ray.direction is normalized)

	// Directional light
	if (light.type == Light::DIRECTIONAL)
	{
		if (light.angularDiameter > 0.0f)
		{
			const float angle = acosf(optix::dot(ray.direction, -light.dir));
			if (angle <= light.angularDiameter)
			{
				edf = light.color * light.pdf;
				pdf = light.pdf; // Precomputed light.pdf = (this->angularDiameter > 0.0f) ? (1.0f / (2.0f * M_PIf * (1.0f - cosAngle))) : PDF_DIRAC;

				return true;
			}
		}
	}

	// Point light
	else if (light.type == Light::POSITIONAL)
	{
		if (light.radius > 0.0f && Ldist > 0.0f)
		{
			const float invDist = 1.0f / Ldist;
			if (Ldist > light.radius)
			{
				const float angle = atanf(light.radius * invDist);
				const float cosAngle = cos(angle);
				pdf = 1.0f / (2.0f * M_PIf * (1.0f - cosAngle));
			}
			else
			{
				// Inside
				pdf = M_1_PIf;
			}

			edf = light.color * pdf * invDist * invDist;


			return true;
		}
	}

	// Quad light
	else if (light.type == Light::QUAD)
	{
		if (light.pdf > 0.0f) // if area > 0
		{
			const float invDist = 1.0f / Ldist;

			optix::float3 lightPos = ray.origin + prd.tHit * ray.direction;
			const optix::float3 L = (lightPos - ray.origin) * invDist;

			const optix::float3 emissionDir = optix::normalize(optix::cross(light.edge1, light.edge2));

			const float cosTheta = fabs(optix::dot(-L, emissionDir));
			pdf = Ldist * Ldist * light.pdf / cosTheta; // Precomputed light.pdf = area > 0.0f ? (1.0f / area) : PDF_DIRAC;

			edf = light.color;

			return true;
		}
	}

	// Spot light
	else if (light.type == Light::SPOT)
	{
		if (light.radius > 0.0f)
		{
			const float angle = acosf(optix::dot(-ray.direction, light.dir));
			const float cosAngle = cos(angle);

			float weight = 1.0f;
			if (angle > light.outerAngle)
			{
				weight = 0.0f;
			}
			else
			{
				if (angle >= light.innerAngle)
				{
					weight = 1.0f - (angle - light.innerAngle) / (light.outerAngle - light.innerAngle);
				}
			}

			float p = light.pdf * fabs(cosAngle); // Precomputed light.pdf = area > 0.0f ? (1.0f / area) : PDF_DIRAC;
			edf = light.color * weight * p;
			pdf = p * (Ldist * Ldist);

			return true;
		}
	}

	// Ambient light
	else if (light.type == Light::AMBIENT)
	{
		edf = light.color;
		pdf = 0.0f; // never sampled

		return true;
	}

	// HDRI light
	else if (light.type == Light::HDRI && light.texture != RT_TEXTURE_ID_NULL)
	{
		const optix::float3 X = -light.dir;
		const optix::float3 Y = optix::cross(X, light.up);
		const optix::float3 Z = optix::cross(X, Y);

		const optix::float3 transformedRayDir = optix::make_float3(optix::dot(ray.direction, X), optix::dot(ray.direction, Y), optix::dot(ray.direction, Z));

		const float u = atan2f(transformedRayDir.y, transformedRayDir.x) * (0.5f * M_1_PIf);
		const float v = acosf(transformedRayDir.z) * M_1_PIf;

		edf = light.color * make_float3(optix::rtTex2D<float4>(light.texture, u, v));
		pdf = 0.0f; // never sampled

		return true;
	}

	return false;
}


RT_FUNCTION void Swap(VolumeStackElement & a, VolumeStackElement & b)
{
	const MaterialId material = a.material;
	const optix::float3 ior = a.ior;
	const optix::float3 abs = a.absorptionCoefficient;

	a.material = b.material;
	a.ior = b.ior;
	a.absorptionCoefficient = b.absorptionCoefficient;

	b.material = material;
	b.ior = ior;
	b.absorptionCoefficient = abs;
}

RT_FUNCTION void PushVolume(const MaterialId material, const optix::float3 & ior, const optix::float3 & absorptionCoefficient, VolumeStackElement stack[VOLUME_MAX_STACK_SIZE], int& stackSize)
{
	// Don't crash when stack is full
	if (stackSize >= VOLUME_MAX_STACK_SIZE)
		return;

	// Append at end and trickle down based on priority
	stack[stackSize].material = material;
	stack[stackSize].ior = ior;
	stack[stackSize].absorptionCoefficient = absorptionCoefficient;
	++stackSize;

	for (int i = stackSize - 1; i > 0; --i)
	{
		// Makes sure a.priority <= b.priority
		if ((stack[i - 1].material & MATERIAL_PRIORITY_MASK) <= (stack[i].material & MATERIAL_PRIORITY_MASK))
			break;
		else
			Swap(stack[i - 1], stack[i]);
	}
}

RT_FUNCTION bool PopVolume(const MaterialId material, VolumeStackElement stack[VOLUME_MAX_STACK_SIZE], int& stackSize)
{
	// Search from end, remove, let others trickly down
	int index = -1;
	for (int i = stackSize - 1; i >= 0; --i)
		if (stack[i].material == material)
			index = i;

	if (index < 0)
		return false;

	for (int i = index; i < stackSize - 1; ++i)
		Swap(stack[i], stack[i + 1]);

	--stackSize;

	return true;
}


RT_FUNCTION float powerHeuristic(const float a, const float b)
{
	const float t = a * a;
	return t / (t + b * b);
}


__device__ const float identity[16] = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f };

__device__ const mi::neuraylib::Resource_data res_data = {
	NULL,
	NULL
};

RT_FUNCTION bool SampleMaterial(PathtracePRD & prd, optix::Ray & ray, VolumeStackElement stack[VOLUME_MAX_STACK_SIZE], int& stackSize)
{
	const uint32_t materialIndex = prd.material & MATERIAL_INDEX_MASK;
	const int numLights = launchParameters[0].numLightsDirect;

	// TODO There is a lot of redundant code between basic and MDL materials -> unify!

	/*
	 * Basic material
	 */
	if (prd.material & BASIC_MATERIAL_BIT)
	{
		const BasicMaterialParameters& parameters = basicMaterialParameters[materialIndex];

		// Cut-out opacity
		float opacity = BasicMaterial_Opacity(parameters, prd.texCoord);
		opacity *= prd.color.w;

		float opacitySample = 0.0f;
		if (opacity < 1.0f)
			opacitySample = Sample1D(prd.randState);

		if (opacitySample < opacity)
		{
			// Init
			optix::Onb onb(prd.normal);

			BasicMaterialState state;
			state.normal = prd.normal;
			state.geometricNormal = prd.geometricNormal;
			state.tangentU = onb.m_tangent;
			state.tangentV = onb.m_binormal;
			state.wo = -ray.direction;

			BasicMaterial_Init(prd, parameters, state);

			// Emission
			const optix::float3 emissive = state.emissive * parameters.luminosity * prd.animationFactor; // *(1.e-4f + (1.f - 1.e-4f) * optix::dot(normal, -ray.direction)); // Disabled: visual hack to make streamlines look better
			prd.radiance += prd.alpha * emissive;

			// Next event estimation
#ifndef TEST_DIRECT_ONLY
			optix::float3 L;
			optix::float3 lightEdf_over_pdf;
			float lightPdf;

			float lightFactor = 1.0f;
			int lightStart = 0;
			int lightEnd = numLights - 1;
			if (launchParameters[0].sampleAllLights <= 0)
			{
				lightFactor = numLights;
				prd.lastLightPdfFactor = numLights > 0 ? 1.0f / lightFactor : 1.0f;
				const float lightSample = Sample1D(prd.randState);
				const int i = optix::clamp(static_cast<int>(floorf(lightSample * numLights)), 0, numLights - 1);
				lightStart = i;
				lightEnd = i;
			}

			for (int i = lightStart; i <= lightEnd; ++i)
			{
				if (SampleLight(lights[i], prd, ray, L, lightEdf_over_pdf, lightPdf))
				{
					state.wi = L;
					BasicMaterial_Eval(parameters, state);

					if (0.0f < state.pdf && isNotNull(state.bsdf))
					{
#ifdef TEST_NEE_ONLY
						const float misWeight = 1.0f;
#endif
#if !defined(TEST_DIRECT_ONLY) && !defined(TEST_NEE_ONLY)
						const float misWeight = (lightPdf <= 0.0f) ? 1.0f : powerHeuristic(lightPdf * prd.lastLightPdfFactor, state.pdf);
#endif

						const optix::float3 radiance = prd.alpha * state.bsdf * lightEdf_over_pdf * lightFactor * misWeight; // state.bsdf contains: bsdf * dot(normal, L)
						prd.radiance += clampRadiance(prd.depth, launchParameters[0].fireflyClampingIndirect, radiance);
					}
				}
			}
#endif


			// Sample
			const optix::float3 sample = Sample3D(prd.randState);

			bool absorp = !BasicMaterial_Sample(parameters, state, sample);

			prd.lastPdf = state.pdf;

			if (absorp)
				return false;

			ray.direction = optix::normalize(state.wi);
			prd.alpha *= state.bsdf_over_pdf;
		}

		if (opacity < 1e-3f)
			prd.numCutoutOpacityHits = prd.numCutoutOpacityHits + 1;
	}

	/*
	 * MDL material
	 */
	else if (prd.material & MDL_MATERIAL_BIT)
	{
		MDLMaterialParameters& parameters = mdlMaterialParameters[materialIndex];

		optix::Onb onb(prd.normal);
		optix::float3 texCoords = optix::make_float3(prd.texCoord, 0.0f);

		optix::float4 texture_results[MDL_MAX_TEXTURES];

		mi::neuraylib::Shading_state_material state;
		state.normal = prd.normal;
		state.geom_normal = prd.geometricNormal;
		state.position = ray.origin;
		state.animation_time = 0.0f;
		state.text_coords = &texCoords;
		state.tangent_u = &onb.m_tangent;
		state.tangent_v = &onb.m_binormal;
		state.text_results = texture_results;
		state.ro_data_segment = NULL;
		state.world_to_object = (float4*)& identity;
		state.object_to_world = (float4*)& identity;
		state.object_id = 0;

		char* argBlock = NULL;
		if (parameters.hasArgBlock)
			argBlock = parameters.argBlock;

		// Check if hit should be ignored
		bool thinwalled = false;
		parameters.thinwalled(&thinwalled, &state, &res_data, NULL, argBlock);

		optix::float3 ior;
		parameters.ior(&ior, &state, &res_data, NULL, argBlock);

		optix::float3 absorptionCoefficient;
		parameters.absorption(&absorptionCoefficient, &state, &res_data, NULL, argBlock);

		uint32_t surroundingVolumePriority = 0;
		if (stackSize >= 1)
		{
			surroundingVolumePriority = MATERIAL_PRIORITY_MASK & stack[stackSize - 1].material; // highest priority element at the end
		}

		bool skipBSDF = false;
		bool updateVolumeStack = false;
		if (!thinwalled && (surroundingVolumePriority > (MATERIAL_PRIORITY_MASK & prd.material)))
		{
			skipBSDF = true;
			updateVolumeStack = true;
		}

		if (!skipBSDF)
		{
			// Cut-out opacity
			float opacity;
			parameters.opacity(&opacity, &state, &res_data, NULL, argBlock);
			opacity *= prd.color.w;

			float opacitySample = 0.0f;
			if (opacity < 1.0f)
				opacitySample = Sample1D(prd.randState);

			if (opacitySample < opacity)
			{
				// IOR
				union // Put the BSDF data structs into a union to reduce number of memory writes
				{
					mi::neuraylib::Bsdf_sample_data sample;
					mi::neuraylib::Bsdf_evaluate_data evaluate;
					mi::neuraylib::Bsdf_pdf_data pdf;
				} data;

				// Select incident ior based on current volume stack
				data.sample.ior1 = optix::make_float3(1.0f);
				if (stackSize >= 1)
				{
					data.sample.ior1 = stack[stackSize - 1].ior; // highest priority element at the end
				}

				// Select outgoing ior
				// Entry: Outgoing IOR is from hit material
				if (prd.frontFacing)
				{
					data.sample.ior2 = ior;
				}
				// Exit: Outgoing IOR is from surrounding volume
				else
				{
					if (stackSize >= 2)
						data.sample.ior2 = stack[stackSize - 2].ior;
					else
						data.sample.ior2 = optix::make_float3(1.0f);
				}

				//rtPrintf("IOR: %f -> %f\n", data.sample.ior1.x, data.sample.ior2.x);


				// Init BSDF
				data.sample.k1 = optix::normalize(-ray.direction);

				parameters.init(&state, &res_data, NULL, argBlock);

				// Next event estimation
#ifndef TEST_DIRECT_ONLY
				optix::float3 L;
				optix::float3 lightEdf_over_pdf;
				float lightPdf;

				float lightFactor = 1.0f;
				int lightStart = 0;
				int lightEnd = numLights - 1;
				if (launchParameters[0].sampleAllLights <= 0)
				{
					lightFactor = numLights;
					prd.lastLightPdfFactor = numLights > 0 ? 1.0f / lightFactor : 1.0f;
					const float lightSample = Sample1D(prd.randState);
					const int i = optix::clamp(static_cast<int>(floorf(lightSample * numLights)), 0, numLights - 1);
					lightStart = i;
					lightEnd = i;
				}

				for (int i = lightStart; i <= lightEnd; ++i)
				{
					if (SampleLight(lights[i], prd, ray, L, lightEdf_over_pdf, lightPdf))
					{
						data.evaluate.k2 = L;
						parameters.evaluate(&data.evaluate, &state, &res_data, NULL, argBlock);

						if (0.0f < data.evaluate.pdf && isNotNull(data.evaluate.bsdf))
						{
#ifdef TEST_NEE_ONLY
							const float misWeight = 1.0f;
#endif
#if !defined(TEST_DIRECT_ONLY) && !defined(TEST_NEE_ONLY)
							const float misWeight = (lightPdf <= 0.0f) ? 1.0f : powerHeuristic(lightPdf * prd.lastLightPdfFactor, data.evaluate.pdf);
#endif

							const optix::float3 radiance = prd.alpha * data.evaluate.bsdf * lightEdf_over_pdf * lightFactor * misWeight; //  data.evaluate.bsdf contains: bsdf * dot(normal, k2)
							prd.radiance += clampRadiance(prd.depth, launchParameters[0].fireflyClampingIndirect, radiance);
						}
					}
				}
#endif

				// Sample BSDF
				data.sample.xi = Sample3D(prd.randState);
				parameters.sample(&data.sample, &state, &res_data, NULL, argBlock);

				prd.lastPdf = data.sample.pdf;

				if (data.sample.event_type == mi::neuraylib::BSDF_EVENT_ABSORB)
					return false;

				ray.direction = optix::normalize(data.sample.k2);
				prd.alpha *= data.sample.bsdf_over_pdf;

				if (!thinwalled && (data.sample.event_type & mi::neuraylib::BSDF_EVENT_TRANSMISSION))
					updateVolumeStack = true;
			}

			if (opacity < 1e-3f)
				prd.numCutoutOpacityHits = prd.numCutoutOpacityHits + 1;
		}

		// Update volume stack
		if (updateVolumeStack)
		{
			// Entry event
			if (prd.frontFacing)
			{
				//rtPrintf("Entry: %d\n", prd.material);
				PushVolume(prd.material, ior, absorptionCoefficient, stack, stackSize);
			}
			// Exit event
			else
			{
				//rtPrintf("Exit: %d\n", prd.material);

				if (!PopVolume(prd.material, stack, stackSize))
				{
					// Special case: Material was not on stack (e.g., when camera is IN the volume)
					// Handle absorption
					prd.alpha *= expf(-prd.tHit * absorptionCoefficient);
				}
			}
		}
	}

    // Offset new ray origin based on ray direction
    ray.origin = offsetRay(prd.hitPoint, (optix::dot(ray.direction, prd.geometricNormal) >= 0.0f) ? prd.geometricNormal : -prd.geometricNormal);

	return true;
}


RT_FUNCTION void Pathtrace(const float3& rayOrigin, const float3& rayDirection, RandState* randState)
{
#ifdef VISRTX_USE_DEBUG_EXCEPTIONS
#ifdef PRINT_PIXEL_X	

	rtPrintf("\n\n--- New frame ---\n");

	int x = PRINT_PIXEL_X;
	int y = PRINT_PIXEL_Y;
	int d = 20;
	if ((launchIndex.x == x - d && launchIndex.y >= y - d && launchIndex.y <= y + d) ||
		(launchIndex.x == x + d && launchIndex.y >= y - d && launchIndex.y <= y + d) ||
		(launchIndex.y == y - d && launchIndex.x >= x - d && launchIndex.x <= x + d) ||
		(launchIndex.y == y + d && launchIndex.x >= x - d && launchIndex.x <= x + d))
		return;
#endif
#endif

	optix::Ray ray = optix::make_Ray(
		rayOrigin,
		rayDirection,
		RADIANCE_RAY_TYPE,
		0.0f,
		RT_DEFAULT_MAX
	);

	PathtracePRD prd;
	prd.randState = randState;
	prd.radiance = make_float3(0.0f);
	prd.alpha = make_float3(1.0f);
	prd.depth = 0;
	prd.lastPdf = PDF_DIRAC;
	prd.lastLightPdfFactor = 1.0f;
	prd.numCutoutOpacityHits = 0;

	float primaryDepth;

	VolumeStackElement stack[VOLUME_MAX_STACK_SIZE];
	int stackSize = 0;

	/*
	 * Pathtracing loop
	 */
	while (prd.depth < launchParameters[0].numBouncesMax)
	{
#ifdef TEST_NEE_ONLY
		if (prd.depth >= 1)
			break;
#endif
		// Trace ray
		prd.depth = prd.depth;
		prd.lightEdf = optix::make_float3(0.0f, 0.0f, 0.0f);
		prd.lightPdf = PDF_DIRAC;
		prd.light = false;

#if OPTIX_VERSION_MAJOR >= 6
        const RTrayflags rayFlags = (launchParameters[0].disableAnyHit > 0) ? RT_RAY_FLAG_DISABLE_ANYHIT : RT_RAY_FLAG_NONE;
		rtTrace(/*launchParameters[0].*/topObject, ray, prd, RT_VISIBILITY_ALL, rayFlags);
#else
        rtTrace(/*launchParameters[0].*/topObject, ray, prd);
#endif

		// Store primary hit depth (can't store albedo here because it's set/modified by the material)
		if (prd.depth == 0)
			primaryDepth = prd.tHit;

		// Terminate path if light/environment hit
		if (prd.light || RT_DEFAULT_MAX == prd.tHit)
		{
			if (prd.lightPdf >= 0.0f)
			{
				// Direct light hit
#ifdef TEST_DIRECT_ONLY
				const float misWeight = 1.0f;
				prd.radiance += clampRadiance(prd.depth, fireflyClampingIndirect, prd.alpha * prd.lightEdf * misWeight);
#endif

#if !defined(TEST_DIRECT_ONLY) && !defined(TEST_NEE_ONLY)
				const float misWeight = (prd.lastPdf <= 0.0f) ? 1.0f : powerHeuristic(prd.lastPdf, prd.lightPdf * prd.lastLightPdfFactor);
				prd.radiance += clampRadiance(prd.depth, launchParameters[0].fireflyClampingIndirect, prd.alpha * prd.lightEdf * misWeight);
#endif
			}

			break;
		}		
        else
        {
            ray.origin = prd.hitPoint;
        }

#if defined(TEST_DIRECT_ONLY) || defined(TEST_MIS)
		if (prd.depth >= 1)
			break;
#endif

		//rtPrintf("HIT: %f, %f, %f (front facing:%d)\n", ray.origin.x, ray.origin.y, ray.origin.z, prd.frontFacing);

		// Volumetric absorption
		if (stackSize > 0)
		{
			prd.alpha *= expf(-prd.tHit * stack[stackSize - 1].absorptionCoefficient);
		}

		// Sample material
		bool kill = !SampleMaterial(prd, ray, stack, stackSize);

		// Absorption / fixed cut off
		if (kill || fmaxf(prd.alpha) < launchParameters[0].alphaCutoff)
			break;

		// Unbiased Russian Roulette path termination
		if ((launchParameters[0].numBouncesMin + prd.numCutoutOpacityHits) <= prd.depth) // Start termination after a minimum number of bounces.
		{
			const float probability = fmaxf(prd.alpha);

			if (probability < Sample1D(randState)) // Paths with lower probability to continue are terminated earlier.
				break;
			prd.alpha /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
		}

		// Make sure we have a valid ray
		if (isnan(ray.direction.x) || isnan(ray.direction.y) || isnan(ray.direction.z) || isinf(ray.direction.x) || isinf(ray.direction.y) || isinf(ray.direction.z))
			break;

		++prd.depth;
	}

	/*
	 * Output / denoising
	 */
	 // Primary misses need to be transparent so ParaView remote rendering correctly blends with environment color at client-side
	float alpha = 1.0;
	if (!launchParameters[0].writeBackground && primaryDepth >= RT_DEFAULT_MAX)
		alpha = 0.0f;

	// Clamp radiance to prevent fireflies (note: can break direct light evaluation for small lights)
	optix::float4 color = make_float4(clampRadiance(1, launchParameters[0].fireflyClampingDirect, prd.radiance), alpha);

	// Accumulate
	if (launchParameters[0].frameNumber == 0)
	{
		//launchParameters[0].accumulationBuffer[pixel] = color;

		if (launchParameters[0].clipMin >= 0.0f)
			launchParameters[0].depthBuffer[launchIndex] = (primaryDepth < launchParameters[0].clipMin ? 1.0 : (primaryDepth - launchParameters[0].clipMin) * launchParameters[0].clipDiv);
		else
			launchParameters[0].depthBuffer[launchIndex] = primaryDepth;
	}
	else
	{
		const float a = 1.0f / (float)(launchParameters[0].frameNumber + 1);
		optix::float4 old_color = launchParameters[0].accumulationBuffer[launchIndex];
		color = optix::lerp(old_color, color, a);
	}

	launchParameters[0].accumulationBuffer[launchIndex] = color;

	// Tone mapping
	optix::float3 ldrColor = optix::make_float3(color);
	if (launchParameters[0].toneMapping > 0)
	{
		ldrColor = launchParameters[0].invWhitePoint * launchParameters[0].colorBalance * ldrColor;
		ldrColor *= (ldrColor * optix::make_float3(launchParameters[0].burnHighlights) + optix::make_float3(1.0f)) / (ldrColor + optix::make_float3(1.0f));

		float luminance = optix::dot(ldrColor, optix::make_float3(0.3f, 0.59f, 0.11f));
		ldrColor = optix::lerp(optix::make_float3(luminance), ldrColor, launchParameters[0].saturation); // This can generate negative values for saturation > 1.0f!
		ldrColor = optix::fmaxf(optix::make_float3(0.0f), ldrColor); // Prevent negative values.

		luminance = optix::dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
		if (luminance < 1.0f)
		{
			const float3 crushed = optix::make_float3(powf(ldrColor.x, launchParameters[0].crushBlacks), powf(ldrColor.y, launchParameters[0].crushBlacks), powf(ldrColor.z, launchParameters[0].crushBlacks));
			ldrColor = optix::lerp(crushed, ldrColor, sqrtf(luminance));
			ldrColor = optix::fmaxf(optix::make_float3(0.0f), ldrColor); // Prevent negative values.
		}
		ldrColor = optix::make_float3(powf(ldrColor.x, launchParameters[0].invGamma), powf(ldrColor.y, launchParameters[0].invGamma), powf(ldrColor.z, launchParameters[0].invGamma));
	}

#ifdef VISRTX_USE_DEBUG_EXCEPTIONS
	// DAR DEBUG Highlight numerical errors.
	if (isnan(ldrColor.x) || isnan(ldrColor.y) || isnan(ldrColor.z))
	{
		ldrColor = make_float3(1000000.0f, 0.0f, 0.0f); // super red
	}
	else if (isinf(ldrColor.x) || isinf(ldrColor.y) || isinf(ldrColor.z))
	{
		ldrColor = make_float3(0.0f, 1000000.0f, 0.0f); // super green
	}
	else if (ldrColor.x < 0.0f || ldrColor.y < 0.0f || ldrColor.z < 0.0f)
	{
		ldrColor = make_float3(0.0f, 0.0f, 1000000.0f); // super blue
	}
#endif

	// Clamp
	ldrColor = fminf(ldrColor, make_float3(1.0f));

	if (launchParameters[0].writeFrameBuffer)
		launchParameters[0].frameBuffer[launchIndex] = optix::make_float4(ldrColor, color.w);

	if (launchParameters[0].writeUcharFrameBuffer)
		launchParameters[0].ucharFrameBuffer[launchIndex] = optix::make_uchar4(ldrColor.x * 255.0f, ldrColor.y * 255.0f, ldrColor.z * 255.0f, color.w * 255.0f);
}


RT_PROGRAM void BufferCast()
{
	optix::float4 c = fminf(launchParameters[0].frameBuffer[launchIndex], optix::make_float4(1.0f)); // Make sure denoiser output is in [0,1] range
	launchParameters[0].ucharFrameBuffer[launchIndex] = optix::make_uchar4(c.x * 255.0f, c.y * 255.0f, c.z * 255.0f, c.w * 255.0f);
}


RT_PROGRAM void RayGen()
{
	optix::float2 invScreen = 1.0f / make_float2(launchParameters[0].width, launchParameters[0].height);
	optix::float2 p = (launchParameters[0].imageBegin + launchParameters[0].imageSize * optix::make_float2(launchIndex) * invScreen) * 2.0f - 1.0f;

	// Sampling
	// 2D pixel
	// 2D DOF
	// Per bounce: 3D BSDF, 1D light selection, 2D area light, 1D cutout, 1D Russian roulette = 8D
	// (512 - 2) / 8 = 63 bounces
	RandState randState;
	InitSampler(&randState, launchIndex.y * launchParameters[0].width + launchIndex.x, launchParameters[0].frameNumber);

	const optix::float2 sampleScreen = Sample2D(&randState);
	const optix::float2 pixel = p + sampleScreen * launchParameters[0].imageSize * invScreen * 2.0f;

	optix::float3 rayOrigin;
	optix::float3 rayDirection;

	if (launchParameters[0].cameraType == PERSPECTIVE_CAMERA)
	{
		rayOrigin = launchParameters[0].pos;
		rayDirection = optix::normalize(pixel.x * launchParameters[0].U + pixel.y * launchParameters[0].V + launchParameters[0].W);

		// Depth of field
		if (launchParameters[0].focalDistance > 0.0f && launchParameters[0].apertureRadius > 0.0f)
		{
			const optix::float3 focalPoint = launchParameters[0].pos + launchParameters[0].focalDistance * rayDirection;

			// Uniform sampling in aperture
			const optix::float2 sampleDOF = Sample2D(&randState);
			const float rho = sqrtf(sampleDOF.x);
			const float phi = sampleDOF.y * 2.0f * M_PIf;
			float sinPhi, cosPhi;
			sincos(phi, &sinPhi, &cosPhi);

			const float dx = launchParameters[0].apertureRadius * rho * cosPhi;
			const float dy = launchParameters[0].apertureRadius * rho * sinPhi;

			rayOrigin = launchParameters[0].pos + (dx * launchParameters[0].U + dy * launchParameters[0].V);
			rayDirection = optix::normalize(focalPoint - rayOrigin);
		}
	}
	else if (launchParameters[0].cameraType == ORTHOGRAPHIC_CAMERA)
	{
		rayOrigin = launchParameters[0].pos + 0.5f * (pixel.x * launchParameters[0].orthoWidth * launchParameters[0].U + pixel.y * launchParameters[0].orthoHeight * launchParameters[0].V);
		rayDirection = launchParameters[0].W;
	}

	// Run pathtracer
	Pathtrace(rayOrigin, rayDirection, &randState);
}


rtDeclareVariable(optix::float3, hitPoint, attribute hitPoint, );
rtDeclareVariable(optix::float4, color, attribute color, );
rtDeclareVariable(optix::float3, normal, attribute normal, );
rtDeclareVariable(optix::float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(optix::float2, texCoord, attribute texCoord, );
rtDeclareVariable(int, primIndex, attribute primIndex, );
rtDeclareVariable(float, animationValue, attribute animationValue, );
rtDeclareVariable(MaterialId, material, attribute material, );

rtDeclareVariable(MaterialId, geometryMaterial, , );

rtDeclareVariable(int, animateSurface, , );
rtDeclareVariable(float, animationTime, , );
rtDeclareVariable(float, animationFrequency, , );
rtDeclareVariable(float, animationScale, , );

rtBuffer<Light> light;


RT_PROGRAM void ClosestHit()
{
	prd.frontFacing = optix::dot(geometricNormal, ray.direction) < 0.0f;

	if (prd.frontFacing)
	{
		prd.geometricNormal = geometricNormal;
		prd.normal = normal;
	}
	else
	{
		prd.geometricNormal = -geometricNormal;
		prd.normal = -normal;
	}

	// Make sure shading normal does not flip
	if (optix::dot(prd.normal, ray.direction) > 0.0f)
		prd.normal = prd.geometricNormal;

	prd.material = (material != MATERIAL_NULL) ? material : geometryMaterial;

    prd.hitPoint = hitPoint;
	prd.tHit = tHit;
	prd.color = color;
	prd.texCoord = texCoord;


	prd.animationFactor = 1.0f;
	if (animateSurface)
	{
		float phase = animationTime * animationFrequency;
		phase -= floorf(phase);
		float t = animationValue * animationScale - phase;
		t = 3.0f * (t - floorf(t));
		prd.animationFactor = (t <= 0.5f) ? 0.3f + 2.0f * fabsf(sinf(M_PIf * t)) : 0.3f;
	}
}


RT_PROGRAM void AnyHit()
{
    // Clipping
    for (int i = 0; i < launchParameters[0].numClippingPlanes; ++i)
    {
        ClippingPlane& plane = launchParameters[0].clippingPlanesBuffer[i];

        if (prd.depth == 0 || plane.primaryRaysOnly == 0)
        {
            if (optix::dot(plane.coefficients, optix::make_float4(ray.origin + tHit * ray.direction, 1.0f)) < 0.0f)
            {
                rtIgnoreIntersection();
                return;
            }
        }
    }
}


RT_PROGRAM void AnyHitOcclusion()
{
    // Clipping
    for (int i = 0; i < launchParameters[0].numClippingPlanes; ++i)
    {
        ClippingPlane& plane = launchParameters[0].clippingPlanesBuffer[i];

        if (plane.primaryRaysOnly == 0)
        {
            if (optix::dot(plane.coefficients, optix::make_float4(ray.origin + tHit * ray.direction, 1.0f)) < 0.0f)
            {
                rtIgnoreIntersection();
                return;
            }
        }
    }


	const float EPSILON = 1e-3f;

	// Flip geometric and shadowing normal towards viewer
	const optix::float3 geometricNormalForward = optix::faceforward(geometricNormal, -ray.direction, geometricNormal);
	optix::float3 normalForward = optix::faceforward(normal, -ray.direction, geometricNormal);

	if (optix::dot(normalForward, ray.direction) > 0.0f)
		normalForward = geometricNormalForward;

	MaterialId mat = (material != MATERIAL_NULL) ? material : geometryMaterial;
	const uint32_t materialIndex = mat & MATERIAL_INDEX_MASK;

	// Evaluate MDL material
	if (mat & BASIC_MATERIAL_BIT)
	{
		const BasicMaterialParameters& parameters = basicMaterialParameters[materialIndex];

		float opacity = BasicMaterial_Opacity(parameters, texCoord);
		opacity = opacity * color.w;

		if (opacity <= EPSILON)
		{
			rtIgnoreIntersection();
			return;
		}

		shadowPrd.occlusion = shadowPrd.occlusion * (opacity * parameters.transparencyFilterColor + make_float3(1.0f - opacity));
	}
	else if (mat & MDL_MATERIAL_BIT)
	{
		MDLMaterialParameters& parameters = mdlMaterialParameters[materialIndex];

		optix::Onb onb(normal);
		optix::float3 texCoords = optix::make_float3(texCoord, 0.0f);

		optix::float4 texture_results[MDL_MAX_TEXTURES];

		mi::neuraylib::Shading_state_material state;
		state.normal = normalForward;
		state.geom_normal = geometricNormalForward;
		state.position = ray.origin;
		state.animation_time = 0.0f;
		state.text_coords = &texCoords;
		state.tangent_u = &onb.m_tangent;
		state.tangent_v = &onb.m_binormal;
		state.text_results = texture_results;
		state.ro_data_segment = NULL;
		state.world_to_object = (float4*)& identity;
		state.object_to_world = (float4*)& identity;
		state.object_id = 0;

		char* argBlock = NULL;
		if (parameters.hasArgBlock)
			argBlock = parameters.argBlock;

		// Cut-out opacity
		float opacity;
		parameters.opacity(&opacity, &state, &res_data, NULL, argBlock);
		opacity = opacity * color.w;

		if (opacity <= EPSILON)
		{
			rtIgnoreIntersection();
			return;
		}

		// BSDF sampling to check for transmission
		parameters.init(&state, &res_data, NULL, argBlock);

		mi::neuraylib::Bsdf_sample_data data;

		data.ior1 = optix::make_float3(1.0f);
		data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
		//data.ior2 = make_float3(1.f);
		data.k1 = optix::normalize(-ray.direction);

		data.xi = Sample3D(shadowPrd.randState);
		parameters.sample(&data, &state, &res_data, NULL, argBlock);

		if (data.event_type & mi::neuraylib::BSDF_EVENT_TRANSMISSION)
		{
			shadowPrd.occlusion = shadowPrd.occlusion * (opacity * optix::clamp(data.bsdf_over_pdf, 0.0f, 1.0f) + make_float3(1.0f - opacity));
		}
		else
		{
			shadowPrd.occlusion = shadowPrd.occlusion * optix::make_float3(1.0f - opacity);
		}
	}

	// Terminate if full occlusion
	if (fmaxf(shadowPrd.occlusion) <= EPSILON)
		rtTerminateRay();
}

RT_PROGRAM void LightClosestHit()
{
	prd.tHit = tHit;
	prd.light = true;

	EvaluateLight(light[0], prd, ray, prd.lightEdf, prd.lightPdf);
}

RT_PROGRAM void LightAnyHit()
{
	const Light& l = light[0];

	// Ignore hit for primary rays if light is invisible
	if (prd.depth <= 0 && l.visible <= 0)
	{
		rtIgnoreIntersection();
		return;
	}

	// Backface culling of light geometries
	if (l.twoSided <= 0)
	{
		const bool frontFacing = optix::dot(normal, ray.direction) < 0.0f;
		if (!frontFacing)
			rtIgnoreIntersection();
	}
}

RT_PROGRAM void LightAnyHitOcclusion()
{
	rtIgnoreIntersection(); // Light geometries don't cast shadows
}


RT_PROGRAM void Miss()
{
	prd.tHit = RT_DEFAULT_MAX;

	// Miss lights
	for (int i = 0; i < launchParameters[0].numLightsMiss; ++i)
	{
		const Light& l = lights[i];

		optix::float3 edf;
		float pdf;
		if (prd.depth > 0 || l.visible > 0)
		{
			if (EvaluateLight(l, prd, ray, edf, pdf))
			{
				prd.lightEdf += edf;
				prd.lightPdf += pdf;
				prd.light = true;
			}
		}
	}
}


// ------- Picking ---------

rtBuffer<PickStruct> pickResult;

rtDeclareVariable(float3, rayOrigin, , );
rtDeclareVariable(float3, rayDirection, , );

rtDeclareVariable(PickStruct, pick, rtPayload, );

rtDeclareVariable(int, id, , );



RT_PROGRAM void RayGenPick()
{
    optix::Ray ray = optix::make_Ray(rayOrigin, rayDirection, PICK_RAY_TYPE, 0.0f, RT_DEFAULT_MAX);

    PickStruct pick;
    pick.geometryId = 0;
    pick.lightId = 0;
    pick.primIndex = 0;
    pick.t = 0.0f;

#if OPTIX_VERSION_MAJOR >= 6
    const RTrayflags rayFlags = (launchParameters[0].disableAnyHit > 0) ? RT_RAY_FLAG_DISABLE_ANYHIT : RT_RAY_FLAG_NONE;
    rtTrace(/*launchParameters[0].*/topObject, ray, pick, RT_VISIBILITY_ALL, rayFlags);
#else
    rtTrace(/*launchParameters[0].*/topObject, ray, pick);
#endif

    pickResult[0] = pick;
}

RT_PROGRAM void ClosestHitPick()
{
    pick.geometryId = id;
    pick.primIndex = primIndex;
    pick.t = tHit;
}

RT_PROGRAM void LightClosestHitPick()
{
	pick.lightId = light[0].id;
	pick.t = tHit;
}

RT_PROGRAM void MissPick()
{
	// Visible directional lights
	for (int i = 0; i < launchParameters[0].numLightsMiss; ++i)
	{
		const Light light = lights[i];

		if (light.visible && light.type == Light::DIRECTIONAL && light.angularDiameter > 0.0f)
		{
			const float angle = acosf(optix::dot(ray.direction, -light.dir));
			if (angle <= light.angularDiameter)
			{
				pick.lightId = light.id;
				break;
			}
		}
	}

	pick.t = RT_DEFAULT_MAX;
}



// ------- Misc ---------

RT_PROGRAM void Exception()
{
#ifdef VISRTX_USE_DEBUG_EXCEPTIONS
	rtPrintExceptionDetails();
#endif
}

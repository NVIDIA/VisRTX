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


// Perfect inlined clone of the following MDL material. Code based on MDL's libbsdf.

//export material OBJMaterial(
//    uniform color Kd = color(0.8f),
//    uniform texture_2d map_Kd = texture_2d(),
//    uniform color Ks = color(0.0f),
//    uniform texture_2d map_Ks = texture_2d(),
//    uniform float Ns = 10.0f,
//    uniform texture_2d map_Ns = texture_2d(),
//    uniform float d = 1.0f,
//    uniform texture_2d map_d = texture_2d(),
//    uniform color Tf = color(0.0f),
//    uniform texture_2d map_Bump = texture_2d())
//    = let{
//        float4 diff_alpha = tex::texture_isvalid(map_Kd) ? texlookup_float4(map_Kd) : make_float4(Kd);
//        color diff = color(diff_alpha.x, diff_alpha.y, diff_alpha.z);
//        float alpha = diff_alpha.w;
//        color spec = tex::texture_isvalid(map_Ks) ? texlookup_color(map_Ks) : Ks;
//        float specExp = tex::texture_isvalid(map_Ns) ? texlookup_float(map_Ns) : Ns;
//        float max_sum = optix::max_value(diff + spec + Tf);
//        float w = max_sum > 1.0f ? 1.0f / max_sum : 1.0f;
//} in material(
//    surface: material_surface(
//        scattering : df::color_normalized_mix(
//            df::color_bsdf_component[](
//                df::color_bsdf_component(weight: w * diff, component : df::diffuse_reflection_bsdf()),
//                df::color_bsdf_component(weight: w * spec, component : df::simple_glossy_bsdf(mode : df::scatter_reflect, roughness_u : optix::sqrt(2.0 / specExp))),
//                df::color_bsdf_component(weight: w * Tf, component : df::specular_bsdf(mode : df::scatter_reflect_transmit))
//                )
//        )
//    ),
//    thin_walled: true,
//    ior : color(1.0f),
//    geometry : material_geometry(
//        normal : tex::texture_isvalid(map_Bump) ? texlookup_normal(map_Bump) : state::normal(),
//        cutout_opacity : alpha * (tex::texture_isvalid(map_d) ? texlookup_float(map_d) : d)
//    )
//);




#include "Common.h"


struct BasicMaterialState
{
    // Initialized state
    optix::float3 diffuse;
    optix::float3 specular;
    optix::float3 emissive;

    // BSDF input
    optix::float3 normal;
    optix::float3 geometricNormal;
    optix::float3 tangentU;
    optix::float3 tangentV;
    optix::float3 wo;               // Outgoing direction, to observer, in world space.

    // BSDF output
    optix::float3 wi;               // Incoming direction, to light, in world space.
    optix::float3 bsdf_over_pdf;    // BSDF sample throughput, pre-multiplied f_over_pdf = bsdf.f * fabsf(dot(wi, ns) / bsdf.pdf; 
    float pdf;                      // The last BSDF sample's pdf, tracked for multiple importance sampling.
    optix::float3 bsdf;

    // Weights and probabilities
    optix::float3 weightDiffuse;
    optix::float3 weightSpecular;
    optix::float3 weightTransmissive;
    optix::float3 weightSum;

    float pDiffuse;
    float pSpecular;
    float pTransmissive;
};







// ---------- Utils ------------

enum BSDF_event_flags
{
    BSDF_EVENT_DIFFUSE = 1,
    BSDF_EVENT_GLOSSY = 1 << 1,
    BSDF_EVENT_SPECULAR = 1 << 2,
    BSDF_EVENT_REFLECTION = 1 << 3,
    BSDF_EVENT_TRANSMISSION = 1 << 4
};

// type of events created by BSDF importance sampling
enum BSDF_event_type
{
    BSDF_EVENT_ABSORB = 0,
    BSDF_EVENT_DIFFUSE_REFLECTION = BSDF_EVENT_DIFFUSE | BSDF_EVENT_REFLECTION,
    BSDF_EVENT_DIFFUSE_TRANSMISSION = BSDF_EVENT_DIFFUSE | BSDF_EVENT_TRANSMISSION,
    BSDF_EVENT_GLOSSY_REFLECTION = BSDF_EVENT_GLOSSY | BSDF_EVENT_REFLECTION,
    BSDF_EVENT_GLOSSY_TRANSMISSION = BSDF_EVENT_GLOSSY | BSDF_EVENT_TRANSMISSION,
    BSDF_EVENT_SPECULAR_REFLECTION = BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION,
    BSDF_EVENT_SPECULAR_TRANSMISSION = BSDF_EVENT_SPECULAR | BSDF_EVENT_TRANSMISSION
};

enum scatter_mode {
    scatter_reflect,
    scatter_transmit,
    scatter_reflect_transmit
};


RT_FUNCTION float Luminance(const optix::float3& v)
{
    return 0.212671f * v.x + 0.715160f * v.y + 0.072169f * v.z;
}

// compute half vector (convention: pointing to outgoing direction, like shading normal)
RT_FUNCTION optix::float3 compute_half_vector(
    const optix::float3 &k1,
    const optix::float3 &k2,
    const optix::float3 &shading_normal,
    const optix::float2 &ior,
    const float nk1,
    const float nk2,
    const bool transmission,
    const bool thin_walled)
{
    float3 h;
    if (transmission)
    {
        if (thin_walled)
            h = k1 + (shading_normal * (nk2 + nk2) + k2); // use corresponding reflection direction
        else
        {
            h = k2 * ior.y + k1 * ior.x; // points into thicker medium
            if (ior.y > ior.x)
                h *= -1.0f; // make pointing to outgoing direction's medium
        }
    }
    else
        h = k1 + k2;
    return optix::normalize(h);
}

// compute refraction direction
RT_FUNCTION optix::float3 refract(
    const optix::float3 &k,    // direction (pointing from surface)
    const optix::float3 &n,    // normal
    const float b,	// (reflected side IOR) / (transmitted side IOR)
    const float nk,	// dot(n, k)
    bool &total_reflection)
{
    const float refract = b * b * (1.0f - nk * nk);
    total_reflection = (refract > 1.0f);
    return total_reflection ? (n * (nk + nk) - k)
        : optix::normalize((-k * b + n * (b * nk - sqrtf(1.0f - refract))));
}

/// Returns the positive fractional part of \p s.
RT_FUNCTION float frac(float x)
{
    if (x >= 0.0f)
        return x - floorf(x);
    else
        return 1.0f + x - floorf(x);
}

// Fresnel equation for an equal mix of polarization
RT_FUNCTION float ior_fresnel(
    const float eta,	// refracted / reflected ior
    const float kh)     // cosine between of angle normal/half-vector and direction
{
    float costheta = 1.0f - (1.0f - kh * kh) / (eta * eta);
    if (costheta < 0.0f)
        return 1.0f;
    costheta = sqrtf(costheta); // refracted angle cosine

    const float n1t1 = kh;
    const float n1t2 = costheta;
    const float n2t1 = kh * eta;
    const float n2t2 = costheta * eta;
    const float r_p = (n1t2 - n2t1) / (n1t2 + n2t1);
    const float r_o = (n1t1 - n2t2) / (n1t1 + n2t2);
    const float fres = 0.5f * (r_p * r_p + r_o * r_o);

    return optix::clamp(fres, 0.0f, 1.0f);
}

// evaluate anisotropic Phong half vector distribution on the non-projected hemisphere
RT_FUNCTION float hvd_phong_eval(
    const optix::float2 &exponent,
    const float nh,     // dot(shading_normal, h)
    const float ht,     // dot(x_axis, h)
    const float hb)     // dot(z_axis, h)
{
    const float p = nh > 0.99999f
        ? 1.0f
        : powf(nh, (exponent.x * ht * ht + exponent.y * hb * hb) / (1.0f - nh * nh));
    return sqrtf((exponent.x + 1.0f) * (exponent.y + 1.0f)) * (float)(0.5 / M_PIf) * p;
}

// sample half vector according to anisotropic Phong distribution
RT_FUNCTION optix::float3 hvd_phong_sample(
    const optix::float2 &samples,
    const optix::float2 &exponent)
{
    const float sy4 = samples.x*4.0f;
    const float cosupper = cosf((float)M_PIf * frac(sy4));

    const float2 e = optix::make_float2(exponent.x + 1.0f, exponent.y + 1.0f);

    const float eu1mcu = e.x*(1.0f - cosupper);
    const float ev1pcu = e.y*(1.0f + cosupper);
    const float t = eu1mcu + ev1pcu;

    const float tt = (powf(1.0f - samples.y, -t / (e.x*e.y)) - 1.0f) / t;
    const float tttv = sqrtf(ev1pcu*tt);
    const float tttu = sqrtf(eu1mcu*tt);

    return optix::normalize(optix::make_float3(
        ((samples.x < 0.75f) && (samples.x >= 0.25f)) ? -tttv : tttv,
        1.0f,
        ((samples.x >= 0.5f) ? -tttu : tttu)));
}

// clamp roughness values such that the numerics for glossy BSDFs don't fall apart close to
// the perfect specular case
RT_FUNCTION float clamp_roughness(const float roughness)
{
    return fmax(roughness, 0.001f); // magic.
}

// convert roughness values to a similar Phong-style distribution exponent
RT_FUNCTION optix::float2 roughness_to_exponent(const float roughness_u, const float roughness_v)
{
    return optix::make_float2(2.0f / (roughness_u * roughness_u), 2.0f / (roughness_v * roughness_v));
}







// ---------- Diffuse BSDF ------------

RT_FUNCTION bool diffuse_sample(
    const optix::float3& normal,
    const optix::float3& geometricNormal,
    const optix::float3& tangentU,
    const optix::float3& tangentV,
    const optix::float3& tint,
    const optix::float3& sample,
    optix::float3& wi,
    optix::float3& bsdf_over_pdf,
    float& pdf
)
{
    optix::float3 cosh;
    optix::cosine_sample_hemisphere(sample.x, sample.y, cosh);

    wi = optix::normalize(tangentU * cosh.x + tangentV * cosh.y + normal * cosh.z);

    if (cosh.z <= 0.0f || optix::dot(wi, geometricNormal) <= 0.0f)
        return false;

    bsdf_over_pdf = tint;
    pdf = cosh.z * (float)(1.0 / M_PIf);

    return true;
}

RT_FUNCTION void diffuse_evaluate(
    const optix::float3& normal,
    const optix::float3& tint,
    const optix::float3& wi,
    optix::float3& bsdf,
    float& pdf
)
{
    const float nk2 = fmax(optix::dot(wi, normal), 0.0f);
    pdf = nk2 * (float)(1.0f / M_PIf);
    bsdf = pdf * tint;
}

RT_FUNCTION float diffuse_pdf(
    const optix::float3& normal,
    const optix::float3& wi
)
{
    const float nk2 = fmax(optix::dot(wi, normal), 0.0f);
    const float pdf = nk2 * (float)(1.0f / M_PIf);
    return pdf;
}






// ---------- Simple Glossy BSDF ------------

// Cook-Torrance style v-cavities masking term
RT_FUNCTION float microfacet_mask_v_cavities(
    const float nh, // abs(dot(normal, half))
    const float kh, // abs(dot(dir, half))
    const float nk) // abs(dot(normal, dir))
{
    return fmin(1.0f, 2.0f * nh * nk / kh);
}

// v-cavities masking and and importance sampling utility
// (see "Eric Heitz and Eugene d'Eon - Importance Sampling Microfacet-Based BSDFs with the
// Distribution of Visible Normals")
class Vcavities_masking {
public:
    RT_FUNCTION optix::float3 flip(const optix::float3 &h, const optix::float3 &k, const float xi) const {
        const float a = h.y * k.y;
        const float b = h.x * k.x + h.z * k.z;
        const float kh = fmax(a + b, 0.0f);
        const float kh_f = fmax(a - b, 0.0f);

        if (xi < kh_f / (kh + kh_f)) {
            return optix::make_float3(-h.x, h.y, -h.z);
        }
        else
            return h;
    }

    RT_FUNCTION float shadow_mask(
        float &G1, float &G2,
        const float nh,
        const optix::float3 &k1, const float k1h,
        const optix::float3 &k2, const float k2h,
        const bool refraction) const {
        G1 = microfacet_mask_v_cavities(nh, k1h, k1.y);
        G2 = microfacet_mask_v_cavities(nh, k2h, k2.y);
        return refraction ? fmax(G1 + G2 - 1.0f, 0.0f) : fmin(G1, G2);
    }
};

// simple_glossy_bsdf uses a v-cavities-masked Phong distribution
class Distribution_phong_vcavities : public Vcavities_masking {
public:
    RT_FUNCTION Distribution_phong_vcavities(const float roughness_u, const float roughness_v) {
        m_exponent = roughness_to_exponent(
            clamp_roughness(roughness_u), clamp_roughness(roughness_v));
    }

    RT_FUNCTION float3 sample(const optix::float3 &xi, const optix::float3 &k) const {
        return flip(hvd_phong_sample(optix::make_float2(xi.x, xi.y), m_exponent), k, xi.z);
    }

    RT_FUNCTION float eval(const optix::float3 &h) const {
        return hvd_phong_eval(m_exponent, h.y, h.x, h.z);
    }

private:
    float2 m_exponent;
};


//
// Most glossy BSDF models in MDL are microfacet-theory based along the lines of
// "Bruce Walter, Stephen R. Marschner, Hongsong Li, Kenneth E. Torrance - Microfacet Models For
// Refraction through Rough Surfaces" and "Eric Heitz - Understanding the Masking-Shadowing
// Function in Microfacet-Based BRDFs
//
// The common utility code uses "Distribution", which has to provide:
// sample():      importance sample visible microfacet normals (i.e. including masking)
// eval():        evaluate microfacet distribution
// mask():        compute masking
// shadow_mask(): combine masking for incoming and outgoing directions
//

template <typename Distribution>
RT_FUNCTION bool microfacet_sample(
    const Distribution &ph,
    const scatter_mode mode,
    const optix::float3& normal,
    const optix::float3& geometricNormal,
    const optix::float3& tangentU,
    const optix::float3& tangentV,
    const optix::float3& wo,
    const optix::float3& tint,
    const optix::float3& sample,
    optix::float3& wi,
    optix::float3& bsdf_over_pdf,
    float& pdf
)
{
    BSDF_event_type event_type;

    const float nk1 = fabs(optix::dot(wo, normal));

    const float3 k10 = make_float3(
        optix::dot(wo, tangentU),
        nk1,
        optix::dot(wo, tangentV));

    // sample half vector / microfacet normal
    const float3 h0 = ph.sample(sample, k10);

    // transform to world
    const float3 h = normal * h0.y + tangentU * h0.x + tangentV * h0.z;
    const float kh = optix::dot(wo, h);

    if (kh <= 0.0f) {
        return false;
    }

    // compute probability of selection refraction over reflection
    const float2 ior = optix::make_float2(1.0f); // process_ior(data);
    float f_refl;
    switch (mode) {
    case scatter_reflect:
        f_refl = 1.0f;
        break;
    case scatter_transmit:
        f_refl = 0.0f;
        break;
    case scatter_reflect_transmit:
        f_refl = ior_fresnel(ior.y / ior.x, kh);
        break;
    }

    const bool thin_walled = true; // get_material_thin_walled();
    if (sample.x < f_refl)
    {
        // BRDF: reflect
        wi = (2.0f * kh) * h - wo;
        event_type = BSDF_EVENT_GLOSSY_REFLECTION;
    }
    else
    {
        bool tir = false;
        if (thin_walled) {
            // pseudo-BTDF: flip a reflected reflection direction to the back side
            wi = (2.0f * kh) * h - wo;
            wi = optix::normalize(
                wi - 2.0f * normal * optix::dot(wi, normal));
        }
        else
            // BTDF: refract
            wi = refract(wo, h, ior.x / ior.y, kh, tir);

        event_type = tir ? BSDF_EVENT_GLOSSY_REFLECTION : BSDF_EVENT_GLOSSY_TRANSMISSION;
        if (tir && (mode == scatter_transmit)) {
            return false;
        }
    }

    // check if the resulting direction is on the correct side of the actual geometry
    const float gnk2 = optix::dot(wi, geometricNormal) * (
        event_type == BSDF_EVENT_GLOSSY_REFLECTION ? 1.0f : -1.0f);
    if (gnk2 <= 0.0f) {
        return false;
    }

    const bool refraction = !thin_walled && (event_type == BSDF_EVENT_GLOSSY_TRANSMISSION);

    // compute weight
    bsdf_over_pdf = tint;

    const float nk2 = fabs(optix::dot(wi, normal));
    const float k2h = fabs(optix::dot(wi, h));

    float G1, G2;
    const float G12 = ph.shadow_mask(
        G1, G2, h0.y,
        k10, kh,
        optix::make_float3(optix::dot(wi, tangentU), nk2, optix::dot(wo, tangentV)), k2h,
        refraction);

    if (G12 <= 0.0f) {
        return false;
    }
    bsdf_over_pdf *= G12 / G1;

    // compute pdf
    {
        pdf = ph.eval(h0) * G1;

        if (refraction) {
            const float tmp = kh * ior.x - k2h * ior.y;
            pdf *= kh * k2h / (nk1 * h0.y * tmp * tmp);
        }
        else
            pdf *= 0.25f / (nk1 * h0.y);
    }

    return true;
}


template <typename Distribution>
RT_FUNCTION float microfacet_evaluate(
    const Distribution &ph,
    const scatter_mode mode,
    const optix::float3& normal,
    const optix::float3& geometricNormal,
    const optix::float3& tangentU,
    const optix::float3& tangentV,
    const optix::float3& wo,
    const optix::float3& wi,
    float& pdf
)
{
    // BTDF or BRDF eval?
    const bool backside_eval = optix::dot(wi, geometricNormal) < 0.0f;

    // nothing to evaluate for given directions?
    if ((backside_eval && (mode == scatter_reflect)) ||
        (!backside_eval && (mode == scatter_transmit))) {
        pdf = 0.0f;
        return 0.0f;
    }

    const float nk1 = fabs(optix::dot(normal, wo));
    const float nk2 = fabs(optix::dot(normal, wi));
    const bool thin_walled = true; // get_material_thin_walled();

    const float2 ior = optix::make_float2(1.0f); // process_ior(data);
    const float3 h = compute_half_vector(
        wo, wi, normal, ior, nk1, nk2,
        backside_eval, thin_walled);

    // invalid for reflection / refraction?
    const float nh = optix::dot(normal, h);
    const float k1h = optix::dot(wo, h);
    const float k2h = optix::dot(wi, h) * (backside_eval ? -1.0f : 1.0f);
    if (nh < 0.0f || k1h < 0.0f || k2h < 0.0f) {
        pdf = 0.0f;
        return 0.0f;
    }

    // compute BSDF and pdf
    const float fresnel_refl = ior_fresnel(ior.y / ior.x, k1h);
    const float weight = mode == scatter_reflect_transmit ?
        (backside_eval ? (1.0f - fresnel_refl) : fresnel_refl) :
        1.0f;

    pdf = ph.eval(make_float3(optix::dot(tangentU, h), nh, optix::dot(tangentV, h)));

    float G1, G2;
    //const float k2h = optix::abs(optix::dot(wi, h));
    const bool refraction = !thin_walled && backside_eval;
    const float G12 = ph.shadow_mask(
        G1, G2, nh,
        make_float3(optix::dot(tangentU, wo), nk1, optix::dot(tangentV, wo)), k1h,
        make_float3(optix::dot(tangentU, wi), nk2, optix::dot(tangentV, wi)), k2h,
        refraction);

    if (refraction) {
        // refraction pdf and BTDF
        const float tmp = k1h * ior.x - k2h * ior.y;
        pdf *= k1h * k2h / (nk1 * nh * tmp * tmp);
    }
    else {
        // reflection pdf and BRDF (and pseudo-BTDF for thin-walled)
        pdf *= 0.25f / (nk1 * nh);
    }

    const float bsdf = pdf * weight * G12;
    pdf *= G1;
    return bsdf;
}




RT_FUNCTION bool simple_glossy_sample(
    const scatter_mode mode,
    const optix::float3& normal,
    const optix::float3& geometricNormal,
    const optix::float3& tangentU,
    const optix::float3& tangentV,
    const optix::float3& wo,
    const optix::float3& tint,
    float roughness,
    const optix::float3& sample,
    optix::float3& wi,
    optix::float3& bsdf_over_pdf,
    float& pdf
)
{
    const Distribution_phong_vcavities ph(roughness, roughness);
    return microfacet_sample(
        ph,
        scatter_reflect,
        normal,
        geometricNormal,
        tangentU,
        tangentV,
        wo,
        tint,
        sample,
        wi,
        bsdf_over_pdf,
        pdf
    );
}


RT_FUNCTION void simple_glossy_evaluate(
    const scatter_mode mode,
    const optix::float3& normal,
    const optix::float3& geometricNormal,
    const optix::float3& tangentU,
    const optix::float3& tangentV,
    const optix::float3& wo,
    const optix::float3& wi,
    const optix::float3& tint,
    float roughness,
    optix::float3& bsdf,
    float& pdf
)
{
    const Distribution_phong_vcavities ph(roughness, roughness);
    bsdf = tint * microfacet_evaluate(
        ph,
        mode,
        normal,
        geometricNormal,
        tangentU,
        tangentV,
        wo,
        wi,
        pdf
    );
}


RT_FUNCTION float simple_glossy_pdf(
    const scatter_mode mode,
    const optix::float3& normal,
    const optix::float3& geometricNormal,
    const optix::float3& tangentU,
    const optix::float3& tangentV,
    const optix::float3& wo,
    const optix::float3& wi,
    float roughness
)
{
    float pdf = 0.0f;

    const Distribution_phong_vcavities ph(roughness, roughness);
    microfacet_evaluate(
        ph,
        mode,
        normal,
        geometricNormal,
        tangentU,
        tangentV,
        wo,
        wi,
        pdf
    );

    return pdf;
}



// ---------- Specular BSDF ------------

RT_FUNCTION bool specular_sample(
    const scatter_mode mode,
    const optix::float3& normal,
    const optix::float3& geometricNormal,
    const optix::float3& tangentU,
    const optix::float3& tangentV,
    const optix::float3& wo,
    const optix::float3& tint,
    const optix::float3& sample,
    optix::float3& wi,
    optix::float3& bsdf_over_pdf,
    float& pdf
)
{
    BSDF_event_type event_type;

    const float nk1 = optix::dot(normal, wo);
    if (nk1 < 0.0f) {
        return false;
    }

    bsdf_over_pdf = tint;
    pdf = 0.0f;

    const float2 ior = optix::make_float2(1.0f); // process_ior(data);

    // reflection
    if ((mode == scatter_reflect) ||
        ((mode == scatter_reflect_transmit) &&
            sample.x < ior_fresnel(ior.y / ior.x, nk1)))
    {
        wi = (nk1 + nk1) * normal - wo;

        event_type = BSDF_EVENT_SPECULAR_REFLECTION;
    }
    else // refraction
    {
        // total internal reflection should only be triggered for scatter_transmit
        // (since we should fall in the code-path above otherwise)
        bool tir = false;
        const bool thin_walled = true; // get_material_thin_walled();
        if (thin_walled) // single-sided -> propagate old direction
            wi = -wo;
        else
            wi = refract(wo, normal, ior.x / ior.y, nk1, tir);

        event_type = tir ? BSDF_EVENT_SPECULAR_REFLECTION : BSDF_EVENT_SPECULAR_TRANSMISSION;
    }

    // check if the resulting direction is on the correct side of the actual geometry
    const float gnk2 = optix::dot(wi, geometricNormal) * (
        event_type == BSDF_EVENT_SPECULAR_REFLECTION ? 1.0f : -1.0f);
    if (gnk2 <= 0.0f) {
        return false;
    }

    return true;
}



RT_FUNCTION void specular_evaluate()
{
    // nothing to do here
}

RT_FUNCTION float specular_pdf()
{
    // nothing to do here
    return 0.0f;
}





// ---------- Basic Material ------------

RT_FUNCTION float BasicMaterial_Opacity(const BasicMaterialParameters& parameters, const optix::float2 texCoord)
{
    float opacity = parameters.opacity;

    if (parameters.opacityTexture != RT_TEXTURE_ID_NULL)
        opacity = opacity * optix::rtTex2D<float4>(parameters.opacityTexture, texCoord.x, texCoord.y).x;

    if (parameters.diffuseTexture != RT_TEXTURE_ID_NULL)
        opacity = opacity * optix::rtTex2D<float4>(parameters.diffuseTexture, texCoord.x, texCoord.y).w;

    return opacity;
}


RT_FUNCTION void BasicMaterial_Init(const PathtracePRD& prd, const BasicMaterialParameters& parameters, BasicMaterialState& state)
{
    // Init state
    state.bsdf_over_pdf = optix::make_float3(0.0f);
    state.pdf = 0.0f;

    // Combine color channels with textures
    state.diffuse = parameters.diffuseColor * optix::make_float3(prd.color); // prd.color stores the diffuse color provided by the geometry as vertex colors
    if (parameters.diffuseTexture != RT_TEXTURE_ID_NULL)
        state.diffuse *= optix::make_float3(optix::rtTex2D<float4>(parameters.diffuseTexture, prd.texCoord.x, prd.texCoord.y));

    state.specular = parameters.specularColor;
    if (parameters.specularTexture != RT_TEXTURE_ID_NULL)
        state.specular *= optix::make_float3(optix::rtTex2D<float4>(parameters.specularTexture, prd.texCoord.x, prd.texCoord.y));

    state.emissive = parameters.emissiveColor;
    if (parameters.emissiveTexture != RT_TEXTURE_ID_NULL)
        state.emissive *= optix::make_float3(optix::rtTex2D<float4>(parameters.emissiveTexture, prd.texCoord.x, prd.texCoord.y));

    // Bump map
    if (parameters.bumpMapTexture != RT_TEXTURE_ID_NULL)
    {
        const optix::float3 N = optix::make_float3(optix::rtTex2D<float4>(parameters.bumpMapTexture, prd.texCoord.x, prd.texCoord.y)) * 2.0f - 1.0f;
        state.normal = optix::normalize(state.tangentU * N.x + state.tangentV * N.y + state.normal * N.z);
        state.tangentU = optix::normalize(optix::cross(state.tangentV, state.normal));
        state.tangentV = optix::normalize(optix::cross(state.normal, state.tangentU));
    }

    // Weights
    const float maxSum = optix::fmaxf(state.diffuse + state.specular + parameters.transparencyFilterColor);
    const float w = maxSum > 1.0f ? 1.0f / maxSum : 1.0f;

    state.weightDiffuse = w * state.diffuse;
    state.weightSpecular = w * state.specular;
    state.weightTransmissive = w * parameters.transparencyFilterColor;

    state.pDiffuse = 0.0f;
    state.pSpecular = 0.0f;
    state.pTransmissive = 0.0f;

    state.weightSum = state.weightDiffuse + state.weightSpecular + state.weightTransmissive;
    if (state.weightSum.x > 0.0f || state.weightSum.y > 0.0f || state.weightSum.z > 0.0f)
    {
        const float invWeightSum = 1.0f / Luminance(state.weightSum);

        state.pDiffuse = Luminance(state.weightDiffuse) * invWeightSum;
        state.pSpecular = Luminance(state.weightSpecular) * invWeightSum;
        state.pTransmissive = Luminance(state.weightTransmissive) * invWeightSum;
    }
}


RT_FUNCTION bool BasicMaterial_Sample(const BasicMaterialParameters& parameters, BasicMaterialState& state, const optix::float3& sample)
{
    const float roughness = sqrtf(2.0 / parameters.specularExponent);
    const float inv_w_sum = 1.0f / Luminance(state.weightSum);

    // Diffuse
    if (sample.z < state.pDiffuse)
    {
        if (!diffuse_sample(
            state.normal,
            state.geometricNormal,
            state.tangentU,
            state.tangentV,
            optix::make_float3(1.0f),
            sample,
            state.wi,
            state.bsdf_over_pdf,
            state.pdf
        ))
            return false;

        state.bsdf_over_pdf *= ((state.weightDiffuse) / (max(state.weightSum, optix::make_float3(1.0f, 1.0f, 1.0f)) * state.pDiffuse));
        state.pdf *= state.pDiffuse;

        // Add specular pdf
        const float q = Luminance(state.weightSpecular) * inv_w_sum;
        if (q > 0.0f)
        {
            state.pdf += q * simple_glossy_pdf(
                scatter_reflect,
                state.normal,
                state.geometricNormal,
                state.tangentU,
                state.tangentV,
                state.wo,
                state.wi,
                roughness
            );
        }

        // Add transmissive pdf
        // Always 0
    }

    // Specular
    else if (sample.z < state.pDiffuse + state.pSpecular)
    {
        if (!simple_glossy_sample(
            scatter_reflect,
            state.normal,
            state.geometricNormal,
            state.tangentU,
            state.tangentV,
            state.wo,
            optix::make_float3(1.0f),
            roughness,
            sample,
            state.wi,
            state.bsdf_over_pdf,
            state.pdf
        ))
            return false;

        state.bsdf_over_pdf *= ((state.weightSpecular) / (max(state.weightSum, optix::make_float3(1.0f, 1.0f, 1.0f)) * state.pSpecular));
        state.pdf *= state.pSpecular;

        // Add diffuse pdf
        const float q = Luminance(state.weightDiffuse) * inv_w_sum;
        if (q > 0.0f)
        {
            state.pdf += q * diffuse_pdf(
                state.normal,
                state.wi
            );
        }

        // Add transmissive pdf
        // Always 0
    }

    // Transmissive
    else if (sample.z < state.pDiffuse + state.pSpecular + state.pTransmissive)
    {
        if (!specular_sample(
            scatter_reflect_transmit,
            state.normal,
            state.geometricNormal,
            state.tangentU,
            state.tangentV,
            state.wo,
            parameters.transparencyFilterColor,
            sample,
            state.wi,
            state.bsdf_over_pdf,
            state.pdf
        ))
            return false;

        state.bsdf_over_pdf *= ((state.weightTransmissive) / (max(state.weightSum, optix::make_float3(1.0f, 1.0f, 1.0f)) * state.pTransmissive));
        state.pdf *= state.pTransmissive;

        // Add diffuse pdf
        const float qd = Luminance(state.weightDiffuse) * inv_w_sum;
        if (qd > 0.0f)
        {
            state.pdf += qd * diffuse_pdf(
                state.normal,
                state.wi
            );
        }

        // Add specular pdf
        const float qs = Luminance(state.weightSpecular) * inv_w_sum;
        if (qs > 0.0f)
        {
            state.pdf += qs * simple_glossy_pdf(
                scatter_reflect,
                state.normal,
                state.geometricNormal,
                state.tangentU,
                state.tangentV,
                state.wo,
                state.wi,
                roughness
            );
        }
    }

    // Absorb
    else
    {
        return false;
    }

    return true;
}


RT_FUNCTION void BasicMaterial_Eval(const BasicMaterialParameters& parameters, BasicMaterialState& state)
{
    if (state.pDiffuse <= 0.0f && state.pSpecular <= 0.0f && state.pTransmissive <= 0.0f)
    {
        state.bsdf = optix::make_float3(0.0f);
        state.pdf = 0.0f;
        return;
    }

    const float invWeightSum = 1.0f / Luminance(state.weightSum);
    const optix::float3 normalize = optix::make_float3(
        state.weightSum.x > 1.0f ? 1.0f / state.weightSum.x : 1.0f,
        state.weightSum.y > 1.0f ? 1.0f / state.weightSum.y : 1.0f,
        state.weightSum.z > 1.0f ? 1.0f / state.weightSum.z : 1.0f);


    optix::float3 bsdf = optix::make_float3(0.0f, 0.0f, 0.0f);
    float pdf = 0.0f;

    // Diffuse
    diffuse_evaluate(
        state.normal,
        optix::make_float3(1.0f),
        state.wi,
        state.bsdf,
        state.pdf
    );

    bsdf += state.bsdf * state.weightDiffuse * normalize;
    pdf += state.pdf * Luminance(state.weightDiffuse) * invWeightSum;

    // Specular
    const float roughness = sqrtf(2.0 / parameters.specularExponent);
    simple_glossy_evaluate(
        scatter_reflect,
        state.normal,
        state.geometricNormal,
        state.tangentU,
        state.tangentV,
        state.wo,
        state.wi,
        optix::make_float3(1.0f),
        roughness,
        state.bsdf,
        state.pdf);

    bsdf += state.bsdf * state.weightSpecular * normalize;
    pdf += state.pdf * Luminance(state.weightSpecular) * invWeightSum;

    // Transmission
    // nothing to do here


    // Save result
    state.bsdf = bsdf;
    state.pdf = pdf;
}
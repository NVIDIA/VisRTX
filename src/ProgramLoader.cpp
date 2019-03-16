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

#include "ProgramLoader.h"
#include "PTX.h"

#include <string>

namespace VisRTX
{
    optix::Program ProgramLoader::LoadProgram(optix::Context ctx, unsigned char* ptx, size_t ptxBytes, const std::string& programName)
    {
        std::string ptxString((char*)ptx, ptxBytes);

        try
        {            
            return ctx->createProgramFromPTXString(ptxString, programName);
        }
        catch (optix::Exception e)
        {
            const std::string message = "Failed to generate OptiX program \"" + programName + "\" from PTX: " + e.getErrorString();
            throw Exception(Error::UNKNOWN_ERROR, message.c_str());
        }
    }

    void ProgramLoader::LoadSharedPrograms()
    {
        optix::Context ctx = OptiXContext::Get();

        this->exceptionProgram = this->LoadPathtracerProgram("Exception");

        this->sphereIsectProgram = LoadProgram(ctx, Sphere_ptx, sizeof(Sphere_ptx), "SphereIntersect");
        this->sphereBoundsProgram = LoadProgram(ctx, Sphere_ptx, sizeof(Sphere_ptx), "SphereBounds");

        this->cylinderIsectProgram = LoadProgram(ctx, Cylinder_ptx, sizeof(Cylinder_ptx), "CylinderIntersect");
        this->cylinderBoundsProgram = LoadProgram(ctx, Cylinder_ptx, sizeof(Cylinder_ptx), "CylinderBounds");

        this->triangleIsectProgram = LoadProgram(ctx, TriangleMesh_ptx, sizeof(TriangleMesh_ptx), "TriangleMeshIntersection");
        this->triangleBoundsProgram = LoadProgram(ctx, TriangleMesh_ptx, sizeof(TriangleMesh_ptx), "TriangleMeshBoundingBox");
#if OPTIX_VERSION_MAJOR >= 6
        this->triangleAttributeProgram = LoadProgram(ctx, TriangleMesh_ptx, sizeof(TriangleMesh_ptx), "TriangleMeshAttribute");
#endif

        this->diskIsectProgram = LoadProgram(ctx, Disk_ptx, sizeof(Disk_ptx), "DiskIntersect");
        this->diskBoundsProgram = LoadProgram(ctx, Disk_ptx, sizeof(Disk_ptx), "DiskBounds");

        this->closestHitProgram = this->LoadPathtracerProgram("ClosestHit");
        //this->anyHitProgram = this->LoadPathtracerProgram("AnyHit");
        this->anyHitOcclusionProgram = this->LoadPathtracerProgram("AnyHitOcclusion");
        this->lightClosestHitProgram = this->LoadPathtracerProgram("LightClosestHit");
        this->lightAnyHitProgram = this->LoadPathtracerProgram("LightAnyHit");
        this->lightAnyHitOcclusionProgram = this->LoadPathtracerProgram("LightAnyHitOcclusion");

        this->closestHitPickProgram = this->LoadPathtracerProgram("ClosestHitPick");
        this->lightClosestHitPickProgram = this->LoadPathtracerProgram("LightClosestHitPick");
    }

    optix::Program ProgramLoader::LoadPathtracerProgram(const std::string& programName)
    {
        return LoadProgram(OptiXContext::Get(), Pathtracer_ptx, sizeof(Pathtracer_ptx), programName);
    }

    std::string ProgramLoader::GetMDLTexturesPTX()
    {
        return std::string((char*) Textures_ptx, sizeof(Textures_ptx));
    }
}
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

#pragma once

#include <string>
#include <vector>
#include <map>

#include "OptiX.h"

namespace VisRTX
{
	class ProgramLoader
	{
    public:
        static ProgramLoader& Get()
        {
            static ProgramLoader loader;
            return loader;
        }

        static std::string GetMDLTexturesPTX();
        
        optix::Program LoadPathtracerProgram(const std::string& programName);

    protected:
        ProgramLoader()
        {
            this->LoadSharedPrograms();
        }

		void LoadSharedPrograms();

        static optix::Program LoadProgram(optix::Context ctx, unsigned char* ptx, size_t ptxBytes, const std::string& programName);

	private:
        ProgramLoader(const ProgramLoader&) = delete;
		void operator=(const ProgramLoader&) = delete;

    public:
        // Shared programs that do not have any parameters assigned to their scope
        optix::Program sphereIsectProgram;
        optix::Program sphereBoundsProgram;
        optix::Program cylinderIsectProgram;
        optix::Program cylinderBoundsProgram;
        optix::Program triangleIsectProgram;
        optix::Program triangleBoundsProgram;
        optix::Program triangleAttributeProgram;
        optix::Program diskIsectProgram;
        optix::Program diskBoundsProgram;

        optix::Program exceptionProgram;

        optix::Program closestHitProgram;
        //optix::Program anyHitProgram;
        optix::Program anyHitOcclusionProgram;
        optix::Program lightClosestHitProgram;
        optix::Program lightAnyHitProgram;
        optix::Program lightAnyHitOcclusionProgram;

        optix::Program closestHitPickProgram;
        optix::Program lightClosestHitPickProgram;
	};
}

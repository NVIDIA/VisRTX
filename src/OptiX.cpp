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

#include "OptiX.h"
#include "Pathtracer/Common.h"

namespace VisRTX
{
    void OptiXContext::Init(optix::Context& context)
    {
#ifdef VISRTX_USE_DEBUG_EXCEPTIONS
        context->setPrintEnabled(true);
        context->setExceptionEnabled(RT_EXCEPTION_ALL, true);

#ifdef PRINT_PIXEL_X
        context->setPrintLaunchIndex(PRINT_PIXEL_X, PRINT_PIXEL_Y); // Launch index (0,0) at lower left.
#endif
#endif

        context->setUsageReportCallback(usageReportCallback, VISRTX_USAGE_REPORT_VERBOSITY, nullptr);
        context->setAttribute(RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES, 1);

        context->setStackSize(1440); // Important for OptiX 5.1

#if OPTIX_VERSION_MAJOR >= 6
        // It's an iterative path tracer, no recursion
        context->setMaxTraceDepth(1);
        context->setMaxCallableProgramDepth(1);
#endif

        context->setRayTypeCount(3); // Radiance, occlusion, pick
        context->setEntryPointCount(3); // Render, buffer cast, pick
    }
}


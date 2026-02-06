/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "egl_context.h"
#include "VisGLDevice.h"

using visgl::anariReportStatus;

static const char *egl_error_string(EGLint err) {
#define ERROR_CASE(X)\
    case X:\
        return #X;
    switch (err) {
        ERROR_CASE(EGL_SUCCESS)
        ERROR_CASE(EGL_NOT_INITIALIZED)
        ERROR_CASE(EGL_BAD_ACCESS)
        ERROR_CASE(EGL_BAD_ALLOC)
        ERROR_CASE(EGL_BAD_ATTRIBUTE)
        ERROR_CASE(EGL_BAD_CONTEXT)
        ERROR_CASE(EGL_BAD_CONFIG)
        ERROR_CASE(EGL_BAD_CURRENT_SURFACE)
        ERROR_CASE(EGL_BAD_DISPLAY)
        ERROR_CASE(EGL_BAD_SURFACE)
        ERROR_CASE(EGL_BAD_MATCH)
        ERROR_CASE(EGL_BAD_PARAMETER)
        ERROR_CASE(EGL_BAD_NATIVE_PIXMAP)
        ERROR_CASE(EGL_BAD_NATIVE_WINDOW)
        ERROR_CASE(EGL_CONTEXT_LOST)
    default:
        return "unkown error";
    }
#undef ERROR_CASE
}


#define EGL_CHECK(X)\
    do {\
        EGLBoolean result = X;\
        if (result == EGL_FALSE) {\
            anariReportStatus(\
                device,\
                device,\
                ANARI_DEVICE,\
                ANARI_SEVERITY_FATAL_ERROR,\
                ANARI_STATUS_UNKNOWN_ERROR,\
                "[EGL] %s failed with error %s",\
                #X,\
                egl_error_string(eglGetError()));\
        }\
    } while (false)

eglContext::eglContext(ANARIDevice device, EGLDisplay display, EGLenum api, EGLint debug)
    : device(device), display(display), api(api), debug(debug)
{

}

void eglContext::init() {
    anariReportStatus(device, device, ANARI_DEVICE,
        ANARI_SEVERITY_INFO, ANARI_STATUS_NO_ERROR,
        "[OpenGL] using EGL");

    // if there is no display injected attempt to use the platform one

    if(display == nullptr) {
        anariReportStatus(device, device, ANARI_DEVICE,
            ANARI_SEVERITY_INFO, ANARI_STATUS_NO_ERROR,
            "[EGL] no EGLDisplay provided, using platform display");


        PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
        if (!eglQueryDevicesEXT) {
            anariReportStatus(device, device, ANARI_DEVICE,
                ANARI_SEVERITY_FATAL_ERROR, ANARI_STATUS_UNKNOWN_ERROR,
                "[EGL] extension eglQueryDevicesEXT not available");
            return;
        }

        PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
            (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
        if (!eglGetPlatformDisplayEXT) {
            anariReportStatus(device, device, ANARI_DEVICE,
                ANARI_SEVERITY_FATAL_ERROR, ANARI_STATUS_UNKNOWN_ERROR,
                "[EGL] extension eglGetPlatformDisplayEXT not available");
            return;
        }

        PFNEGLQUERYDEVICESTRINGEXTPROC eglQueryDeviceStringEXT =
            (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT");
        if (!eglQueryDeviceStringEXT) {
            anariReportStatus(device, device, ANARI_DEVICE,
                ANARI_SEVERITY_FATAL_ERROR, ANARI_STATUS_UNKNOWN_ERROR,
                "[EGL] extension eglQueryDeviceStringEXT not available");
            return;
        }

        static const int MAX_DEVICES = 16;
        EGLDeviceEXT devices[MAX_DEVICES];
        EGLint numDevices;

        EGL_CHECK(eglQueryDevicesEXT(MAX_DEVICES, devices, &numDevices));

        if (numDevices == 0) {
            anariReportStatus(device, device, ANARI_DEVICE,
                ANARI_SEVERITY_FATAL_ERROR, ANARI_STATUS_UNKNOWN_ERROR,
                "[EGL] no EGL devices found");
            return;
        }

        this->display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[0], 0);
    }

    EGL_CHECK(eglInitialize(this->display, &major, &minor));

    EGL_CHECK(eglBindAPI(api));

    if(api == EGL_OPENGL_ES_API) {
        static const EGLint contextAttribs[] = {
            EGL_CONTEXT_OPENGL_DEBUG,   debug,
            EGL_CONTEXT_MAJOR_VERSION,  3,
            EGL_NONE
        };

        context = eglCreateContext(this->display, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, contextAttribs);
    } else if(api == EGL_OPENGL_API) {
        static const EGLint contextAttribs[] = {
            EGL_CONTEXT_OPENGL_DEBUG,   debug,
            EGL_CONTEXT_OPENGL_PROFILE_MASK,  EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_CONTEXT_MAJOR_VERSION,  4,
            EGL_CONTEXT_MINOR_VERSION,  3,
            EGL_NONE
        };

        context = eglCreateContext(this->display, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, contextAttribs);
    }

    if (context == EGL_NO_CONTEXT) {
        anariReportStatus(device, device, ANARI_DEVICE,
            ANARI_SEVERITY_FATAL_ERROR, ANARI_STATUS_UNKNOWN_ERROR,
            "[EGL] no context, error: %s", egl_error_string(eglGetError()));
    }
}

void eglContext::makeCurrent() {
    EGL_CHECK(eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context));
}

glContextInterface::loader_func_t* eglContext::loaderFunc() {
    return eglGetProcAddress;
}

void eglContext::release() {
    if (display != nullptr) {
        eglTerminate(display);
    }
}


eglContext::~eglContext() {
}
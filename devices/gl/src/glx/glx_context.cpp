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

#include "glx_context.h"
#include "VisGLDevice.h"

#include <cstring>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

using visgl::anariReportStatus;

glxContext::glxContext(ANARIDevice device, Display *display, GLXContext glx_context, int32_t debug)
 : device(device), display(display), share(glx_context), debug(debug)
{
}

void glxContext::init() {
    anariReportStatus(device, device, ANARI_DEVICE,
        ANARI_SEVERITY_INFO, ANARI_STATUS_NO_ERROR,
        "[OpenGL] using GLX");

    int attrib[] = {
        GLX_RENDER_TYPE,   GLX_RGBA_BIT,
        GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
        None
    };

    int pbAttrib[] = {
        GLX_PBUFFER_WIDTH, 128,
        GLX_PBUFFER_HEIGHT, 128,
        GLX_LARGEST_PBUFFER, 0,
        None
    };

    if(!display) {
        display = XOpenDisplay(NULL);
    }

    int screen = DefaultScreen(display);

    const char *extensions = glXQueryExtensionsString(display, screen);
    bool create_context_profile = std::strstr(extensions, "GLX_ARB_create_context_profile") != NULL;
    bool no_config_context = std::strstr(extensions, "GLX_EXT_no_config_context") != NULL;
    bool create_context_es2_profile = std::strstr(extensions, "GLX_EXT_create_context_es2_profile") != NULL;


    int count;
    GLXFBConfig *config = glXChooseFBConfig(display, screen, attrib, &count);

    if(count == 0) {
        anariReportStatus(device, device, ANARI_DEVICE,
            ANARI_SEVERITY_FATAL_ERROR, ANARI_STATUS_UNKNOWN_ERROR,
            "[OpenGL] no config");
        return;
    }

    pbuffer = glXCreatePbuffer(display, config[0], pbAttrib);

    if(!pbuffer) {
        anariReportStatus(device, device, ANARI_DEVICE,
            ANARI_SEVERITY_FATAL_ERROR, ANARI_STATUS_UNKNOWN_ERROR,
            "[OpenGL] failed to create pbuffer");
        return;
    }

    if(create_context_profile) {
        PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB =
            (PFNGLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB");

        if(debug) {
            anariReportStatus(device, device, ANARI_DEVICE,
                ANARI_SEVERITY_INFO, ANARI_STATUS_NO_ERROR,
                "[OpenGL] create debug context");
        }


        const int contextAttribs[] = {
            GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
            GLX_CONTEXT_FLAGS_ARB, (debug ? GLX_CONTEXT_DEBUG_BIT_ARB : 0),
            GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
            GLX_CONTEXT_MINOR_VERSION_ARB, 3,
            None
        };
        context = glXCreateContextAttribsARB(display, config[0], share, true, contextAttribs);
    } else {
        anariReportStatus(device, device, ANARI_DEVICE,
            ANARI_SEVERITY_INFO, ANARI_STATUS_NO_ERROR,
            "[OpenGL] fall back to legacy context creation");
        context = glXCreateNewContext(display, config[0], GLX_RGBA_TYPE, share, true);
    }
    if(!context) {
        anariReportStatus(device, device, ANARI_DEVICE,
            ANARI_SEVERITY_FATAL_ERROR, ANARI_STATUS_UNKNOWN_ERROR,
            "[OpenGL] failed to create context");
        return;
    }

    XFree(config);
}

void glxContext::makeCurrent() {
    glXMakeCurrent(display, pbuffer, context);
}

static void (*glx_loader(char const *name))(void) {
    return glXGetProcAddress((const GLubyte*)name);
}

glContextInterface::loader_func_t* glxContext::loaderFunc() {
    return glx_loader;
}

void glxContext::release() {
    glXMakeCurrent(display, None, NULL);
}

glxContext::~glxContext() {
    glXDestroyContext(display, context);
    glXDestroyPbuffer(display, pbuffer);
}
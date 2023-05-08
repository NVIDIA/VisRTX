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

        const int contextAttribs[] = {
            GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB | (debug ? GLX_CONTEXT_DEBUG_BIT_ARB : 0),
            GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
            GLX_CONTEXT_MINOR_VERSION_ARB, 3,
            None
        };
        context = glXCreateContextAttribsARB(display, config[0], share, true, contextAttribs);
    } else {
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
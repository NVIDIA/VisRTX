#pragma once

#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "glContextInterface.h"
#include  "anari/anari.h"

class eglContext : public glContextInterface {
private:
    ANARIDevice device;
    EGLint major = 0, minor = 0;
    EGLenum api;
    int32_t debug;
    EGLDisplay display;
    EGLConfig config;
    EGLContext context;

public:
    eglContext(ANARIDevice device, EGLDisplay display, EGLenum api, EGLint debug);
    void init() override;
    void release() override;
    void makeCurrent() override;
    loader_func_t* loaderFunc() override;
    ~eglContext();
};

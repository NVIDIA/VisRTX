#pragma once

#include "glad/gl.h"

#include <GL/glx.h>

#include "glContextInterface.h"
#include  "anari/anari.h"

class glxContext : public glContextInterface {
private:
    ANARIDevice device;
    
    Display *display = NULL;
    int32_t debug;
    GLXContext share = NULL;
    GLXContext context = NULL;
    GLXPbuffer pbuffer;

public:
    glxContext(ANARIDevice device, Display *display, GLXContext context, int32_t debug);
    void init() override;
    void release() override;
    void makeCurrent() override;
    loader_func_t* loaderFunc() override;
    ~glxContext();
};

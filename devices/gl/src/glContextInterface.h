#pragma once

class glContextInterface {
public:
    typedef void (*loader_func_t(char const *))(void);
    virtual void init() = 0;
    virtual void release() = 0;
    virtual void makeCurrent() = 0;
    virtual loader_func_t* loaderFunc() = 0;
    virtual ~glContextInterface() { }
};
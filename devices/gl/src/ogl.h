#pragma once

#include "glad/gl.h"

// undef problematic defines pulled in from windows headers
#ifdef far
#undef far 
#endif

#ifdef near
#undef near
#endif

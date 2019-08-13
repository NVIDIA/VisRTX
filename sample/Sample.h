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

/*
 * Shared base class for all example applications.
 */

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "Timer.h"

//#define VISRTX_DYNLOAD  // <-- Enable this line for dynamic loading instead of dynamic linking
#include <VisRTX.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <limits>
#include <stdio.h>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <string>
#include <set>
#include <fstream>

#include <glad/glad.h>

#ifdef VISRTX_SAMPLE_WITH_GLFW
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#endif

#ifdef VISRTX_SAMPLE_WITH_EGL
#include <EGL/egl.h>
#endif

using namespace VisRTX;

const float piOver180 = 0.01745329251f;

enum BenchmarkPhase
{
    INIT,
    WARMUP,
    DYNAMIC,
    STATIC,
    COMPLETE
};


/*
 * Abstract base class for sample applications
 */
class Sample
{
public:
    void Run(const std::string& title, int argc, char **argv);

public:
    virtual bool Init(int argc, char **argv) = 0;
    virtual void UpdateScene(bool benchmark, BenchmarkPhase benchmarkPhase, float benchmarkTimeDelta, bool pauseAllAnimations, bool& reset) = 0;
    virtual void UpdateGUI(bool& reset) = 0;

protected:
    struct ParameterGUI
    {
        std::string name;
        VisRTX::ParameterType type;
        float valueFloat;
        int valueInt;
        bool valueBool;
        Vec3f valueColor;
    };

    struct MaterialGUI
    {
        std::string objectName;
        VisRTX::MDLMaterial* mdl;
        std::vector<ParameterGUI> parameters;
        int id;
        bool dirty = false;

		void LoadCurrentValues()
		{
			for (ParameterGUI& p : this->parameters)
			{
				if (p.type == VisRTX::ParameterType::FLOAT)
					p.valueFloat = mdl->GetParameterFloat(p.name.c_str());
				else if (p.type == VisRTX::ParameterType::DOUBLE)
					p.valueFloat = (float)mdl->GetParameterDouble(p.name.c_str());
				else if (p.type == VisRTX::ParameterType::INT)
					p.valueInt = mdl->GetParameterInt(p.name.c_str());
				else if (p.type == VisRTX::ParameterType::BOOL)
					p.valueBool = mdl->GetParameterBool(p.name.c_str());
				else if (p.type == VisRTX::ParameterType::COLOR)
					p.valueColor = mdl->GetParameterColor(p.name.c_str());
			}
		}
    };

    VisRTX::MDLMaterial* LoadMDL(const std::string& material, const std::string& source, const std::vector<std::string>& modulePaths, VisRTX::CompilationType compilationType, uint8_t priority, const std::string& objectName);

    void DumpFrame(bool useOpenGL);

public:
    int width = 1920;
    int height = 1080;
    bool escapePressed = false;
    bool showGUI = true;

#ifdef VISRTX_SAMPLE_WITH_GLFW
    GLFWwindow* window = nullptr;        
#endif

#ifdef VISRTX_SAMPLE_WITH_EGL
    EGLDisplay display = nullptr;
    EGLSurface surface = nullptr;
#endif

    GLuint fullscreenQuadProgram;
    GLuint fullscreenTextureLocation;
    GLuint fullscreenVAO;

    double mouseX, mouseY;

    uint32_t frameNumber = 0;
    bool pauseAllAnimations = true;
    bool progressiveRendering = true;

    Timer fpsTimer;
    Timer fpsPrintTimer;
    uint32_t fpsCounter = 0;
    float fps = 0.0f;
    const float fpsUpdateInterval = 1.0f; // sec

    Timer renderTimer;
    Timer displayTimer;
    double renderTime = 0.0;
    double displayTime = 0.0;
    uint32_t renderTimeCounter = 0;
    uint32_t displayTimeCounter = 0;
    double renderTimeAverage = 0.0;
    double displayTimeAverage = 0.0;

	GLuint colorTex = 0;
	Timer updateGLTimer;
	float updateGLInterval = 200.0f;

    // Renderer
    Renderer* renderer;
    Model* model;
    bool aiDenoiser = false;
    int samplesPerPixel = 1;
    int numBouncesMin = 1;
    int numBouncesMax = 5;
    float fireflyClampingDirect = 0.0f;
    float fireflyClampingIndirect = 0.0f;
    bool sampleAllLights = false;

    bool toneMapping = true;
    float gamma = 2.2f;
    Vec3f colorBalance = Vec3f(1.0f, 1.0f, 1.0f);
    float whitePoint = 1.0f;
    float burnHighlights = 0.8f;
    float crushBlacks = 0.2f;
    float saturation = 1.2f;
    float brightness = 0.8f;

    // Framebuffer
    FrameBuffer* frameBuffer;

    // Ambient light
    AmbientLight* ambientLight;
    Vec3f ambientColor = Vec3f(0.0f, 0.0f, 0.0f);

    // Camera
    int camera = 0; // 0 = perspective, 1 = ortho
    int lastCamera = -1;
    bool depthOfField = false;
    float focalDistance = 0.0f;
    float apertureRadius = 0.0f;
	Vec2f imageBegin = Vec2f(0.0f, 0.0f);
	Vec2f imageEnd = Vec2f(1.0f, 1.0f);

    float rotationHorizontal = 20.0f;
    float rotationVertical = 6.0f;
    float distance = 5.1f;

    bool autoRotate = false;
    float rotateSpeed = 10.0; // deg/sec
    Timer rotateTimer;

    // Materials
    std::vector<MaterialGUI> materials;

    // Picking
    bool pick = false;
    int pickMode = 0; // 0: no picking, 1: set focal distance, 2: add marker, 3: randomize light, 4: remove geometry
    VisRTX::PickResult pickResult;
    std::set<VisRTX::Object*> releaseLater; // Stuff we want to keep until the very end (and not immediately delete when picked)

    // Clipping
    int numClippingPlanes = 0;
    std::vector<VisRTX::ClippingPlane> clippingPlanes;
    bool clippingPlanesDirty = true;
};



#ifdef VISRTX_SAMPLE_WITH_GLFW
void glfwErrorCallback(int error, const char* description)
{
    std::cerr << "Error: " << std::string(description) << std::endl;
}


void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    Sample* sample = reinterpret_cast<Sample*>(glfwGetWindowUserPointer(window));

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        sample->escapePressed = true;

    if (key == GLFW_KEY_TAB && action == GLFW_PRESS)
        sample->showGUI = !sample->showGUI;
}


void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    Sample* sample = reinterpret_cast<Sample*>(glfwGetWindowUserPointer(window));

    if (action == GLFW_PRESS)
    {
        glfwGetCursorPos(window, &sample->mouseX, &sample->mouseY);
    }

    // Picking
    if (action == GLFW_PRESS && glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
    {
        sample->pick = true;
    }
}


void cursorCallback(GLFWwindow* window, double xpos, double ypos)
{
    Sample* sample = reinterpret_cast<Sample*>(glfwGetWindowUserPointer(window));

    // Rotate
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        double dx = xpos - sample->mouseX;
        double dy = ypos - sample->mouseY;

        sample->rotationHorizontal += (float)(0.2 * dx);
        sample->rotationVertical += (float)(0.2 * dy);

        if (sample->rotationVertical > 89.0f)
            sample->rotationVertical = 89.0f;
        else if (sample->rotationVertical < -89.0f)
            sample->rotationVertical = -89.0f;

        sample->mouseX = xpos;
        sample->mouseY = ypos;

        sample->frameNumber = 0;
    }

    // Zoom
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
    {
        double dy = ypos - sample->mouseY;

        sample->distance -= (float)(0.01 * dy);

        if (sample->distance < 0.0f)
            sample->distance = 0.0f;

        sample->mouseX = xpos;
        sample->mouseY = ypos;

        sample->frameNumber = 0;
    }
}
#endif



static inline float randomValue()
{
    static bool initialized = false;
    if (!initialized)
    {
        srand(static_cast <unsigned> (time(0)));
        initialized = true;
    }

    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}



void savePPM(const uint8_t* rgba, uint32_t width, uint32_t height, const std::string& path)
{
    // Convert to RGB
    std::vector<uint8_t> rgb(width * height * 3);
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            uint32_t i1 = y * width + x;
            uint32_t i2 = (height - 1 - y) * width + x;

            for (uint32_t j = 0; j < 3; ++j)
            {
                rgb[3 * i1 + j] = rgba[4 * i2 + j];
            }
        }
    }

    // Write PPM
    std::ofstream outFile;
    outFile.open(path.c_str(), std::ios::binary);

    outFile << "P6" << "\n"
            << width << " " << height << "\n"
            << "255\n";

    outFile.write((char*) rgb.data(), rgb.size());
}



void Sample::Run(const std::string& title, int argc, char **argv)
{
    try
    {
#ifdef VISRTX_DYNLOAD
        // Load library first
        if (!VisRTX_LoadLibrary())
        {
            std::cerr << "Error: Failed to load VisRTX library" << std::endl;
            return;
        }
#endif

        // Device query and selection
        VisRTX::Context* context = VisRTX_GetContext();
        if (!context)
        {
            std::cerr << "Error: No capable device found (OptiX 6 requires SM 5.0 / Maxwell and up)" << std::endl;
            return;
        }

        // Command line args
        bool benchmark = false;
        bool offscreen = false;
        bool egl = false;
        bool useOpenGL = false;
        bool dump = false;
        for (int i = 0; i < argc; ++i)
        {
            if (std::string(argv[i]) == "benchmark")
            {
                benchmark = true;
            }
            else if (std::string(argv[i]) == "offscreen")
            {
                offscreen = true;
            }
            else if (std::string(argv[i]) == "egl")
            {
#ifdef VISRTX_SAMPLE_WITH_EGL
                egl = true;
                offscreen = true;
#else
                std::cout << "Error: EGL not enabled. Build VisRTX with VISRTX_SAMPLE_WITH_EGL=ON." << std::endl;
#endif
            }
            else if (std::string(argv[i]) == "dump")
            {
                dump = true;
            }
        }

        if (benchmark)
        {
            std::cout << "--- VisRTX Benchmark ---" << std::endl;
            std::cout << "Resolution: " << width << " x " << height << std::endl;
            std::cout << "Ray bounces: " << numBouncesMin << " - " << numBouncesMax << std::endl;
            std::cout << "Samples per pixel: " << samplesPerPixel << std::endl;
            std::cout << std::endl;
        }

        PerspectiveCamera* perspectiveCamera = context->CreatePerspectiveCamera();
        OrthographicCamera* orthographicCamera = context->CreateOrthographicCamera();

        model = context->CreateModel();

        renderer = context->CreateRenderer();
        renderer->SetModel(model);
        

        // Pick geometry
        std::vector<Vec3f> pickSpheres;

        SphereGeometry* pickSphereGeo = context->CreateSphereGeometry();
        pickSphereGeo->SetSpheres(0, nullptr);
        pickSphereGeo->SetRadius(0.05f);

        BasicMaterial* pickSphereMaterial = context->CreateBasicMaterial();
        pickSphereMaterial->SetDiffuse(Vec3f(1.0f, 0.0f, 0.0f));
        pickSphereMaterial->SetLuminosity(0.5f);
        pickSphereGeo->SetMaterial(pickSphereMaterial);
        model->AddGeometry(pickSphereGeo);


        // Ambient light
        ambientLight = context->CreateAmbientLight();
        renderer->AddLight(ambientLight);


#ifdef VISRTX_SAMPLE_WITH_GLFW
        // Create window
        if (!offscreen)
        {
            glfwSetErrorCallback(glfwErrorCallback);

            if (!glfwInit())
            {
                std::cerr << "Error: Failed to initialize GLFW." << std::endl;
                return;
            }

            const char* glsl_version = "#version 450";
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

            std::string fullTitle = benchmark ? (title + " - Benchmark") : title;
            window = glfwCreateWindow(width, height, fullTitle.c_str(), NULL, NULL);
            if (window == nullptr)
            {
                std::cerr << "Error: Failed to create window." << std::endl;
                return;
            }

            glfwSetWindowUserPointer(window, this);

            glfwMakeContextCurrent(window);
            glfwSwapInterval(0); // no vsync

            gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

            glfwSetKeyCallback(window, keyCallback);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorCallback);


            // Setup Dear ImGui binding
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO& io = ImGui::GetIO();
            io.FontAllowUserScaling = true;
            io.FontGlobalScale = 2.0f;

            ImGui_ImplGlfw_InitForOpenGL(window, true);
            ImGui_ImplOpenGL3_Init(glsl_version);

            ImGui::StyleColorsDark();

            useOpenGL = true;
        }
#endif


#ifdef VISRTX_SAMPLE_WITH_EGL
        if (offscreen && egl)
        {
            display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

            EGLint major, minor;
            if (eglInitialize(display, &major, &minor))
                std::cout << "EGL: YES" << std::endl;
            else
                std::cerr << "Error: Failed to initialize EGL" << std::endl;

            const EGLint configAttribs[] = {
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_BLUE_SIZE, 8,
                EGL_GREEN_SIZE, 8,
                EGL_RED_SIZE, 8,
                EGL_DEPTH_SIZE, 8,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_NONE
            };

            EGLint numConfigs;
            EGLConfig config;
            eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);

            const EGLint pbufferAttribs[] = {
                EGL_WIDTH, width,
                EGL_HEIGHT, height,
                EGL_NONE,
            };

            surface = eglCreatePbufferSurface(display, config, pbufferAttribs);

            eglBindAPI(EGL_OPENGL_API);
            EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
            eglMakeCurrent(display, surface, surface, context);

            gladLoadGLLoader((GLADloadproc) eglGetProcAddress);

            useOpenGL = true;
        }
#endif

        // Init OpenGL
        if (useOpenGL)
        {

            glClearColor(0.0f, 1.0f, 0.0f, 0.0f);

            // Create shader and texture for fullscreen display of decompressed frame
            const GLchar* clientVertexShader =
                    "#version 330\n"
                    "void main() {}";

            const GLchar* clientGeometryShader =
                    "#version 330 core\n"
                    "layout(points) in;"
                    "layout(triangle_strip, max_vertices = 4) out;"
                    "out vec2 texcoord;"
                    "void main() {"
                    "gl_Position = vec4( 1.0, 1.0, 0.0, 1.0 ); texcoord = vec2( 1.0, 1.0 ); EmitVertex();"
                    "gl_Position = vec4(-1.0, 1.0, 0.0, 1.0 ); texcoord = vec2( 0.0, 1.0 ); EmitVertex();"
                    "gl_Position = vec4( 1.0,-1.0, 0.0, 1.0 ); texcoord = vec2( 1.0, 0.0 ); EmitVertex();"
                    "gl_Position = vec4(-1.0,-1.0, 0.0, 1.0 ); texcoord = vec2( 0.0, 0.0 ); EmitVertex();"
                    "EndPrimitive();"
                    "}";

            const GLchar* clientFragmentShader =
                    "#version 330\n"
                    "uniform sampler2D tex;"
                    "in vec2 texcoord;"
                    "out vec4 color;"
                    "void main() {"
                    "	color = texture(tex, texcoord);"
                    "}";

            GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(vertexShader, 1, &clientVertexShader, 0);
            glCompileShader(vertexShader);

            GLuint geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
            glShaderSource(geometryShader, 1, &clientGeometryShader, 0);
            glCompileShader(geometryShader);

            GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(fragmentShader, 1, &clientFragmentShader, 0);
            glCompileShader(fragmentShader);

            fullscreenQuadProgram = glCreateProgram();
            glAttachShader(fullscreenQuadProgram, vertexShader);
            glAttachShader(fullscreenQuadProgram, geometryShader);
            glAttachShader(fullscreenQuadProgram, fragmentShader);
            glLinkProgram(fullscreenQuadProgram);

            fullscreenTextureLocation = glGetUniformLocation(fullscreenQuadProgram, "tex");

            glGenVertexArrays(1, &fullscreenVAO);
        }

        frameBuffer = context->CreateFrameBuffer(VisRTX::FrameBufferFormat::RGBA8);

        // Init sample
        if (!this->Init(argc, argv))
            escapePressed = true;


        /*
         * Main loop
         */
        rotateTimer.Reset();
        fpsTimer.Reset();

        Timer benchmarkTimer;
        uint32_t benchmarkFrame = 0;
        const uint32_t benchmarkWarmupFrames = 100;
        const uint32_t benchmarkDynamicFrames = 1000;
        const uint32_t benchmarkStaticFrames = 1000;

        const float benchmarkTimeDelta = 1.0f / 60.0f; // 60 Hz animation speed target

        double dynamicTime = 0.0;
        double staticTime = 0.0;

        BenchmarkPhase benchmarkPhase = INIT;

        while (!escapePressed)
        {
#ifdef VISRTX_SAMPLE_WITH_GLFW
            if (!offscreen && glfwWindowShouldClose(window))
                break;
#endif

            // Benchmark update
            if (benchmark)
            {
                BenchmarkPhase lastPhase = benchmarkPhase;

                if (benchmarkFrame < benchmarkWarmupFrames)
                    benchmarkPhase = WARMUP;
                else if (benchmarkFrame < benchmarkWarmupFrames + benchmarkDynamicFrames)
                    benchmarkPhase = DYNAMIC;
                else if (benchmarkFrame < benchmarkWarmupFrames + benchmarkDynamicFrames + benchmarkStaticFrames)
                    benchmarkPhase = STATIC;
                else
                    benchmarkPhase = COMPLETE;

                // Phase changed
                if (benchmarkPhase != lastPhase)
                {
                    double elapsed = benchmarkTimer.GetElapsedSeconds();

                    if (lastPhase == DYNAMIC)
                        dynamicTime = elapsed;
                    else if (lastPhase == STATIC)
                        staticTime = elapsed;

                    if (benchmarkPhase == WARMUP)
                        std::cout << "Warmup..." << std::endl;
                    else if (benchmarkPhase == DYNAMIC)
                        std::cout << "Dynamic..." << std::endl;
                    else if (benchmarkPhase == STATIC)
                        std::cout << "Static..." << std::endl;

                    lastPhase = benchmarkPhase;

                    benchmarkTimer.Reset();
                }

                if (benchmarkPhase == COMPLETE)
                {
                    std::cout << std::endl;

                    auto printPerf = [](const std::string& title, uint32_t numFrames, double elapsedSeconds)
                    {
                        double frameRate = numFrames / elapsedSeconds;
                        double frameTime = elapsedSeconds * 1000.0 / numFrames;
                        std::cout << title << ": " << std::setprecision(1) << std::fixed << frameRate << " Hz (" << frameTime << " ms)" << std::endl;
                    };

                    printPerf("Dynamic", benchmarkDynamicFrames, dynamicTime);
                    printPerf("Static", benchmarkStaticFrames, staticTime);

                    // Dump final frame onyl
                    if (dump)
                        this->DumpFrame(useOpenGL);

                    break;
                }


                ++benchmarkFrame;
            }


            // Begin frame
#ifdef VISRTX_SAMPLE_WITH_GLFW
            if (!offscreen)
            {

                glfwMakeContextCurrent(window);

                if (!benchmark)
                {
                    ImGui_ImplOpenGL3_NewFrame();
                    ImGui_ImplGlfw_NewFrame();
                    ImGui::NewFrame();
                }

                int w, h;
                glfwGetFramebufferSize(window, &w, &h);
                if (w != width || h != height)
                {
                    frameNumber = 0;
                    width = w;
                    height = h;
                }
            }
#endif

#ifdef VISRTX_SAMPLE_WITH_EGL
            if (offscreen && egl)
            {
                //eglMakeCurrent(display, surface, surface, context);
            }
#endif

            if (useOpenGL)
            {
                glViewport(0, 0, width, height);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            }


            // Update camera
            if (!benchmark && autoRotate && !pauseAllAnimations)
            {
                rotationHorizontal += rotateSpeed * (float)rotateTimer.GetElapsedSeconds();
                frameNumber = 0;
            }
            rotateTimer.Reset();

            float rho = rotationVertical * piOver180;
            float phi = -rotationHorizontal * piOver180;

            float camx = distance * cos(rho) * sin(phi);
            float camy = distance * sin(rho);
            float camz = distance * cos(rho) * cos(phi);

            perspectiveCamera->SetPosition(Vec3f(camx, camy, camz));
            perspectiveCamera->SetDirection(Vec3f(-camx, -camy, -camz));
            perspectiveCamera->SetFocalDistance(depthOfField ? focalDistance : -1.0f);
            perspectiveCamera->SetApertureRadius(depthOfField ? apertureRadius : -1.0f);
            perspectiveCamera->SetAspect((float)width / (float)height);
			perspectiveCamera->SetImageRegion(this->imageBegin, this->imageEnd);

            orthographicCamera->SetPosition(Vec3f(camx, camy, camz));
            orthographicCamera->SetDirection(Vec3f(-camx, -camy, -camz));
            orthographicCamera->SetHeight(distance);
            orthographicCamera->SetAspect((float)width / (float)height);
			orthographicCamera->SetImageRegion(this->imageBegin, this->imageEnd);

            if (camera != lastCamera)
            {
                if (camera == 1)
                    renderer->SetCamera(orthographicCamera);
                else
                    renderer->SetCamera(perspectiveCamera);

                lastCamera = camera;
                frameNumber = 0;
            }

            // Picking
            if (pick && pickMode > 0)
            {
                Vec2f screenPos((float)mouseX / (float)width, (float)mouseY / (float)height);
                if (renderer->Pick(screenPos, pickResult))
                {
                    // Set focal distance
                    if (pickMode == 1)
                    {
                        float dx = pickResult.position.x - camx;
                        float dy = pickResult.position.y - camy;
                        float dz = pickResult.position.z - camz;

                        focalDistance = sqrt(dx * dx + dy * dy + dz * dz);
                        depthOfField = true;

                        if (apertureRadius <= 0.0f)
                            apertureRadius = 0.05f;
                    }

                    // Add marker
                    else if (pickMode == 2)
                    {
                        if (pickResult.geometryHit || pickResult.lightHit)
                        {
                            pickSpheres.push_back(pickResult.position);
                            pickSphereGeo->SetSpheres(static_cast<uint32_t>(pickSpheres.size()), pickSpheres.data());

                            model->AddGeometry(pickSphereGeo); // automatically re-added if removed via picking
                        }
                    }

                    // Randomize light
                    else if (pickMode == 3)
                    {
                        if (pickResult.lightHit)
                            pickResult.light->SetColor(Vec3f(randomValue(), randomValue(), randomValue()));
                    }

                    // Remove geometry
                    else if (pickMode == 4)
                    {
                        if (pickResult.geometryHit)
                            model->RemoveGeometry(pickResult.geometry);
                    }

                    // Reset accumulation
                    frameNumber = 0;
                }

                pick = false;
            }



            // Update render settings
            renderer->SetToneMapping(toneMapping, gamma, colorBalance, whitePoint, burnHighlights, crushBlacks, saturation, brightness);
            renderer->SetDenoiser(aiDenoiser ? DenoiserType::AI : DenoiserType::NONE);
            renderer->SetSamplesPerPixel(samplesPerPixel);
            renderer->SetNumBounces(numBouncesMin, numBouncesMax);
            renderer->SetFireflyClamping(fireflyClampingDirect, fireflyClampingIndirect);
            renderer->SetSampleAllLights(sampleAllLights);			

            if (this->clippingPlanesDirty)
            {
                renderer->SetClippingPlanes(this->numClippingPlanes, this->clippingPlanes.data());
                this->clippingPlanesDirty = false;
            }

            // Update light
            ambientLight->SetColor(ambientColor);

            // Update sample
            bool reset = false;
            this->UpdateScene(benchmark, benchmarkPhase, benchmarkTimeDelta, pauseAllAnimations, reset);

            if (!progressiveRendering)
                reset = true;

            if (reset)
                frameNumber = 0;

            // Render
            frameBuffer->Resize(VisRTX::Vec2ui(width, height));
            if (frameNumber == 0)
                frameBuffer->Clear();

            this->renderTimer.Reset();
            renderer->Render(frameBuffer);
            this->renderTime += this->renderTimer.GetElapsedMilliseconds();
            ++this->renderTimeCounter;


            // Display image
            if (useOpenGL)
            {
                this->displayTimer.Reset();

				if (frameNumber == 0 || this->updateGLTimer.GetElapsedMilliseconds() >= this->updateGLInterval)
				{
					this->updateGLTimer.Reset();				
					this->colorTex = frameBuffer->GetColorTextureGL();
				}

				glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
				glUseProgram(fullscreenQuadProgram);
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, this->colorTex);
				glUniform1i(fullscreenTextureLocation, 0);
				glBindVertexArray(fullscreenVAO);
				glDrawArrays(GL_POINTS, 0, 1);

                this->displayTime += this->displayTimer.GetElapsedMilliseconds();
                ++this->displayTimeCounter;
            }

            // GUI
            reset = false;
            if (!benchmark)
            {
                ++fpsCounter;
                const float elapsed = (float)fpsTimer.GetElapsedSeconds();
                if (elapsed >= fpsUpdateInterval)
                {
                    fps = (float)fpsCounter / elapsed;
                    fpsCounter = 0;
                    fpsTimer.Reset();

                    this->renderTimeAverage = this->renderTimeCounter > 0 ? (this->renderTime / this->renderTimeCounter) : 0.0;
                    this->displayTimeAverage = this->displayTimeCounter > 0 ? (this->displayTime / this->displayTimeCounter) : 0.0;

                    this->renderTime = 0.0;
                    this->displayTime = 0.0;

                    this->renderTimeCounter = 0;
                    this->displayTimeCounter = 0;
                }

                if (fpsPrintTimer.GetElapsedSeconds() > 3.0f)
                {
                    std::cout << std::fixed << std::setprecision(1) << fps << " Hz (" << (fps > 0.0f ? 1000.0f / fps : 0.0f) << " ms)" << std::endl;
                    fpsPrintTimer.Reset();
                }

#ifdef VISRTX_SAMPLE_WITH_GLFW
                if (useOpenGL && !offscreen)
                {
                    if (showGUI)
                    {
                        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiSetCond_FirstUseEver);
                        ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings);

                        ImGui::Value("Frame Number", (int)frameNumber);
                        ImGui::Text("Resolution  : %u x %u", width, height);
                        ImGui::Text("Frame Rate  : %.1f Hz (%.1f ms)", fps, fps > 0.0f ? 1000.0f / fps : 0.0f);
                        ImGui::Text(" - Render   : %.1f ms", this->renderTimeAverage);
                        ImGui::Text(" - Display  : %.1f ms", this->displayTimeAverage);
                        ImGui::Separator();

                        ImGui::Checkbox("Progressive Rendering", &progressiveRendering);
                        ImGui::Checkbox("Pause All Animations", &pauseAllAnimations);
						ImGui::SliderFloat("##Update GL Interval", &this->updateGLInterval, 0.0f, 1000.0f, "Display every %.0f ms");
                        ImGui::Spacing();

                        if (ImGui::CollapsingHeader("Renderer"))
                        {
                            reset |= ImGui::SliderInt("Samples", &samplesPerPixel, 1, 32);
                            reset |= ImGui::SliderInt("Min Bounces", &numBouncesMin, 0, 10);
                            reset |= ImGui::SliderInt("Max Bounces", &numBouncesMax, 0, 50);
                            reset |= ImGui::SliderFloat("Clamping Direct", &fireflyClampingDirect, 0.0f, 1000.0f);
                            reset |= ImGui::SliderFloat("Clamping Indirect", &fireflyClampingIndirect, 0.0f, 1000.0f);
                            reset |= ImGui::Checkbox("Sample All Lights", &sampleAllLights);
                            ImGui::Separator();
                            reset |= ImGui::Checkbox("AI Denoiser", &aiDenoiser);
                        }
                        
                        ImGui::Spacing();

                        if (ImGui::CollapsingHeader("Tone Mapping"))
                        {
                            reset |= ImGui::Checkbox("Enabled", &toneMapping);
                            reset |= ImGui::ColorEdit3("Color Balance", (float*)&colorBalance);
                            reset |= ImGui::SliderFloat("Gamma", &gamma, 0.01f, 10.0f); // Must not get 0.0f
                            reset |= ImGui::SliderFloat("White Point", &whitePoint, 0.01f, 255.0f, "%.2f", 2.0f); // Must not get 0.0f
                            reset |= ImGui::SliderFloat("Burn Hightlights", &burnHighlights, 0.0f, 10.0f, "%.2f");
                            reset |= ImGui::SliderFloat("Crush Blacks", &crushBlacks, 0.0f, 1.0f, "%.2f");
                            reset |= ImGui::SliderFloat("Saturation", &saturation, 0.0f, 10.0f, "%.2f");
                            reset |= ImGui::SliderFloat("Brightness", &brightness, 0.0f, 100.0f, "%.2f", 2.0f);
                        }

                        ImGui::Spacing();

                        if (ImGui::CollapsingHeader("Camera"))
                        {
                            ImGui::Checkbox("Auto Rotate", &autoRotate);
                            if (autoRotate)
                                ImGui::SliderFloat("Speed", &rotateSpeed, 0.0f, 1000.0f, "%.1f deg/s");

                            ImGui::Separator();

                            ImGui::RadioButton("Perspective", &camera, 0); ImGui::SameLine();
                            ImGui::RadioButton("Orthographic", &camera, 1);

                            if (camera == 0)
                            {
                                reset |= ImGui::Checkbox("Depth of Field", &depthOfField);
                                if (depthOfField)
                                {
                                    reset |= ImGui::SliderFloat("Focal Distance", &focalDistance, 0.0f, 2.0f * distance, "%.1f");
                                    reset |= ImGui::SliderFloat("Aperture", &apertureRadius, 0.0f, 1.0f, "%.2f");
                                }
                            }

							ImGui::Separator();

							reset |= ImGui::SliderFloat2("Begin", &this->imageBegin.x, 0.0f, 1.0f, "%.2f");
							reset |= ImGui::SliderFloat2("End", &this->imageEnd.x, 0.0f, 1.0f, "%.2f");
                        }

                        ImGui::Spacing();

                        if (ImGui::CollapsingHeader("Clipping"))
                        {
                            this->clippingPlanesDirty |= ImGui::SliderInt("Planes", &this->numClippingPlanes, 0, 5);
                            if (this->clippingPlanesDirty)
                                this->clippingPlanes.resize(this->numClippingPlanes);

                            if (this->numClippingPlanes > 0)
                            {
                                ImGui::Separator();

                                for (int i = 0; i < this->numClippingPlanes; ++i)
                                {
                                    ClippingPlane* p = &this->clippingPlanes[i];

                                    std::string header = "Clipping Plane " + std::to_string(i + 1);
                                    if (ImGui::CollapsingHeader(header.c_str()))
                                    {
                                        std::string posName = "Position##plane" + std::to_string(i);
                                        this->clippingPlanesDirty |= ImGui::DragFloat3(posName.c_str(), &p->position.x, 0.1f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.2f");

                                        std::string normalName = "Normal##plane" + std::to_string(i);
                                        this->clippingPlanesDirty |= ImGui::SliderFloat3(normalName.c_str(), &p->normal.x, -1.0f, 1.0f, "%.2f");

                                        std::string primaryOnlyName = "Primary Rays Only##plane" + std::to_string(i);
                                        this->clippingPlanesDirty |= ImGui::Checkbox(primaryOnlyName.c_str(), &p->primaryRaysOnly);
                                    }
                                }
                            }

                            reset |= this->clippingPlanesDirty;
                        }

                        ImGui::Spacing();

                        if (ImGui::CollapsingHeader("Picking"))
                        {
                            ImGui::Text("Hold 'P' + Mouse Click");
                            ImGui::Separator();
                            ImGui::RadioButton("No Picking", &pickMode, 0);
                            ImGui::RadioButton("Set Focal Distance", &pickMode, 1);
                            ImGui::RadioButton("Add Marker", &pickMode, 2);
                            ImGui::RadioButton("Randomize Light", &pickMode, 3);
                            ImGui::RadioButton("Remove Geometry", &pickMode, 4);
                        }

                        ImGui::Spacing();

                        if (ImGui::CollapsingHeader("Ambient Light"))
                        {
                            reset |= ImGui::ColorEdit3("Color", &ambientColor.x);
                        }

                        ImGui::Spacing();

                        this->UpdateGUI(reset);

                        if (!this->materials.empty())
                        {
                            ImGui::Spacing();
                            ImGui::Text("MDL Materials");
                            ImGui::Spacing();

                            for (MaterialGUI& mat : this->materials)
                            {
                                std::string matTitle = std::string(mat.mdl->GetName());
                                if (!mat.objectName.empty())
                                    matTitle += " (" + mat.objectName + ")";
                                matTitle += "##" + std::to_string(mat.id);

                                if (ImGui::CollapsingHeader(matTitle.c_str()))
                                {
                                    for (ParameterGUI& p : mat.parameters)
                                    {
                                        const std::string title = p.name + "##" + std::to_string(mat.id);

                                        if (p.type == VisRTX::ParameterType::FLOAT)
                                        {
                                            if (ImGui::DragFloat(title.c_str(), &p.valueFloat, 0.001f))
                                            {
                                                mat.mdl->SetParameterFloat(p.name.c_str(), p.valueFloat);
                                                reset = mat.mdl->GetCompilationType() == VisRTX::CompilationType::CLASS;
                                                mat.dirty = true;
                                            }
                                        }
                                        else if (p.type == VisRTX::ParameterType::DOUBLE)
                                        {
                                            if (ImGui::DragFloat(title.c_str(), &p.valueFloat, 0.001f))
                                            {
                                                mat.mdl->SetParameterDouble(p.name.c_str(), p.valueFloat);
                                                reset = mat.mdl->GetCompilationType() == VisRTX::CompilationType::CLASS;
                                                mat.dirty = true;
                                            }
                                        }
                                        else if (p.type == VisRTX::ParameterType::INT)
                                        {
                                            if (ImGui::DragInt(title.c_str(), &p.valueInt))
                                            {
                                                mat.mdl->SetParameterInt(p.name.c_str(), p.valueInt);
                                                reset = mat.mdl->GetCompilationType() == VisRTX::CompilationType::CLASS;
                                                mat.dirty = true;
                                            }
                                        }
                                        else if (p.type == VisRTX::ParameterType::BOOL)
                                        {
                                            if (ImGui::Checkbox(title.c_str(), &p.valueBool))
                                            {
                                                mat.mdl->SetParameterBool(p.name.c_str(), p.valueBool);
                                                reset = mat.mdl->GetCompilationType() == VisRTX::CompilationType::CLASS;
                                                mat.dirty = true;
                                            }
                                        }
                                        else if (p.type == VisRTX::ParameterType::COLOR)
                                        {
                                            if (ImGui::ColorEdit3(title.c_str(), &p.valueColor.r, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR))
                                            {
                                                mat.mdl->SetParameterColor(p.name.c_str(), p.valueColor);
                                                reset = mat.mdl->GetCompilationType() == VisRTX::CompilationType::CLASS;
                                                mat.dirty = true;
                                            }
                                        }
                                    }

                                    // Recompile button
                                    if (mat.mdl->GetCompilationType() == VisRTX::CompilationType::INSTANCE && mat.dirty)
                                    {
                                        ImGui::PushStyleColor(0, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));

                                        const std::string buttonStr = "Compile##" + std::to_string(mat.id);
                                        if (ImGui::Button(buttonStr.c_str()))
                                        {
                                            mat.mdl->Compile();
                                            reset = true;
                                            mat.dirty = false;
                                        }

                                        ImGui::PopStyleColor();
                                    }
                                }
                            }
                        }

                        ImGui::End();
                    }

                    ImGui::Render();
                    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
                }
#endif
            }

            // End frame
            if (reset)
                frameNumber = 0;
            else
                ++frameNumber;

            // Dump frame
            if (dump && !benchmark)
                this->DumpFrame(useOpenGL);

#ifdef VISRTX_SAMPLE_WITH_GLFW
            if (!offscreen)
            {
                glfwSwapBuffers(window);
                glfwPollEvents();
            }

#endif

#ifdef VISRTX_SAMPLE_WITH_EGL
            if (offscreen && egl)
            {
                eglSwapBuffers(display, surface);
            }
#endif
        }

        // Clean up
        for (VisRTX::Object* obj : this->releaseLater)
            obj->Release();

        pickSphereMaterial->Release();
        pickSphereGeo->Release();
        ambientLight->Release();

        perspectiveCamera->Release();
        orthographicCamera->Release();
        renderer->Release();
        model->Release();
        frameBuffer->Release();

#ifdef VISRTX_SAMPLE_WITH_GLFW
        if (!offscreen)
        {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();

            glfwDestroyWindow(window);
            glfwTerminate();
        }
#endif

#ifdef VISRTX_SAMPLE_WITH_EGL
        if (offscreen && egl)
        {
            eglTerminate(display);
        }
#endif


    }
    catch (VisRTX::Exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
}


VisRTX::MDLMaterial* Sample::LoadMDL(const std::string& material, const std::string& source, const std::vector<std::string>& modulePaths, VisRTX::CompilationType compilationType, uint8_t priority, const std::string& objectName)
{
    Timer timer;

    std::vector<const char*> modules;
    for (const std::string& p : modulePaths)
        modules.push_back(p.c_str());

    VisRTX::Context* context = VisRTX_GetContext();
    VisRTX::MDLMaterial* mdl = context->CreateMDLMaterial(material.c_str(), source.c_str(), (uint32_t)source.size(), (uint32_t)modules.size(), modules.data(), compilationType, priority);
    mdl->Compile();


    //// Texture test
    //const uint32_t checkerboardRes = 512;
    //const uint32_t tiles = 8;
    //const uint32_t tileSize = checkerboardRes / tiles;

    //Vec4b* checkerboardPixels = new Vec4b[checkerboardRes * checkerboardRes];

    //for (uint32_t y = 0; y < checkerboardRes; ++y)
    //{
    //    for (uint32_t x = 0; x < checkerboardRes; ++x)
    //    {
    //        uint32_t tx = x / tileSize;
    //        uint32_t ty = y / tileSize;

    //        checkerboardPixels[y * checkerboardRes + x] = ((tx + ty) % 2 == 0) ? Vec4b(0, 0, 0) : Vec4b(255, 255, 255);
    //    }
    //}

    //Texture checkerboard = context->CreateTexture(Vec2ui(checkerboardRes, checkerboardRes), TextureFormat::RGBA8, checkerboardPixels);
    //mdl->SetParameterTexture("map_Kd", checkerboard);

    //mdl->Compile();


    // Register for GUI editing
	uint32_t numParameters = mdl->GetParameterCount();
	if (numParameters > 0)
	{
		MaterialGUI mat;
		mat.mdl = mdl;
		mat.objectName = objectName;

		static int matCounter = 0;
		mat.id = matCounter++;

		for (uint32_t i = 0; i < numParameters; ++i)
		{
			const char* name = mdl->GetParameterName(i);
			if (mdl->GetParameterType(name) == VisRTX::ParameterType::TEXTURE)
			{
				// Skip texture maps
				continue;
			}

			ParameterGUI p;
			p.name = std::string(name);
			p.type = mdl->GetParameterType(name);

			if (p.type == VisRTX::ParameterType::FLOAT)
				p.valueFloat = mdl->GetParameterFloat(name);
			else if (p.type == VisRTX::ParameterType::DOUBLE)
				p.valueFloat = (float)mdl->GetParameterDouble(name);
			else if (p.type == VisRTX::ParameterType::INT)
				p.valueInt = mdl->GetParameterInt(name);
			else if (p.type == VisRTX::ParameterType::BOOL)
				p.valueBool = mdl->GetParameterBool(name);
			else if (p.type == VisRTX::ParameterType::COLOR)
				p.valueColor = mdl->GetParameterColor(name);
			else if (p.type == VisRTX::ParameterType::TEXTURE)
			{
				p.valueInt = mdl->GetParameterInt(name);
				p.type = VisRTX::ParameterType::INT;
			}

			mat.parameters.push_back(p);
		}

		this->materials.push_back(mat);
	}

    std::cout << "Load MDL: " << mdl->GetName() << ": " << timer.GetElapsedMilliseconds() << " ms" << std::endl;

    return mdl;
}


void Sample::DumpFrame(bool useOpenGL)
{
    static int counter = 0;

    std::string path = std::to_string(counter++) + ".ppm";

    if (useOpenGL)
    {
        std::vector<uint8_t> rgba(this->width * this->height * 4);
        glReadPixels(0, 0, this->width, this->height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        savePPM(rgba.data(), this->width, this->height, path);
        std::cout << "Frame dumped from OpenGL: " << path << std::endl;
    }
    else
    {
        const uint8_t* rgba = (const uint8_t*) this->frameBuffer->MapColorBuffer();
        savePPM(rgba, this->width, this->height, path);
        this->frameBuffer->Unmap(rgba);

        std::cout << "Frame dumped from VisRTX framebuffer map: " << path << std::endl;
    }
}

#include <iostream>
#include <chrono>

#define TFN_WIDGET_NO_STB_IMAGE_IMPL
#include "transfer_function_widget.h"

#include "glfwHandler.h"
#include "rawFile.h"
#include <argparse/argparse.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "renderer.h"

struct Viewer
{
public:
    /* data */
    Viewer(int argc, char *argv[]);
    void Run();

    void TakeSnapshot(std::string filename = "frame.png", bool overlayColormap = false);
    void LeftMouseDrag(const owl::vec2i &where, const owl::vec2i &delta);
    void RightMouseDrag(const owl::vec2i &where, const owl::vec2i &delta);
    void CenterMouseDrag(const owl::vec2i &where, const owl::vec2i &delta);
    void printCameraParameters();

private:
    std::shared_ptr<dtracker::Renderer> renderer;
    std::shared_ptr<camera::Manipulator> manipulator;
    std::vector<ImTF::TransferFunctionWidget> tfnWidgets;
    size_t numFiles = 0;
    size_t selectedTF = 0;

    // imgui
    void InitImGui();
    void RequestImGuiFrame();
    void RenderImGuiFrame();

    const float kbd_rotate_degrees = 100.f;
    const float degrees_per_drag_fraction = 250;
    const float pixels_per_move = 90.f;
    bool camGivenAsParam = false;
    unsigned int rendererMode = 0;
    int heatMapMode = 0;
    float dt = -1.f;
    unsigned int offlineFrames = 100;
    unsigned int wuFrames = 0;
    unsigned int nthFrame = 0;
    std::string outputFileName = "out";
    std::string modeString = "";

    friend class dtracker::Renderer;
};

Viewer::Viewer(int argc, char *argv[])
{
    // invert image
    stbi_flip_vertically_on_write(true);
    
    // parse arguments
    argparse::ArgumentParser program("Viewer");

    program.add_argument("-fu", "--umesh-data")
        .help("path to the .umesh file");
    program.add_argument("-fr", "--raw-data")
        .help("path to the .raw data file(s)")
        .nargs(argparse::nargs_pattern::any);
    program.add_argument("-c", "--camera")
        .help("camera pos<x,y,z>, gaze<x,y,z>, up<x,y,z>, cosfovy(degrees)")
        .nargs(10)
        .scan<'g', float>();
    program.add_argument("-t", "--transfer-function")
        .help("path to the .tf file")
        .nargs(argparse::nargs_pattern::any);
    program.add_argument("-mc", "--macrocells")
        .help("number of macrocells per side")
        .nargs(0,3)
        .scan<'u', unsigned int>();
    program.add_argument("-bg", "--background")
        .help("background color")
        .nargs(3)
        .scan<'g', float>();
    program.add_argument("-dt", "--delta-t")
        .help("sampling distance")
        .scan<'g', float>();
    program.add_argument("-cb", "--correct-bounds")
        .help("correct the bounds of the raw file to the grid dimensions")
        .nargs(0,3)
        .scan<'g', float>();
    program.add_argument("-r", "--resolution")
        .help("resolution of the framebuffer")
        .nargs(2)
        .scan<'u', unsigned int>();
    program.add_argument("-l", "--light")
        .help("set up light. Takes enable shadows(0/1), enable gradient (0/1) light direction<x,y,z>, light intensity, ambient intensity")
        .scan<'g', float>()
        .nargs(5,7);
    std::string modeHelpText(
        "sets rendering mode. 0 = multiple DDA traversals using multiple majorant buffers,\n"
        "1 = single DDA traversal using multiple majorant buffers,\n" 
        "2 = alt. imp. single DDA traversal using multiple majorant buffer,\n"
        "3 = single DDA traversal using a cummulative majorant buffer,\n"
        "4 = MAX blending based Woodcock tracking, 5 = MIX blending based Woodcock tracking,\n"
        "6 = majorant weighted blending for Ray Marcher,\n"
        "7 = MAX blending based Ray Marcher, 8 = MIX blending based Ray Marcher\n"
        "9 = composite"); 
    program.add_argument("-m", "--mode")
        .help(modeHelpText)
        .scan<'u', unsigned int>()
        .default_value(0);
    program.add_argument("-spp", "--samples-per-pixel")
        .help("number of samples per pixel")
        .default_value(1)
        .scan<'i', int>();
    program.add_argument("-acc", "--accumulation")
        .help("enable accumulation")
        .default_value(true)
        .scan<'i', int>();

#if OFFLINE_VIEWER
    program.add_argument("-n", "--num-frame")
        .help("number of frames to render")
        .default_value(100)
        .scan<'u', unsigned int>();
    program.add_argument("-wu", "warm-up")
        .help("number of frames to render before measuring")
        .default_value(0)
        .scan<'u', unsigned int>();
    program.add_argument("-nt", "--nth-frame")
        .help("dump every Nth frame")
        .scan<'u', unsigned int>();
    program.add_argument("-o", "--output")
        .help("output file name");
    program.add_argument("-hm", "--heatmap")
        .help("enable timing heatmap. Set scale factor as argument")
        .scan<'g', float>();
#endif

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    renderer = std::make_shared<dtracker::Renderer>();

    if (program.is_used("-r"))
    {
        auto res = program.get<std::vector<unsigned int>>("-r");
        renderer->fbSize = vec2i(res[0], res[1]);
    }
#if !OFFLINE_VIEWER
    GLFWHandler::getInstance()->initWindow(renderer->fbSize.x, renderer->fbSize.y, "Multi-Density WT Viewer");
    // init imgui
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    // Initialize ImGui
    InitImGui();
#else
    if(program.is_used("-n"))
        offlineFrames = program.get<unsigned int>("-n");
    if(program.is_used("-wu"))
        wuFrames = program.get<unsigned int>("-wu");
    printf("used %d value %u\n", program.is_used("-nt"), nthFrame);
    if(program.is_used("-nt"))
        nthFrame = program.get<unsigned int>("-nt");
    if(program.is_used("-o"))
        outputFileName = program.get<std::string>("-o");
    if(program.is_used("-hm"))
    {
        heatMapMode = 3;
        renderer->heatMapScale = program.get<float>("-hm");
    }
#endif

    if (program.is_used("-c"))
    {
        auto camera = program.get<std::vector<float>>("-c");
        renderer->camera.setOrientation(vec3f(camera[0], camera[1], camera[2]),
                                        vec3f(camera[3], camera[4], camera[5]),
                                        vec3f(camera[6], camera[7], camera[8]),
                                        camera[9]);
        camGivenAsParam = true;
    }
    if (program.is_used("-mc"))
    {
        //check if mc is positive
        auto mc = program.get<std::vector<unsigned int>>("-mc");
        int i =0;
        if(mc.size() == 1)
        {
            mc.push_back(mc[0]);
            mc.push_back(mc[0]);
        }
        for (auto m : mc)
        {
            if (m < 1)
            {
                std::cerr << "Number of macrocells per side must be positive" << std::endl;
                std::exit(1);
            }
            renderer->macrocellDims[i++] = m;
        }
        printf("Using MC Grid Dims: %u, %u, %u\n", renderer->macrocellDims.x, renderer->macrocellDims.y, renderer->macrocellDims.z);
    }
    if (program.is_used("-bg"))
    {
        auto bg = program.get<std::vector<float>>("-bg");
        renderer->bgColor = vec3f(bg[0], bg[1], bg[2]);
    }
    if(program.is_used("-m"))
    {
        rendererMode = program.get<unsigned int>("-m");
    }
    if(program.is_used("-spp"))
    {
        renderer->spp = program.get<int>("-spp");
    }
    switch (rendererMode)
    {
    case 0:
        modeString = "BLN"; //baseline (multiple DDA traversals)
        break;
    case 1:
        modeString = "MMB"; //multiple Majorant Buffers
        break;
    case 2:
        modeString = "AMMB"; //alternative multiple Majorant Buffer
        break;
    case 3:
        modeString = "CMB"; //cummulative Majorant Buffer
        break;
    case 4:
        modeString = "MAX"; //MAX blending for Woodcock tracking
        break;
    case 5:
        modeString = "MIX"; //MIX blending for Woodcock tracking
        break;
    case 6:
        modeString = "MM_RM"; //multiple Majorant Buffer for Ray Marcher
        break;
    case 7:
        modeString = "MAX_RM"; //MAX blending for Ray Marcher
        break;
    case 8:
        modeString = "MIX_RM"; //MIX blending for Ray Marcher
        break;
    case 9:
        modeString = "COMP"; //composite
        break;
    }
    if(program.is_used("-l"))
    {
        auto sh = program.get<std::vector<float>>("-l");
        renderer->enableShadows = sh[0];
        renderer->enableGradientShading = sh[1];
        renderer->lightDir = vec3f(sh[2], sh[3], sh[4]);
        if (sh.size() > 5)
            renderer->lightIntensity = sh[5];
        if (sh.size() > 6)
            renderer->ambient = sh[6];
    }

    if(program.is_used("-fu"))
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto umeshHdlPtr = umesh::io::loadBinaryUMesh(program.get<std::string>("-fu"));
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken to load umesh: " << duration.count() << " milliseconds" << std::endl;
        std::cout << "found " << umeshHdlPtr->tets.size() << " tetrahedra" << std::endl;
        std::cout << "found " << umeshHdlPtr->pyrs.size() << " pyramids" << std::endl;
        std::cout << "found " << umeshHdlPtr->wedges.size() << " wedges" << std::endl;
        std::cout << "found " << umeshHdlPtr->hexes.size() << " hexahedra" << std::endl;
        std::cout << "found " << umeshHdlPtr->vertices.size() << " vertices" << std::endl;
        renderer->PushMesh(umeshHdlPtr);
        numFiles++;
    }
    else if(program.is_used("-fr"))
    {
        auto paths = program.get<std::vector<std::string>>("-fr");
        for (auto path : paths)
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto rawFile = std::make_shared<raw::RawR>(path.c_str(), "rb");
            if (program.is_used("-cb"))
            {
                auto bounds = program.get<std::vector<float>>("-cb");
                if(bounds.size() == 3)
                    rawFile->reshapeBounds(vec3f(bounds[0], bounds[1], bounds[2]));
                else if(bounds.size() == 0)
                    rawFile->reshapeBounds();
                else
                {
                    std::cerr << "Bounds must be given as 3 floats or not at all" << std::endl;
                    std::exit(1);
                }

            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Time taken to load raw data: " << duration.count() << " milliseconds" << std::endl;
            std::cout << "found " << rawFile->getDims().x << " x " << rawFile->getDims().y << " x " << rawFile->getDims().z << " voxels and " << rawFile->getBytesPerVoxel() << " byte(s) per voxel" << std::endl;
            std::cout << "total size: " << rawFile->getDims().x * rawFile->getDims().y * rawFile->getDims().z * rawFile->getBytesPerVoxel() / 1024.0f / 1024.0f << " MB" << std::endl;
            renderer->PushMesh(rawFile);
            numFiles++;
        }
    }
    else
    {
        std::cerr << "No data file given" << std::endl;
        std::exit(1);
    }
    
    // init transfer function widgets
    tfnWidgets = std::vector<ImTF::TransferFunctionWidget>(numFiles, OFFLINE_VIEWER);
    // if tf argument is given, load it
    if (program.is_used("-t"))
    {
        auto tfStates = program.get<std::vector<std::string>>("-t");
        for (int i = 0; i < tfStates.size(); i++)
        {
            tfnWidgets[i].LoadState(tfStates[i]);
        }
    }
    if(program.is_used("-dt"))
    {
        dt = program.get<float>("-dt");
    }
    if(program.is_used("-acc"))
    {
        renderer->enableAccumulation = (bool)program.get<int>("-acc");
    }

    manipulator = std::make_shared<camera::Manipulator>(&(renderer->camera));
}

void Viewer::TakeSnapshot(std::string filename, bool overlayColormap)
{
    const uint32_t *fb = (const uint32_t *)
        owlBufferGetPointer(renderer->frameBuffer, 0);
    
    // Create a copy of the framebuffer for modification
    std::vector<uint32_t> modifiedFb(renderer->fbSize.x * renderer->fbSize.y);
    std::copy(fb, fb + renderer->fbSize.x * renderer->fbSize.y, modifiedFb.begin());
    

    // Add colormap overlays using the transfer function widget utility
    if (numFiles > 0 && overlayColormap) {
        const float scale = 1.1f; // Default scale
        const int marginBottom = renderer->fbSize.y - 250;
        const int barSpacing = 85; // Spacing between bars
        const int marginLeft = renderer->fbSize.x - barSpacing * numFiles - 25;
        
        for (int fileIndex = 0; fileIndex < numFiles; ++fileIndex) {
            // Calculate position for this colormap bar using distance from bottom-left corner
            vec2f pos(
                marginLeft + fileIndex * barSpacing,  // Distance from left edge
                marginBottom  // Distance from bottom of image
            );
            
            // Get the data range for this file
            vec2f dataRange(
                renderer->rawPtrs[fileIndex]->getMinValue(),
                renderer->rawPtrs[fileIndex]->getMaxValue()
            );
            
            // Overlay the colormap bar
            tfnWidgets[fileIndex].OverlayColormapBar(modifiedFb, renderer->fbSize.x, renderer->fbSize.y,
                                                    {pos.x, pos.y}, {dataRange.x, dataRange.y}, scale, true);
        }
    }
    
    stbi_write_png(filename.c_str(),
                   renderer->fbSize.x, renderer->fbSize.y, 4,
                   modifiedFb.data(), renderer->fbSize.x * sizeof(uint32_t));
    printf("Saved current frame to '%s'\n", filename.c_str());
}

void Viewer::printCameraParameters()
{
    vec3f pos = renderer->camera.getFrom();
    vec3f gaze = renderer->camera.getAt();
    vec3f up = renderer->camera.getUp();
    float cosfovy = renderer->camera.getFovyInDegrees();
    printf("-c %f %f %f %f %f %f %f %f %f %f\n", 
        pos.x, pos.y, pos.z,
        gaze.x, gaze.y, gaze.z,
        up.x, up.y, up.z,
        cosfovy);
}

void Viewer::LeftMouseDrag(const owl::vec2i &where, const owl::vec2i &delta)
{
    auto glfw = GLFWHandler::getInstance();
    const owl::vec2f fraction = owl::vec2f(delta) /
                                owl::vec2f(glfw->getWindowSize());
    manipulator->rotate(fraction.x * degrees_per_drag_fraction,
                        fraction.y * degrees_per_drag_fraction);
    renderer->UpdateCamera();
}

void Viewer::RightMouseDrag(const owl::vec2i &where, const owl::vec2i &delta)
{
    auto glfw = GLFWHandler::getInstance();
    const owl::vec2f fraction = owl::vec2f(delta) /
                                owl::vec2f(glfw->getWindowSize());
    manipulator->move(fraction.y * pixels_per_move);
    renderer->UpdateCamera();
}

void Viewer::CenterMouseDrag(const owl::vec2i &where, const owl::vec2i &delta)
{
    auto glfw = GLFWHandler::getInstance();
    const owl::vec2f fraction = owl::vec2f(delta) /
                                owl::vec2f(glfw->getWindowSize());
    manipulator->strafe(fraction * pixels_per_move);
    renderer->UpdateCamera();
}

void Viewer::InitImGui()
{
    ImGui::StyleColorsDark(); //We always want dark mode
    ImGui_ImplGlfw_InitForOpenGL(GLFWHandler::getInstance()->getWindow(), true);
    ImGui_ImplOpenGL3_Init("#version 130");
}

void Viewer::RequestImGuiFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Viewer::RenderImGuiFrame()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

const ImVec4 red = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
const ImVec4 orange = ImVec4(1.0f, 0.65f, 0.0f, 1.0f);
const ImVec4 green = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);

void Viewer::Run()
{
#if !OFFLINE_VIEWER
    GLFWHandler *glfw = GLFWHandler::getInstance();
#endif 
    renderer->Init(rendererMode,!camGivenAsParam);
    if(dt > 0.f)
        renderer->Setdt(dt);
    renderer->UpdateCamera();
    //loop over all transfer functions and set them
    for (int i = 0; i < numFiles; i++)
    {
        auto cm = tfnWidgets[i].GetColormapf();
        std::vector<owl::vec4f> colorMapVec;
        for (int i = 0; i < cm.size(); i += 4)
        {
            colorMapVec.push_back(owl::vec4f(cm[i],
                    cm[i + 1], cm[i + 2], cm[i + 3]));
        }
        renderer->SetXFColormap(colorMapVec, i);
        renderer->SetXFOpacityScale(tfnWidgets[i].GetOpacityScale(), i);
        renderer->SetXFRange(vec2f(tfnWidgets[i].GetRange().x,
                tfnWidgets[i].GetRange().y), i);
    }
    while (
#if OFFLINE_VIEWER
        renderer->frameID < (offlineFrames + wuFrames)
#else
        !glfw->windowShouldClose()
#endif  
        )
    {
#if !OFFLINE_VIEWER
        glfw->pollEvents();

        glfw->mouseState.imGuiPolling = ImGui::GetIO().WantCaptureMouse;
#else   
        if(renderer->frameID <= wuFrames)
            renderer->ResetAccumulation();
#endif  

        renderer->Render(heatMapMode);

        renderer->Update();
#if !OFFLINE_VIEWER
        glfw->draw((const uint32_t *)
                       owlBufferGetPointer(renderer->frameBuffer, 0));

        RequestImGuiFrame();
        ImGui::Begin("Renderer Controls");
        ImGui::Text("Mode: %s", modeString.c_str());
        ImGui::SeparatorText("Stats");
        // write the fps metrics as a colored text in imgui
        ImGui::Text("avg. fps:");
        ImGui::SameLine();
        ImVec4 color = renderer->avgTime > 0.6f ? red : renderer->avgTime < 0.11f ? green : orange;
        ImGui::TextColored(color," %.3f (%0.3f sec)", 1.0f/renderer->avgTime, renderer->avgTime);
        ImGui::Text("best. fps:");
        ImGui::SameLine();
        color = renderer->minTime > 0.6f ? red : renderer->minTime < 0.11f ? green : orange;
        ImGui::TextColored(color," %.3f (%0.3f sec)", 1.0f/renderer->minTime, renderer->minTime);


        if(ImGui::CollapsingHeader("Transfer function(s)", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::BeginTabBar("##42#left_tabs_bar");
            for (int i = 0; i < numFiles; i++)
            {
                if (ImGui::BeginTabItem(std::string("TF"+std::to_string(i)).c_str()))
                {
                    tfnWidgets[i].DrawColorMap(false);
                    tfnWidgets[i].DrawOpacityScale();
                    tfnWidgets[i].DrawRanges();
                    ImGui::EndTabItem();
                    //tabChanged = selectedTF != i;
                    selectedTF = i;
                }
            }
            ImGui::EndTabBar();
        }
        if(ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if(ImGui::Checkbox("Shadows", &(renderer->enableShadows)))
                renderer->SetLightDirection(renderer->lightDir); // to reset accumulation
            ImGui::SameLine();
            if(ImGui::Checkbox("Gradient shading", &(renderer->enableGradientShading)))
                renderer->ResetAccumulation();
            static float lightIntensity = renderer->lightIntensity;
            if(ImGui::DragFloat("Light Intensity", &lightIntensity, 0.01f, 0.0f, 1e20f))
                renderer->SetLightIntensity(lightIntensity);
            if(renderer->enableShadows || renderer->enableGradientShading)
            {
                static float ambient = renderer->ambient;
                if (ImGui::DragFloat("Ambient", &ambient, 0.01f, 0.0f, 10.0f))
                    renderer->SetAmbient(ambient);
                static vec3f lightDir = renderer->lightDir;
                if(ImGui::DragFloat3("Light Direction", &lightDir.x, 0.01f, -1.0f, 1.0f))
                    renderer->SetLightDirection(lightDir);
            }
        }
        dt = renderer->dt;
        if(ImGui::DragFloat("dt/G.O.", &dt, 0.005f, 0.0f, 1e20f, "%.5f"))
            renderer->Setdt(dt);
        if(ImGui::Button("Reset dt/G.O."))
            renderer->Resetdt();
        if(ImGui::InputInt("SPP", &(renderer->spp),1,100,0))
        {
            renderer->ResetAccumulation();
            renderer->spp = std::max(1,renderer->spp);
        }
        //render radio button group for heatmap mode
        ImGui::Text("Heatmap Mode");
        ImGui::BeginGroup();
        if(ImGui::RadioButton("Off", (int*)&heatMapMode, 0))
            renderer->ResetAccumulation();
        ImGui::SameLine();
        if(ImGui::RadioButton("samples", (int*)&heatMapMode, 1))
            renderer->ResetAccumulation();
        ImGui::SameLine();
        if(ImGui::RadioButton("rejects", (int*)&heatMapMode, 2))
            renderer->ResetAccumulation();
        ImGui::SameLine();
        if(ImGui::RadioButton("timing", (int*)&heatMapMode, 3))
            renderer->ResetAccumulation();
        ImGui::EndGroup();

        if(heatMapMode > 0 && ImGui::DragFloat("HeatMap Scale", &(renderer->heatMapScale), 0.01f, 0.0f))
            renderer->ResetAccumulation();

        if(ImGui::Checkbox("Accumulation", &(renderer->enableAccumulation)))
            renderer->ResetAccumulation();
        ImGui::End();
        RenderImGuiFrame();

        if(tfnWidgets[selectedTF].ColorMapChanged())
        {
            auto cm = tfnWidgets[selectedTF].GetColormapf();
            std::vector<owl::vec4f> colorMapVec;
            for (int i = 0; i < cm.size(); i += 4)
            {
                colorMapVec.push_back(owl::vec4f(cm[i],
                        cm[i + 1], cm[i + 2], cm[i + 3]));
            }
            renderer->SetXFColormap(colorMapVec, selectedTF);
        }

        if(tfnWidgets[selectedTF].OpacityScaleChanged())
            renderer->SetXFOpacityScale(tfnWidgets[selectedTF].GetOpacityScale(), selectedTF);
            
        if(tfnWidgets[selectedTF].RangeChanged())
            renderer->SetXFRange(vec2f(tfnWidgets[selectedTF].GetRange().x,
                    tfnWidgets[selectedTF].GetRange().y), selectedTF);

        glfw->swapBuffers();

        //==== Input Processing ====

        // Taking a snapshot of the current frame
        if (glfw->key.isPressed(GLFW_KEY_1) &&
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"!"
            TakeSnapshot("frame.png", glfw->key.isDown(GLFW_KEY_LEFT_CONTROL));

        if (glfw->key.isPressed(GLFW_KEY_EQUAL) &&
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"+"
            renderer->Setdt(renderer->dt + 0.01f );
        else if (glfw->key.isPressed(GLFW_KEY_MINUS) &&
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"-"
            renderer->Setdt(renderer->dt - 0.01f);

        else if (glfw->key.isRepeated(GLFW_KEY_EQUAL) &&
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"+"
            renderer->Setdt(renderer->dt + 0.04f);
        else if (glfw->key.isRepeated(GLFW_KEY_MINUS) &&
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"-"
            renderer->Setdt(renderer->dt - 0.04f);

        if(glfw->key.isPressed(GLFW_KEY_T) && 
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"T"
            tfnWidgets[selectedTF].SaveState("tfn_state_" + std::to_string(selectedTF) + ".tf");
        if(glfw->key.isPressed(GLFW_KEY_C) && 
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"C"
            printCameraParameters();
        if(glfw->key.isPressed(GLFW_KEY_X)) //switch up vector to x/-x
        {
            vec3f up = vec3f(-1.f, .0f, .0f) * renderer->camera.getUp();
            up.x = up.x == .0f ? 1.f : up.x;
            renderer->camera.setUpVector(up);
            renderer->UpdateCamera();
        }
        if(glfw->key.isPressed(GLFW_KEY_Y)) //switch up vector to y/-y
        {
            vec3f up = vec3f(.0f, -1.f, .0f) * renderer->camera.getUp();
            up.y = up.y == .0f ? 1.f : up.y;
            renderer->camera.setUpVector(up);
            renderer->UpdateCamera();
        }
        if(glfw->key.isPressed(GLFW_KEY_Z)) //switch up vector to z/-z
        {
            vec3f up = vec3f(.0f, .0f, -1.f) * renderer->camera.getUp();
            up.z = up.z == .0f ? 1.f : up.z;
            renderer->camera.setUpVector(up);
            renderer->UpdateCamera();
        }
        if(glfw->key.isPressed(GLFW_KEY_H)) //"h"
        {
            heatMapMode = ++heatMapMode % 4;
            renderer->ResetAccumulation();
        }
        // Camera movement
        if (glfw->mouseState.leftButtonDown)
            LeftMouseDrag(owl::vec2i(glfw->mouseState.position), owl::vec2i(glfw->mouseState.delta));

        if (glfw->mouseState.rightButtonDown)
            RightMouseDrag(owl::vec2i(glfw->mouseState.position), owl::vec2i(glfw->mouseState.delta));

        if (glfw->mouseState.middleButtonDown || glfw->key.isDown(GLFW_KEY_LEFT_ALT))
            CenterMouseDrag(owl::vec2i(glfw->mouseState.position), owl::vec2i(glfw->mouseState.delta));
#else   
        if(renderer->frameID < wuFrames)
            printf("Remaining warm-up ");
        else
        {
            printf("Rendered ");
            if( nthFrame != 0 && (renderer->frameID - wuFrames) % nthFrame == 0 )
                TakeSnapshot(outputFileName + "_" + modeString + "_w_" 
                    + std::to_string(renderer->frameID - wuFrames) + 
                    + "_shadows_" + (renderer->enableShadows ? "on" : "off") +
                    "_frames.png");
        }
        printf("frame(s) %u\n", owl::abs(renderer->frameID - (int)wuFrames));
#endif
    }
#if OFFLINE_VIEWER
//print scene info
    printf("============ Scene Info ============\n");
    printf("Number of frames: %u\n", renderer->frameID - wuFrames);
    printf("Shadowing: %s\n", renderer->enableShadows ? "on" : "off");
    printf("Macrocell dimesions: %u, %u, %u\n", renderer->macrocellDims.x, renderer->macrocellDims.y, renderer->macrocellDims.z);
    printf("Mode: %s\n", modeString.c_str());
    for(int i = 0; i < numFiles; i++)
    {
        printf("Volume #%d size: %ux%ux%u (%u MiB)\n", 
            i, renderer->rawPtrs[i]->getDims().x, renderer->rawPtrs[i]->getDims().y,
            renderer->rawPtrs[i]->getDims().z, renderer->rawPtrs[i]->getDims().x *
            renderer->rawPtrs[i]->getDims().y * renderer->rawPtrs[i]->getDims().z *
            renderer->rawPtrs[i]->getBytesPerVoxel() / 1024 / 1024);
    }
    printf("avg. fps: %.3f (%0.4f sec)\n", 1.0f/renderer->avgTime, renderer->avgTime);
    printf("best. fps: %.3f (%0.4f sec)\n", 1.0f/renderer->minTime, renderer->minTime);
//write the frame as number of frames taken
    TakeSnapshot(outputFileName + "_" + modeString + "_w_" 
    + std::to_string(renderer->frameID - wuFrames) + 
    + "_shadows_" + (renderer->enableShadows ? "on" : "off") +
    "_frames.png");
#endif
    renderer->Terminate();
#if !OFFLINE_VIEWER
    glfw->destroyWindow();
#endif
}

int main(int argc, char *argv[])
{
    Viewer viewer(argc, argv);
    viewer.Run();
    return 0;
}
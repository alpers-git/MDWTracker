#include <iostream>

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

    void TakeSnapshot(std::string filename = "frame.png");
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
    bool heatmap = false;

    friend class dtracker::Renderer;
};

Viewer::Viewer(int argc, char *argv[])
{
    // invert image
    stbi_flip_vertically_on_write(true);
    
    // parse arguments
    argparse::ArgumentParser program("DTracker Viewer");

    program.add_argument("-du", "--umesh-data")
        .help("path to the .umesh file");
    program.add_argument("-dr", "--raw-data")
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
        .scan<'u', unsigned int>();
    program.add_argument("-bg", "--background")
        .help("background color")
        .nargs(3)
        .scan<'g', float>();

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
    GLFWHandler::getInstance()->initWindow(720, 720, "DTracker Viewer");

    // init imgui
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    // Initialize ImGui
    InitImGui();

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
        auto mc = program.get<unsigned int>("-mc");
        printf("mc: %d\n", mc);
        if (mc < 1)
        {
            std::cerr << "Number of macrocells per side must be positive" << std::endl;
            std::exit(1);
        }
        renderer->macrocellsPerSide = mc;
    }
    if (program.is_used("-bg"))
    {
        auto bg = program.get<std::vector<float>>("-bg");
        renderer->bgColor = vec3f(bg[0], bg[1], bg[2]);
    }
    if(program.is_used("-du"))
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto umeshHdlPtr = umesh::io::loadBinaryUMesh(program.get<std::string>("-du"));
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken to load umesh: " << duration.count() << " milliseconds" << std::endl;
        std::cout << "found " << umeshHdlPtr->tets.size() << " tetrahedra" << std::endl;
        std::cout << "found " << umeshHdlPtr->pyrs.size() << " pyramids" << std::endl;
        std::cout << "found " << umeshHdlPtr->wedges.size() << " wedges" << std::endl;
        std::cout << "found " << umeshHdlPtr->hexes.size() << " hexahedra" << std::endl;
        std::cout << "found " << umeshHdlPtr->vertices.size() << " vertices" << std::endl;
        renderer->umeshPtr = umeshHdlPtr;
        numFiles++;
    }
    else if(program.is_used("-dr"))
    {
        auto paths = program.get<std::vector<std::string>>("-dr");
        for (auto path : paths)
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto rawFile = std::make_shared<raw::RawR>(path.c_str(), "rb");
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Time taken to load raw data: " << duration.count() << " milliseconds" << std::endl;
            std::cout << "found " << rawFile->getDims().x << " x " << rawFile->getDims().y << " x " << rawFile->getDims().z << " voxels and " << rawFile->getBytesPerVoxel() << " byte(s) per voxel" << std::endl;
            std::cout << "total size: " << rawFile->getDims().x * rawFile->getDims().y * rawFile->getDims().z * rawFile->getBytesPerVoxel() / 1024.0f / 1024.0f << " MB" << std::endl;
            renderer->rawPtrs.push_back(rawFile);
            numFiles++;
        }
    }
    else
    {
        std::cerr << "No data file given" << std::endl;
        std::exit(1);
    }
    
    // init transfer function widgets
    tfnWidgets = std::vector<ImTF::TransferFunctionWidget>(numFiles);
    // if tf argument is given, load it
    if (program.is_used("-t"))
    {
        auto tfStates = program.get<std::vector<std::string>>("-t");
        for (int i = 0; i < tfStates.size(); i++)
        {
            tfnWidgets[i].LoadState(tfStates[i]);
        }
    }

    manipulator = std::make_shared<camera::Manipulator>(&(renderer->camera));
}

void Viewer::TakeSnapshot(std::string filename)
{
    const uint32_t *fb = (const uint32_t *)
        owlBufferGetPointer(renderer->frameBuffer, 0);
    stbi_write_png(filename.c_str(),
                   renderer->fbSize.x, renderer->fbSize.y, 4,
                   fb, renderer->fbSize.x * sizeof(uint32_t));
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
    GLFWHandler *glfw = GLFWHandler::getInstance();
    renderer->Init(!camGivenAsParam);
    renderer->UpdateCamera();
    while (!glfw->windowShouldClose())
    {
        glfw->pollEvents();

        glfw->mouseState.imGuiPolling = ImGui::GetIO().WantCaptureMouse;

        renderer->Render(heatmap);

        renderer->Update();

        glfw->draw((const uint32_t *)
                       owlBufferGetPointer(renderer->frameBuffer, 0));

        RequestImGuiFrame();
        ImGui::Begin("Renderer Controls");
        // write the fps metrics as a colored text in imgui
        ImGui::Text("avg. fps:");
        ImGui::SameLine();
        ImVec4 color = renderer->avgTime > 0.6f ? red : renderer->avgTime < 0.11f ? green : orange;
        ImGui::TextColored(color," %.3f (%0.3f sec)", 1.0f/renderer->avgTime, renderer->avgTime);
        ImGui::Text("min. fps:");
        ImGui::SameLine();
        color = renderer->minTime > 0.6f ? red : renderer->minTime < 0.11f ? green : orange;
        ImGui::TextColored(color," %.3f (%0.3f sec)", 1.0f/renderer->minTime, renderer->minTime);

        static bool tabChanged = false;
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
                    tabChanged = selectedTF != i;
                    selectedTF = i;
                }
            }
            ImGui::EndTabBar();
        }
        if(ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if(ImGui::Checkbox("Shadows", &(renderer->enableShadows)))
                renderer->SetLightDirection(renderer->lightDir); // to reset accumulation
            static float lightIntensity = renderer->lightIntensity;
            if(ImGui::DragFloat("Light Intensity", &lightIntensity, 0.01f, 0.0f, 1e20f))
                renderer->SetLightIntensity(lightIntensity);
            if(renderer->enableShadows)
            {
                static float ambient = renderer->ambient;
                if (ImGui::DragFloat("Ambient", &ambient, 0.01f, 0.0f, 10.0f))
                    renderer->SetAmbient(ambient);
                static vec3f lightDir = renderer->lightDir;
                if(ImGui::DragFloat3("Light Direction", &lightDir.x, 0.01f, -1.0f, 1.0f))
                    renderer->SetLightDirection(lightDir);
            }
        }
        float dt = renderer->dt;
        if(ImGui::DragFloat("dt", &dt, 0.005f, 0.0f, 1e20f, "%.5f"))
            renderer->SetDt(dt);
        ImGui::SameLine();
        if(ImGui::Button("Reset Dt"))
            renderer->ResetDt();
        if(ImGui::Checkbox("Heatmap", &heatmap))
            renderer->ResetAccumulation();
        ImGui::SameLine();
        if(ImGui::Checkbox("Accumulation", &(renderer->enableAccumulation)))
            renderer->ResetAccumulation();
        ImGui::End();
        RenderImGuiFrame();

        if(tabChanged || tfnWidgets[selectedTF].ColorMapChanged())
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

        if(tabChanged || tfnWidgets[selectedTF].OpacityScaleChanged())
            renderer->SetXFOpacityScale(tfnWidgets[selectedTF].GetOpacityScale(), selectedTF);
            
        if(tabChanged || tfnWidgets[selectedTF].RangeChanged())
            renderer->SetXFRange(vec2f(tfnWidgets[selectedTF].GetRange().x,
                    tfnWidgets[selectedTF].GetRange().y), selectedTF);
        

        glfw->swapBuffers();

        //==== Input Processing ====

        // Taking a snapshot of the current frame
        if (glfw->key.isPressed(GLFW_KEY_1) &&
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"!"
            TakeSnapshot();

        if (glfw->key.isPressed(GLFW_KEY_EQUAL) &&
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"+"
            renderer->SetDt(renderer->dt + 0.01f );
        else if (glfw->key.isPressed(GLFW_KEY_MINUS) &&
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"-"
            renderer->SetDt(renderer->dt - 0.01f);

        else if (glfw->key.isRepeated(GLFW_KEY_EQUAL) &&
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"+"
            renderer->SetDt(renderer->dt + 0.04f);
        else if (glfw->key.isRepeated(GLFW_KEY_MINUS) &&
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"-"
            renderer->SetDt(renderer->dt - 0.04f);

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
            heatmap = !heatmap;
            renderer->ResetAccumulation();
        }
        // Camera movement
        if (glfw->mouseState.leftButtonDown)
            LeftMouseDrag(owl::vec2i(glfw->mouseState.position), owl::vec2i(glfw->mouseState.delta));

        if (glfw->mouseState.rightButtonDown)
            RightMouseDrag(owl::vec2i(glfw->mouseState.position), owl::vec2i(glfw->mouseState.delta));

        if (glfw->mouseState.middleButtonDown)
            CenterMouseDrag(owl::vec2i(glfw->mouseState.position), owl::vec2i(glfw->mouseState.delta));
    }
    renderer->Terminate();
    glfw->destroyWindow();
}

int main(int argc, char *argv[])
{
    Viewer viewer(argc, argv);
    viewer.Run();
    return 0;
}
#include <iostream>

#define TFN_WIDGET_NO_STB_IMAGE_IMPL
#include "transfer_function_widget.h"

#include "glfwHandler.h"
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

private:
    std::shared_ptr<dtracker::Renderer> renderer;
    std::shared_ptr<camera::Manipulator> manipulator;
    std::shared_ptr<ImTF::TransferFunctionWidget> tfnWidget;

    // imgui
    void InitImGui();
    void RequestImGuiFrame();
    void RenderImGuiFrame();

    const float kbd_rotate_degrees = 100.f;
    const float degrees_per_drag_fraction = 250;
    const float pixels_per_move = 90.f;

    friend class dtracker::Renderer;
};

Viewer::Viewer(int argc, char *argv[])
{
    // invert image
    stbi_flip_vertically_on_write(true);
    
    // parse arguments
    argparse::ArgumentParser program("rqs-viewer");

    program.add_argument("-d", "--data")
        .help("path to the umesh file")
        .required();
    program.add_argument("-c", "--camera")
        .help("camera pos<x,y,z>, gaze<x,y,z>, up<x,y,z>, cosfovy(degrees)")
        .nargs(10)
        .scan<'g', float>();
    program.add_argument("-t", "--transfer-function")
        .help("path to the .tf file");

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

    // init renderer and open window
    auto start = std::chrono::high_resolution_clock::now();
    auto umeshHdlPtr = umesh::io::loadBinaryUMesh(program.get<std::string>("-d"));
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
    std::cout << "found " << umeshHdlPtr->tets.size() << " tetrahedra" << std::endl;
    std::cout << "found " << umeshHdlPtr->pyrs.size() << " pyramids" << std::endl;
    std::cout << "found " << umeshHdlPtr->wedges.size() << " wedges" << std::endl;
    std::cout << "found " << umeshHdlPtr->hexes.size() << " hexahedra" << std::endl;
    std::cout << "found " << umeshHdlPtr->vertices.size() << " vertices" << std::endl;
    renderer = std::make_shared<dtracker::Renderer>();

    GLFWHandler::getInstance()->initWindow(512, 512, "RQS-Viewer");

    // init imgui
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    // Initialize ImGui
    InitImGui();

    // Initialize transfer function editor
    tfnWidget = std::make_shared<ImTF::TransferFunctionWidget>();
    // if tf argument is given, load it
    if (program.is_used("-t"))
    {
        tfnWidget->LoadState(program.get<std::string>("-t"));
    }

    renderer->umeshPtr = umeshHdlPtr;
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

void Viewer::Run()
{
    GLFWHandler *glfw = GLFWHandler::getInstance();
    renderer->Init();
    renderer->UpdateCamera();
    while (!glfw->windowShouldClose())
    {
        glfw->pollEvents();

        glfw->mouseState.imGuiPolling = ImGui::GetIO().WantCaptureMouse;

        renderer->Render();

        renderer->Update();

        glfw->draw((const uint32_t *)
                       owlBufferGetPointer(renderer->frameBuffer, 0));

        RequestImGuiFrame();
        ImGui::Begin("Renderer Controls");
        tfnWidget->DrawColorMap();
        tfnWidget->DrawOpacityScale();
        tfnWidget->DrawRanges();
        float dt = renderer->dt;
        if(ImGui::DragFloat("dt", &dt, 0.01f, 0.0f, 1e20f))
            renderer->SetDt(dt);
        vec3f lightDir = renderer->lightDir;
        if(ImGui::DragFloat3("lightDir", &lightDir.x, 0.01f, -1.0f, 1.0f));
            //renderer->SetLightDir(lightDir);
        ImGui::End();
        RenderImGuiFrame();

        if(tfnWidget->ColorMapChanged())
        {
            auto cm = tfnWidget->GetColormapf();
            std::vector<owl::vec4f> colorMapVec;
            for (int i = 0; i < cm.size(); i += 4)
            {
                colorMapVec.push_back(owl::vec4f(cm[i],
                                        cm[i + 1], cm[i + 2], cm[i + 3]));
            }
            renderer->SetXFColormap(colorMapVec);
        }

        if(tfnWidget->OpacityScaleChanged())
            renderer->SetXFOpacityScale(tfnWidget->GetOpacityScale());
            
        if(tfnWidget->RangeChanged())
            renderer->SetXFRange(vec2f(tfnWidget->GetRange().x, tfnWidget->GetRange().y));
        

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
            tfnWidget->SaveState("tfn_state.tf");

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
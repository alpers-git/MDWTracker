#include <iostream>

#include "glfwHandler.h"

#include <argparse/argparse.hpp>
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

private:
    std::shared_ptr<dtracker::Renderer> renderer;
};

Viewer::Viewer(int argc, char *argv[])
{
    // parse arguments
    argparse::ArgumentParser program("rqs-viewer");

    program.add_argument("-d", "--data")
        .help("path to the umesh file")
        .required();
    program.add_argument("-c", "--camera")
        .help("camera pos<x,y,z>, gaze<x,y,z>, up<x,y,z>, cosfovy(degrees)")
        .nargs(10)
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

    // init renderer and open window
    auto start = std::chrono::high_resolution_clock::now();
    auto umeshHdlPtr = umesh::io::loadBinaryUMesh(program.get<std::string>("-d"));
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
    renderer = std::make_shared<dtracker::Renderer>();
    GLFWHandler::getInstance()->initWindow(1024, 1024, "RQS-Viewer");
    renderer->umeshPtr = umeshHdlPtr;
}

void Viewer::TakeSnapshot(std::string filename)
{
    const uint32_t *fb = (const uint32_t *)
        owlBufferGetPointer(renderer->frameBuffer, 0);
    stbi_write_png(filename.c_str(),
                   renderer->fbSize.x, renderer->fbSize.y, 4,
                   fb, renderer->fbSize.x * sizeof(uint32_t));
}

void Viewer::Run()
{
    GLFWHandler *glfw = GLFWHandler::getInstance();
    renderer->Init();
    while (!glfw->windowShouldClose())
    {
        glfw->pollEvents();

        renderer->Render();

        renderer->Update();

        glfw->draw((const uint32_t *)
                       owlBufferGetPointer(renderer->frameBuffer, 0));

        glfw->swapBuffers();

        // Taking a snapshot of the current frame
        if (glfw->key.isPressed(GLFW_KEY_1) &&
            glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //"!"
            TakeSnapshot();
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
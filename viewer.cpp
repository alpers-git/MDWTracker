#include <iostream>

#include "glfwHandler.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "renderer.h"

struct Viewer
{
public:
    /* data */
    Viewer(/* args */);
    void Run();
private:
    dtracker::Renderer* renderer;
};

Viewer::Viewer(/* args */)
{
    renderer = new dtracker::Renderer();
    GLFWHandler::getInstance()->initWindow(1024, 1024, "RQS-Viewer");
}

void Viewer::Run()
{
    GLFWHandler* glfw = GLFWHandler::getInstance();
    while (!glfw->windowShouldClose())
    {
        glfw->pollEvents();
        renderer->Render();
        renderer->Update();
        glfw->swapBuffers();
    }
    glfw->destroyWindow();
}


int main(int argc, char* argv[])
{
    Viewer viewer;
    viewer.Run();
    return 0;
}
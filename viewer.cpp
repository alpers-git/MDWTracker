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
    //renderer->Resize(vec2i(1024,1024));
}

void Viewer::Run()
{
    GLFWHandler* glfw = GLFWHandler::getInstance();
    renderer->Init();
    while (!glfw->windowShouldClose())
    {
        glfw->pollEvents();
        renderer->Render();
        const uint32_t *fb = 
            (const uint32_t *)owlBufferGetPointer(renderer->frameBuffer, 0);
        glfw->draw(fb);
        renderer->Update();
        glfw->swapBuffers();
    }
    renderer->Terminate();
    glfw->destroyWindow();
}


int main(int argc, char* argv[])
{
    Viewer viewer;
    viewer.Run();
    return 0;
}
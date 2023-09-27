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
    void TakeSnapshot();
private:
    std::shared_ptr<dtracker::Renderer> renderer;
};

Viewer::Viewer(/* args */)
{
    renderer = std::make_shared<dtracker::Renderer>();
    GLFWHandler::getInstance()->initWindow(1024, 1024, "RQS-Viewer");
}

void Viewer::TakeSnapshot()
{
    const uint32_t *fb = (const uint32_t *)
            owlBufferGetPointer(renderer->frameBuffer, 0);
    stbi_write_png(std::string("frame.png").c_str(), 
            renderer->fbSize.x, renderer->fbSize.y, 4,
            fb, renderer->fbSize.x * sizeof(uint32_t));
}

void Viewer::Run()
{
    GLFWHandler* glfw = GLFWHandler::getInstance();
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


int main(int argc, char* argv[])
{
    Viewer viewer;
    viewer.Run();
    return 0;
}
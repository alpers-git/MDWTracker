#include <iostream>

#ifdef __linux__ 
#include "GL/gl.h"
#endif
#include "GLFW/glfw3.h"

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
    Renderer* renderer;
};

Viewer::Viewer(/* args */)
{
    renderer = new Renderer();
}

void Viewer::Run()
{
    glfwInit();
    glfwCreateWindow(1920, 720, "potato", NULL, NULL);
    while (true)
    {
        renderer->Render();
        renderer->Update();
    }
}


int main(int argc, char* argv[])
{
    Viewer viewer;
    viewer.Run();
    return 0;
}
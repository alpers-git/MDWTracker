#include <iostream>

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
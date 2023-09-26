#pragma once

#include <iostream>

#include "camera.h"

#include "owl/owl.h"

namespace dtracker
{
class Renderer
{
private:
    Camera camera;
public:
    /* data */
    OWLRayGen  rayGen  { 0 };
    OWLBuffer  accumBuffer { 0 };

    Renderer(/* args */);
    ~Renderer();
    void Render();
    void Update();
    void Resize();
};

} // namespace dtracker
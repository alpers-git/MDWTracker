#pragma once

#include <iostream>
#include "owl/owl.h"

class Renderer
{
private:
    /* data */
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
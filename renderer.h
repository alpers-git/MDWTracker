#pragma once

#include <iostream>

#include "camera.h"

#include "owl/owl.h"
#include "owl/owl_host.h"
#include "umesh/io/UMesh.h"

#include "deviceCode.h"

namespace dtracker
{
class Renderer
{
private:
    Camera camera;
public:
    Renderer();
    ~Renderer();

    std::shared_ptr<umesh::UMesh> umeshPtr;

    /* raygen */
    OWLRayGen  rayGen       { 0 };
    OWLParams  lp           { 0 };
    
    /* owl */
    OWLContext context      { 0 };
    OWLModule module;

    /* geom */
    OWLBuffer vertexBuffer;
    OWLBuffer indexBuffer;

    /* frame */
    OWLBuffer  accumBuffer  { 0 };
    OWLBuffer  frameBuffer  { 0 };
    int        accumID      { 0 };
    int        frameID      { 0 };

    /* scene */
    float dt = 0.5f;
    bool shadows = false;
    vec3f lightDir = vec3f(0.f,-1.f,0.f);
    vec2i fbSize = vec2i(1024,1024);

    /*! initializes renderer */
    void Init();
    /*! renders the scene, visualizes heatmaps if param is true*/
    void Render(bool heatMap = false);
    /*! updates the launch params on device */
    void Update();
    /*! terminates the renderer*/
    void Terminate();
    /*! resizes the frame buffer */
    void Resize(vec2i newSize);
    
};

} // namespace dtracker
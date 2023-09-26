#include "renderer.h"

#include "glfwHandler.h"

#define LOG(message)                                          \
  std::cout << OWL_TERMINAL_BLUE;                             \
  std::cout << "#owl.sample(main): " << message << std::endl; \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                       \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                       \
  std::cout << "#owl.sample(main): " << message << std::endl; \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_ERROR(message)                                    \
  std::cout << OWL_TERMINAL_RED;                              \
  std::cout << "#owl.sample(main): " << message << std::endl; \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

OWLVarDecl rayGenVars[] = {
    {nullptr /* sentinel to mark end of list */}};

OWLVarDecl launchParamVars[] = {
    // framebuffer
    {"fbPtr", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, fbPtr)},
    {"fbSize", OWL_INT2, OWL_OFFSETOF(LaunchParams, fbSize)},
    {"shadows", OWL_BOOL, OWL_OFFSETOF(LaunchParams, shadows)},
    {"lightDir", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, lightDir)},
    // accum buffer
    {"accumID", OWL_INT, OWL_OFFSETOF(LaunchParams, accumID)},
    {"accumBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, accumBuffer)},
    {"frameID", OWL_INT, OWL_OFFSETOF(LaunchParams, frameID)},
    {"triangleTLAS", OWL_GROUP, OWL_OFFSETOF(LaunchParams, triangleTLAS)},
    // camera
    {"camera.org", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.origin)},
    {"camera.llc", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.lower_left_corner)},
    {"camera.horiz", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.horizontal)},
    {"camera.vert", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.vertical)},
    // Volume data
    {"volume.elementTLAS", OWL_GROUP, OWL_OFFSETOF(LaunchParams, volume.elementTLAS)},
    {"volume.macrocellTLAS", OWL_GROUP, OWL_OFFSETOF(LaunchParams, volume.macrocellTLAS)},
    {"volume.macrocellDims", OWL_INT3, OWL_OFFSETOF(LaunchParams, volume.macrocellDims)},
    {"volume.dt", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, volume.dt)},
    {"volume.globalBoundsLo", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, volume.globalBoundsLo)},
    {"volume.globalBoundsHi", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, volume.globalBoundsHi)},
    // transfer function
    {"transferFunction.xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction.xf)},
    {"transferFunction.volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction.volumeDomain)},
    {"transferFunction.opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction.opacityScale)},
    //{"volume.mecrocells"}
    {/* sentinel to mark end of list */}};

namespace dtracker
{

Renderer::Renderer()
{}

Renderer::~Renderer()
{}

void Renderer::Init()
{
    // Init owl
    LOG("Initializing owl...");
    context = owlContextCreate(nullptr, 1);
    module = owlModuleCreate(context,deviceCode_ptx);
    owlContextSetRayTypeCount(context, 1);
    
    LOG("Creating programs...");
    rayGen = owlRayGenCreate(context, module, "testRayGen",
                             sizeof(RayGenData),
                             rayGenVars, -1);

    OWLVarDecl missProgVars[] = {
        {"color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color0)},
        {"color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color1)},
        {/* sentinel to mark end of list */}};
    // ----------- create object  ----------------------------
    OWLMissProg missProg = owlMissProgCreate(context, module, "miss", sizeof(MissProgData),
                                             missProgVars, -1);
    owlMissProgSet3f(missProg, "color0", owl3f{.2f, .2f, .26f});
    owlMissProgSet3f(missProg, "color1", owl3f{.1f, .1f, .16f});

    lp = owlParamsCreate(context, sizeof(LaunchParams), launchParamVars, -1);

    owlBuildPrograms(context);

    LOG("Setting buffers ...");
    frameBuffer = owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y);
    if (!accumBuffer)
      accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(accumBuffer, fbSize.x * fbSize.y);
    owlParamsSetBuffer(lp, "accumBuffer", accumBuffer);
    
    accumID = 0;
    owlParamsSet1i(lp, "accumID", accumID);
    frameID = 0;
    owlParamsSet1i(lp, "frameID", frameID);

    LOG("Building geometries ...");

    owlParamsSetBuffer(lp, "fbPtr", frameBuffer);
    owlParamsSet2i(lp, "fbSize", (const owl2i &)fbSize);

    cudaDeviceSynchronize();

    LOG("Building programs...");
    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);
}

void Renderer::Render(bool heatMap)
{
    owlBuildSBT(context);
    owlLaunch2D(rayGen, fbSize.x, fbSize.y, lp);

    owlParamsSet1i(lp, "accumID", accumID++);
    owlParamsSet1i(lp, "frameID", frameID++);
}

void Renderer::Update()
{
    auto glfw = GLFWHandler::getInstance();
    if (glfw->getWindowSize() != fbSize)
      Resize(glfw->getWindowSize());
}

void Renderer::Terminate()
{
    LOG("Terminating...\n");
    owlContextDestroy(context);
}

void Renderer::Resize(const vec2i newSize)
  {
    fbSize = newSize;
    owlBufferResize(frameBuffer, fbSize.x * fbSize.y);
    owlParamsSet2i(lp, "fbSize", (const owl2i &)fbSize);
    if (!accumBuffer)
      accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(accumBuffer, fbSize.x * fbSize.y);
    owlParamsSetBuffer(lp, "accumBuffer", accumBuffer);
    //OnCameraChange();
  }

} // namespace dtracker
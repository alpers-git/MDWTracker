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
    UpdateCamera();
  }

  void Renderer::UpdateCamera()
  {
    const vec3f lookFrom = camera.getFrom();
    const vec3f lookAt = camera.getAt();
    const vec3f lookUp = camera.getUp();
    const float cosFovy = camera.getCosFovy();
    const float vfov = toDegrees(acosf(cosFovy));
    // ........... compute variable values  ..................
    const vec3f vup = lookUp;
    const float aspect = fbSize.x / float(fbSize.y);
    const float theta = vfov * ((float)M_PI) / 180.0f;
    const float half_height = tanf(theta / 2.0f);
    const float half_width = aspect * half_height;
    const float focusDist = 10.f;
    const vec3f origin = lookFrom;
    const vec3f w = normalize(lookFrom - lookAt);
    const vec3f u = normalize(cross(vup, w));
    const vec3f v = cross(w, u);
    const vec3f lower_left_corner = origin - half_width * focusDist * u - half_height * focusDist * v - focusDist * w;
    const vec3f horizontal = 2.0f * half_width * focusDist * u;
    const vec3f vertical = 2.0f * half_height * focusDist * v;
    camera.motionSpeed = umesh::length(umeshPtr->getBounds().size()) / 50.f;

    accumID = 0;

    // ----------- set variables  ----------------------------
    owlParamsSetGroup(lp, "triangleTLAS", triangleTLAS);
    owlParamsSet3f(lp, "camera.org", (const owl3f &)origin);
    owlParamsSet3f(lp, "camera.llc", (const owl3f &)lower_left_corner);
    owlParamsSet3f(lp, "camera.horiz", (const owl3f &)horizontal);
    owlParamsSet3f(lp, "camera.vert", (const owl3f &)vertical);
    owlParamsSet1i(lp, "accumID", accumID);
  }

} // namespace dtracker
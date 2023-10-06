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

// test data for cube

const int NUM_VERTICES = 8;
vec3f vertices[NUM_VERTICES] =
    {
        {-1.f, -1.f, -1.f},
        {+1.f, -1.f, -1.f},
        {-1.f, +1.f, -1.f},
        {+1.f, +1.f, -1.f},
        {-1.f, -1.f, +1.f},
        {+1.f, -1.f, +1.f},
        {-1.f, +1.f, +1.f},
        {+1.f, +1.f, +1.f}};

const int NUM_INDICES = 12;
vec3i indices[NUM_INDICES] =
    {
        {0, 1, 3}, {2, 3, 0}, {5, 7, 6}, {5, 6, 4}, {0, 4, 5}, {0, 5, 1}, {2, 3, 7}, {2, 7, 6}, {1, 5, 7}, {1, 7, 3}, {4, 0, 2}, {4, 2, 6}};

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
  {
  }

  Renderer::~Renderer()
  {
  }

  void Renderer::Init()
  {
    // Init owl
    LOG("Initializing owl...");
    context = owlContextCreate(nullptr, 1);
    module = owlModuleCreate(context, deviceCode_ptx);
    owlContextSetRayTypeCount(context, 1);

    LOG("Creating programs...");
    rayGen = owlRayGenCreate(context, module, "testRayGen",
                             sizeof(RayGenData),
                             rayGenVars, -1);

    OWLVarDecl missProgVars[] = {
        {"color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color0)},
        {"color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color1)},
        {/* sentinel to mark end of list */}};

    OWLMissProg missProg = owlMissProgCreate(context, module, "miss", sizeof(MissProgData),
                                             missProgVars, -1);
    owlMissProgSet3f(missProg, "color0", owl3f{.2f, .2f, .26f});
    owlMissProgSet3f(missProg, "color1", owl3f{.1f, .1f, .16f});

    lp = owlParamsCreate(context, sizeof(LaunchParams), launchParamVars, -1);
    // -------------------------------------------------------
    // declare geometry types
    // -------------------------------------------------------
    // Different intersection programs for different element types
    OWLVarDecl unstructuredElementVars[] = {
        {"tetrahedra", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, tetrahedra)},
        {"pyramids", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, pyramids)},
        {"hexahedra", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, hexahedra)},
        {"wedges", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, wedges)},
        {"bytesPerIndex", OWL_UINT, OWL_OFFSETOF(UnstructuredElementData, bytesPerIndex)},
        {"vertices", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, vertices)},
        {"scalars", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, scalars)},
        {"offset", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, offset)},
        {"numTetrahedra", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numTetrahedra)},
        {"numPyramids", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numPyramids)},
        {"numWedges", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numWedges)},
        {"numHexahedra", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numHexahedra)},
        {"maxima", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, maxima)},
        //{"bboxes", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, bboxes)},
        {/* sentinel to mark end of list */}};

    OWLVarDecl triangleVars[] = {
        {"triVertices", OWL_BUFPTR, OWL_OFFSETOF(TriangleData, vertices)},
        {"indices", OWL_BUFPTR, OWL_OFFSETOF(TriangleData, indices)},
        {"color", OWL_FLOAT3, OWL_OFFSETOF(TriangleData, color)},
        {/* sentinel to mark end of list */}};

    OWLVarDecl macrocellVars[] = {
        {"bboxes", OWL_BUFPTR, OWL_OFFSETOF(MacrocellData, bboxes)},
        {"maxima", OWL_BUFPTR, OWL_OFFSETOF(MacrocellData, maxima)},
        {/* sentinel to mark end of list */}};

    // Declare the geometry types
    //macrocellType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(MacrocellData), macrocellVars, -1);
    tetrahedraType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    pyramidType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    wedgeType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    hexahedraType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    triangleType = owlGeomTypeCreate(context, OWL_GEOMETRY_TRIANGLES, sizeof(TriangleData), triangleVars, -1);


    // Set intersection programs
    //owlGeomTypeSetIntersectProg(macrocellType, /*ray type */ 1, module, "MacrocellIntersection");
    //owlGeomTypeSetIntersectProg(macrocellType, /*ray type */ 0, module, "VolumeIntersection");
    owlGeomTypeSetIntersectProg(tetrahedraType, /*ray type */ 0, module, "TetrahedraPointQuery");
    owlGeomTypeSetIntersectProg(pyramidType, /*ray type */ 0, module, "PyramidPointQuery");
    owlGeomTypeSetIntersectProg(wedgeType, /*ray type */ 0, module, "WedgePointQuery");
    owlGeomTypeSetIntersectProg(hexahedraType, /*ray type */ 0, module, "HexahedraPointQuery");

    // Set boundary programs
    owlGeomTypeSetBoundsProg(tetrahedraType, module, "TetrahedraBounds");
    owlGeomTypeSetBoundsProg(pyramidType, module, "PyramidBounds");
    owlGeomTypeSetBoundsProg(wedgeType, module, "WedgeBounds");
    owlGeomTypeSetBoundsProg(hexahedraType, module, "HexahedraBounds");
    //owlGeomTypeSetBoundsProg(macrocellType, module, "MacrocellBounds");

    owlGeomTypeSetClosestHit(triangleType, /*ray type */ 0, module, "triangle_test");
    
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

    owlParamsSetBuffer(lp, "fbPtr", frameBuffer);
    owlParamsSet2i(lp, "fbSize", (const owl2i &)fbSize);

    // transfer function
    SetXFOpacityScale(1.0f);
    volDomain = interval<float>({umeshPtr->getBounds4f().lower.w, umeshPtr->getBounds4f().upper.w});
    owlParamsSet4f(lp, "volume.globalBoundsLo",
                   owl4f{umeshPtr->getBounds4f().lower.x, umeshPtr->getBounds4f().lower.y,
                         umeshPtr->getBounds4f().lower.z, umeshPtr->getBounds4f().lower.w});
    owlParamsSet4f(lp, "volume.globalBoundsHi",
                   owl4f{umeshPtr->getBounds4f().upper.x, umeshPtr->getBounds4f().upper.y,
                         umeshPtr->getBounds4f().upper.z, umeshPtr->getBounds4f().upper.w});
    printf("volDomain: %f %f\n", volDomain.lower, volDomain.upper);
    owlParamsSet2f(lp, "transferFunction.volumeDomain", owl2f{volDomain.lower, volDomain.upper});

    // camera
    auto center = umeshPtr->getBounds().center();
    vec3f eye = vec3f(center.x, center.y, center.z + 2.5f * (umeshPtr->getBounds().upper.z - umeshPtr->getBounds().lower.z));
    camera.setOrientation(eye, vec3f(center.x, center.y, center.z), vec3f(0, 1, 0), 45.0f);
    UpdateCamera();

    // Allocate buffers for volume data
    tetrahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->tets.size() * 4, nullptr);
    pyramidsData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->pyrs.size() * 5, nullptr);
    wedgesData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->wedges.size() * 6, nullptr);
    hexahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->hexes.size() * 8, nullptr);
    verticesData = owlDeviceBufferCreate(context, OWL_FLOAT3, umeshPtr->vertices.size(), nullptr);
    scalarData = owlDeviceBufferCreate(context, OWL_FLOAT, umeshPtr->perVertex->values.size(), nullptr);

    // Upload data
    owlBufferUpload(tetrahedraData, umeshPtr->tets.data());
    owlBufferUpload(pyramidsData, umeshPtr->pyrs.data());
    owlBufferUpload(wedgesData, umeshPtr->wedges.data());
    owlBufferUpload(hexahedraData, umeshPtr->hexes.data());
    owlBufferUpload(verticesData, umeshPtr->vertices.data());
    owlBufferUpload(scalarData, umeshPtr->perVertex->values.data());

    indexBuffer = owlDeviceBufferCreate(context, OWL_INT3, NUM_INDICES, indices);
    vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, NUM_VERTICES, vertices);

    LOG("Building geometries ...");

    cudaDeviceSynchronize();
    // Surface geometry
    //just loads a cube for now
    trianglesGeom = owlGeomCreate(context, triangleType);
    owlTrianglesSetIndices(trianglesGeom, indexBuffer, NUM_INDICES, sizeof(vec3i), 0);
    owlTrianglesSetVertices(trianglesGeom, vertexBuffer, NUM_VERTICES, sizeof(vec3f), 0);
    owlGeomSetBuffer(trianglesGeom, "indices", indexBuffer);
    owlGeomSetBuffer(trianglesGeom, "triVertices", vertexBuffer);
    owlGeomSet3f(trianglesGeom, "color", owl3f{0, 1, 1});

    trianglesGroup = owlTrianglesGeomGroupCreate(context, 1, &trianglesGeom);
    owlGroupBuildAccel(trianglesGroup);
    triangleTLAS = owlInstanceGroupCreate(context, 1, &trianglesGroup);
    owlGroupBuildAccel(triangleTLAS);

    // Volume geometry
    if (umeshPtr->tets.size() > 0)
    {
      OWLGeom tetrahedraGeom = owlGeomCreate(context, tetrahedraType);
      owlGeomSetPrimCount(tetrahedraGeom, umeshPtr->tets.size() * 4);
      owlGeomSetBuffer(tetrahedraGeom, "tetrahedra", tetrahedraData);
      owlGeomSetBuffer(tetrahedraGeom, "vertices", verticesData);
      owlGeomSetBuffer(tetrahedraGeom, "scalars", scalarData);
      owlGeomSet1ul(tetrahedraGeom, "offset", 0);
      owlGeomSet1ui(tetrahedraGeom, "bytesPerIndex", 4);
      owlGeomSet1ul(tetrahedraGeom, "numTetrahedra", umeshPtr->tets.size());
      owlGeomSet1ul(tetrahedraGeom, "numPyramids", umeshPtr->pyrs.size());
      owlGeomSet1ul(tetrahedraGeom, "numWedges", umeshPtr->wedges.size());
      owlGeomSet1ul(tetrahedraGeom, "numHexahedra", umeshPtr->hexes.size());
      OWLGroup tetBLAS = owlUserGeomGroupCreate(context, 1, &tetrahedraGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
      owlGroupBuildAccel(tetBLAS);
      elementBLAS.push_back(tetBLAS);
      elementGeom.push_back(tetrahedraGeom);
    }
    if (umeshPtr->pyrs.size() > 0)
    {
      OWLGeom pyramidGeom = owlGeomCreate(context, pyramidType);
      owlGeomSetPrimCount(pyramidGeom, umeshPtr->pyrs.size());
      owlGeomSetBuffer(pyramidGeom, "pyramids", pyramidsData);
      owlGeomSetBuffer(pyramidGeom, "vertices", verticesData);
      owlGeomSetBuffer(pyramidGeom, "scalars", scalarData);
      owlGeomSet1ul(pyramidGeom, "offset", 0);
      owlGeomSet1ui(pyramidGeom, "bytesPerIndex", 4);
      owlGeomSet1ul(pyramidGeom, "numTetrahedra", umeshPtr->tets.size());
      owlGeomSet1ul(pyramidGeom, "numPyramids", umeshPtr->pyrs.size());
      owlGeomSet1ul(pyramidGeom, "numWedges", umeshPtr->wedges.size());
      owlGeomSet1ul(pyramidGeom, "numHexahedra", umeshPtr->hexes.size());
      OWLGroup pyramidBLAS = owlUserGeomGroupCreate(context, 1, &pyramidGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
      owlGroupBuildAccel(pyramidBLAS);
      elementBLAS.push_back(pyramidBLAS);
      elementGeom.push_back(pyramidGeom);
    }

    if (umeshPtr->wedges.size() > 0)
    {
      OWLGeom wedgeGeom = owlGeomCreate(context, wedgeType);
      owlGeomSetPrimCount(wedgeGeom, umeshPtr->wedges.size());
      owlGeomSetBuffer(wedgeGeom, "wedges", wedgesData);
      owlGeomSetBuffer(wedgeGeom, "vertices", verticesData);
      owlGeomSetBuffer(wedgeGeom, "scalars", scalarData);
      owlGeomSet1ul(wedgeGeom, "offset", 0);
      owlGeomSet1ui(wedgeGeom, "bytesPerIndex", 4);
      owlGeomSet1ul(wedgeGeom, "numTetrahedra", umeshPtr->tets.size());
      owlGeomSet1ul(wedgeGeom, "numPyramids", umeshPtr->pyrs.size());
      owlGeomSet1ul(wedgeGeom, "numWedges", umeshPtr->wedges.size());
      owlGeomSet1ul(wedgeGeom, "numHexahedra", umeshPtr->hexes.size());
      OWLGroup wedgeBLAS = owlUserGeomGroupCreate(context, 1, &wedgeGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
      owlGroupBuildAccel(wedgeBLAS);
      elementBLAS.push_back(wedgeBLAS);
      elementGeom.push_back(wedgeGeom);
    }

    if (umeshPtr->hexes.size() > 0)
    {
      OWLGeom hexahedraGeom = owlGeomCreate(context, hexahedraType);
      owlGeomSetPrimCount(hexahedraGeom, umeshPtr->hexes.size());
      owlGeomSetBuffer(hexahedraGeom, "hexahedra", hexahedraData);
      owlGeomSetBuffer(hexahedraGeom, "vertices", verticesData);
      owlGeomSetBuffer(hexahedraGeom, "scalars", scalarData);
      owlGeomSet1ul(hexahedraGeom, "offset", 0);
      owlGeomSet1ui(hexahedraGeom, "bytesPerIndex", 4);
      owlGeomSet1ul(hexahedraGeom, "numTetrahedra", umeshPtr->tets.size());
      owlGeomSet1ul(hexahedraGeom, "numPyramids", umeshPtr->pyrs.size());
      owlGeomSet1ul(hexahedraGeom, "numWedges", umeshPtr->wedges.size());
      owlGeomSet1ul(hexahedraGeom, "numHexahedra", umeshPtr->hexes.size());
      OWLGroup hexBLAS = owlUserGeomGroupCreate(context, 1, &hexahedraGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
      owlGroupBuildAccel(hexBLAS);
      elementBLAS.push_back(hexBLAS);
      elementGeom.push_back(hexahedraGeom);
    }

    elementTLAS = owlInstanceGroupCreate(context, elementBLAS.size(), nullptr, nullptr, nullptr, OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
    for (int i = 0; i < elementBLAS.size(); ++i)
    {
      size_t peak = 0;
      size_t final = 0;
      owlInstanceGroupSetChild(elementTLAS, i, elementBLAS[i]);
      owlGroupGetAccelSize(elementBLAS[i], &final, &peak);
    }
    owlGroupBuildAccel(elementTLAS);
    owlParamsSetGroup(lp, "volume.elementTLAS", elementTLAS);

    size_t peak = 0;
    size_t final = 0;
    owlGroupGetAccelSize(elementTLAS, &final, &peak);

    LOG("Building programs...");
    //owlBuildPrograms(context);
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

  void Renderer::SetXFColormap(std::vector<vec4f> newCM)
  {
    for (uint32_t i = 0; i < newCM.size(); ++i)
    {
      newCM[i].w = powf(newCM[i].w, 3.f);
    }

    this->colorMap = newCM;
    if (!colorMapBuffer)
      colorMapBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4,
                                             newCM.size(), nullptr);
    owlBufferUpload(colorMapBuffer, newCM.data());

    if (colorMapTexture != 0)
    {
      (cudaDestroyTextureObject(colorMapTexture));
      colorMapTexture = 0;
    }

    cudaResourceDesc res_desc = {};
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();

    // cudaArray_t   voxelArray;
    if (colorMapArray == 0)
    {
      (cudaMallocArray(&colorMapArray,
                       &channel_desc,
                       newCM.size(), 1));
    }

    int pitch = newCM.size() * sizeof(newCM[0]);
    (cudaMemcpy2DToArray(colorMapArray,
                         /* offset */ 0, 0,
                         newCM.data(),
                         pitch, pitch, 1,
                         cudaMemcpyHostToDevice));

    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = colorMapArray;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 0.0f;
    tex_desc.borderColor[1] = 0.0f;
    tex_desc.borderColor[2] = 0.0f;
    tex_desc.borderColor[3] = 0.0f;
    tex_desc.sRGB = 0;
    (cudaCreateTextureObject(&colorMapTexture, &res_desc, &tex_desc,
                             nullptr));

    // OWLTexture xfTexture
    //   = owlTexture2DCreate(owl,OWL_TEXEL_FORMAT_RGBA32F,
    //                        colorMap.size(),1,
    //                        colorMap.data());
    owlParamsSetRaw(lp, "transferFunction.xf", &colorMapTexture);
    accumID = 0;
    owlParamsSet1i(lp, "accumID", accumID);
    //RecalculateDensityRanges();
  }

  void Renderer::SetXFOpacityScale(float newOpacityScale)
  {
    opacityScale = newOpacityScale;
    owlParamsSet1f(lp, "transferFunction.opacityScale", opacityScale);
    accumID = 0;
    owlParamsSet1i(lp, "accumID", accumID);
  }


  void Renderer::SetXFRange(const vec2f newRange)
  {
    range = interval<float>(newRange.x, newRange.y);
    //owlParamsSet2f(lp, "transferFunction.volumeDomain", (const owl2f &)volDomain); TODO: fix this
    accumID = 0;
    owlParamsSet1i(lp, "accumID", accumID);
  }

} // namespace dtracker
#include "renderer.h"
#include <chrono>
#include "owl/helper/cuda.h"
#include "owl/Texture.h"

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
    // renderer variables
    {"enableShadows", OWL_BOOL, OWL_OFFSETOF(LaunchParams, enableShadows)},
    {"heatMapMode", OWL_SHORT, OWL_OFFSETOF(LaunchParams, heatMapMode)},
    {"heatMapScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, heatMapScale)},
    {"enableAccumulation", OWL_BOOL, OWL_OFFSETOF(LaunchParams, enableAccumulation)},
    {"enableGradientShading", OWL_BOOL, OWL_OFFSETOF(LaunchParams, enableGradientShading)},
    {"bgColor", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, bgColor)},
    {"mode",    OWL_INT,    OWL_OFFSETOF(LaunchParams, mode)},
    // light variables
    {"lightDir", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, lightDir)},
    {"lightIntensity", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, lightIntensity)},
    {"ambientIntensity", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, ambient)},
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
    {"volume.numChannels",   OWL_INT,   OWL_OFFSETOF(LaunchParams, volume.numChannels)},
    {"volume.elementTLAS", OWL_GROUP, OWL_OFFSETOF(LaunchParams, volume.elementTLAS)},
    {"volume.macrocellTLAS", OWL_GROUP, OWL_OFFSETOF(LaunchParams, volume.macrocellTLAS)},
    {"volume.rootMacrocellTLAS", OWL_GROUP, OWL_OFFSETOF(LaunchParams, volume.rootMacrocellTLAS)},
    {"volume.macrocellDims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.macrocellDims)},
    {"volume.macrocells",   OWL_BUFPTR,       OWL_OFFSETOF(LaunchParams,volume.macrocells) },
    {"volume.majorants",    OWL_BUFPTR,       OWL_OFFSETOF(LaunchParams,volume.majorants) },
    {"volume.dt", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, volume.dt)},
    {"volume.globalBoundsLo", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, volume.globalBoundsLo)},
    {"volume.globalBoundsHi", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, volume.globalBoundsHi)},
    {"volume.meshType", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.meshType)},
    //    structured volume data
    {"volume.sGrid[0].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[0].scalarTex[0])},
    {"volume.sGrid[0].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[0].scalarTex[1])},
    {"volume.sGrid[0].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[0].dims)},
    {"volume.sGrid[0].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[0].splitAxis)},
    {"volume.sGrid[0].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[0].splitPos)},
    {"volume.sGrid[0].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[0].varyingDims)},
    {"volume.sGrid[0].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[0].varyingDim)},
    
    {"volume.sGrid[1].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[1].scalarTex[0])},
    {"volume.sGrid[1].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[1].scalarTex[1])},
    {"volume.sGrid[1].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[1].dims)},
    {"volume.sGrid[1].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[1].splitAxis)},
    {"volume.sGrid[1].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[1].splitPos)},
    {"volume.sGrid[1].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[1].varyingDims)},
    {"volume.sGrid[1].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[1].varyingDim)},

    {"volume.sGrid[2].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[2].scalarTex[0])},
    {"volume.sGrid[2].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[2].scalarTex[1])},
    {"volume.sGrid[2].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[2].dims)},
    {"volume.sGrid[2].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[2].splitAxis)},
    {"volume.sGrid[2].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[2].splitPos)},
    {"volume.sGrid[2].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[2].varyingDims)},
    {"volume.sGrid[2].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[2].varyingDim)},

    {"volume.sGrid[3].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[3].scalarTex[0])},
    {"volume.sGrid[3].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[3].scalarTex[1])},
    {"volume.sGrid[3].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[3].dims)},
    {"volume.sGrid[3].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[3].splitAxis)},
    {"volume.sGrid[3].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[3].splitPos)},
    {"volume.sGrid[3].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[3].varyingDims)},
    {"volume.sGrid[3].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[3].varyingDim)},

    {"volume.sGrid[4].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[4].scalarTex[0])},
    {"volume.sGrid[4].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[4].scalarTex[1])},
    {"volume.sGrid[4].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[4].dims)},
    {"volume.sGrid[4].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[4].splitAxis)},
    {"volume.sGrid[4].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[4].splitPos)},
    {"volume.sGrid[4].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[4].varyingDims)},
    {"volume.sGrid[4].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[4].varyingDim)},

    {"volume.sGrid[5].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[5].scalarTex[0])},
    {"volume.sGrid[5].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[5].scalarTex[1])},
    {"volume.sGrid[5].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[5].dims)},
    {"volume.sGrid[5].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[5].splitAxis)},
    {"volume.sGrid[5].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[5].splitPos)},
    {"volume.sGrid[5].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[5].varyingDims)},
    {"volume.sGrid[5].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[5].varyingDim)},

    {"volume.sGrid[6].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[6].scalarTex[0])},
    {"volume.sGrid[6].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[6].scalarTex[1])},
    {"volume.sGrid[6].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[6].dims)},
    {"volume.sGrid[6].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[6].splitAxis)},
    {"volume.sGrid[6].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[6].splitPos)},
    {"volume.sGrid[6].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[6].varyingDims)},
    {"volume.sGrid[6].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[6].varyingDim)},

    {"volume.sGrid[7].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[7].scalarTex[0])},
    {"volume.sGrid[7].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[7].scalarTex[1])},
    {"volume.sGrid[7].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[7].dims)},
    {"volume.sGrid[7].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[7].splitAxis)},
    {"volume.sGrid[7].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[7].splitPos)},
    {"volume.sGrid[7].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[7].varyingDims)},
    {"volume.sGrid[7].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[7].varyingDim)},

    {"volume.sGrid[8].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[8].scalarTex[0])},
    {"volume.sGrid[8].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[8].scalarTex[1])},
    {"volume.sGrid[8].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[8].dims)},
    {"volume.sGrid[8].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[8].splitAxis)},
    {"volume.sGrid[8].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[8].splitPos)},
    {"volume.sGrid[8].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[8].varyingDims)},
    {"volume.sGrid[8].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[8].varyingDim)},

    {"volume.sGrid[9].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[9].scalarTex[0])},
    {"volume.sGrid[9].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[9].scalarTex[1])},
    {"volume.sGrid[9].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[9].dims)},
    {"volume.sGrid[9].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[9].splitAxis)},
    {"volume.sGrid[9].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[9].splitPos)},
    {"volume.sGrid[9].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[9].varyingDims)},
    {"volume.sGrid[9].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[9].varyingDim)},

    {"volume.sGrid[10].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[10].scalarTex[0])},
    {"volume.sGrid[10].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[10].scalarTex[1])},
    {"volume.sGrid[10].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[10].dims)},
    {"volume.sGrid[10].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[10].splitAxis)},
    {"volume.sGrid[10].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[10].splitPos)},
    {"volume.sGrid[10].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[10].varyingDims)},
    {"volume.sGrid[10].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[10].varyingDim)},

    {"volume.sGrid[11].scalarTex[0]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[11].scalarTex[0])},
    {"volume.sGrid[11].scalarTex[1]",OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, volume.sGrid[11].scalarTex[1])},
    {"volume.sGrid[11].dims", OWL_UINT3, OWL_OFFSETOF(LaunchParams, volume.sGrid[11].dims)},
    {"volume.sGrid[11].splitAxis", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[11].splitAxis)},
    {"volume.sGrid[11].splitPos", OWL_UINT, OWL_OFFSETOF(LaunchParams, volume.sGrid[11].splitPos)},
    {"volume.sGrid[11].varyingDims", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, volume.sGrid[11].varyingDims)},
    {"volume.sGrid[11].varyingDim", OWL_INT, OWL_OFFSETOF(LaunchParams, volume.sGrid[11].varyingDim)},

    // transfer functions (IM SORRY)
    {"transferFunction[0].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[0].xf)},
    {"transferFunction[0].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[0].volumeDomain)},
    {"transferFunction[0].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[0].opacityScale)},
    {"transferFunction[0].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[0].xfDomain)},
    {"transferFunction[1].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[1].xf)},
    {"transferFunction[1].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[1].volumeDomain)},
    {"transferFunction[1].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[1].opacityScale)},
    {"transferFunction[1].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[1].xfDomain)},
    {"transferFunction[2].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[2].xf)},
    {"transferFunction[2].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[2].volumeDomain)},
    {"transferFunction[2].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[2].opacityScale)},
    {"transferFunction[2].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[2].xfDomain)},
    {"transferFunction[3].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[3].xf)},
    {"transferFunction[3].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[3].volumeDomain)},
    {"transferFunction[3].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[3].opacityScale)},
    {"transferFunction[3].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[3].xfDomain)},
    {"transferFunction[4].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[4].xf)},
    {"transferFunction[4].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[4].volumeDomain)},
    {"transferFunction[4].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[4].opacityScale)},
    {"transferFunction[4].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[4].xfDomain)},
    {"transferFunction[5].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[5].xf)},
    {"transferFunction[5].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[5].volumeDomain)},
    {"transferFunction[5].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[5].opacityScale)},
    {"transferFunction[5].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[5].xfDomain)},
    {"transferFunction[6].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[6].xf)},
    {"transferFunction[6].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[6].volumeDomain)},
    {"transferFunction[6].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[6].opacityScale)},
    {"transferFunction[6].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[6].xfDomain)},
    {"transferFunction[7].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[7].xf)},
    {"transferFunction[7].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[7].volumeDomain)},
    {"transferFunction[7].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[7].opacityScale)},
    {"transferFunction[7].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[7].xfDomain)},
    {"transferFunction[8].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[8].xf)},
    {"transferFunction[8].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[8].volumeDomain)},
    {"transferFunction[8].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[8].opacityScale)},
    {"transferFunction[8].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[8].xfDomain)},
    {"transferFunction[9].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[9].xf)},
    {"transferFunction[9].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[9].volumeDomain)},
    {"transferFunction[9].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[9].opacityScale)},
    {"transferFunction[9].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[9].xfDomain)},
    {"transferFunction[10].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[10].xf)},
    {"transferFunction[10].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[10].volumeDomain)},
    {"transferFunction[10].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[10].opacityScale)},
    {"transferFunction[10].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[10].xfDomain)},
    {"transferFunction[11].xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction[11].xf)},
    {"transferFunction[11].volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[11].volumeDomain)},
    {"transferFunction[11].opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction[11].opacityScale)},
    {"transferFunction[11].xfDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction[11].xfDomain)},
    {/* sentinel to mark end of list */}};

    cudaTextureObject_t create3DTexture(float* data, vec3i dims)
    {
      //Create texture for scalars
        cudaTextureObject_t volumeTexture;
        cudaResourceDesc res_desc = {};
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();

        cudaArray_t   voxelArray;
        cudaMalloc3DArray(&voxelArray,
                          &channel_desc,
                          make_cudaExtent(dims.x,
                                          dims.y,
                                          dims.z));
        OWL_CUDA_SYNC_CHECK();
        cudaMemcpy3DParms copyParams = {0};
        cudaExtent volumeSize = make_cudaExtent(dims.x,
                                                dims.y,
                                                dims.z);
        copyParams.srcPtr
          = make_cudaPitchedPtr((void *)data,
                                volumeSize.width
                                * sizeof(float),
                                volumeSize.width,
                                volumeSize.height);
        copyParams.dstArray = voxelArray;
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        
        cudaResourceDesc            texRes;
        memset(&texRes,0,sizeof(cudaResourceDesc));
        
        texRes.resType            = cudaResourceTypeArray;
        texRes.res.array.array    = voxelArray;
        
        cudaTextureDesc             texDescr;
        memset(&texDescr,0,sizeof(cudaTextureDesc));
        
        texDescr.normalizedCoords = true; // access with normalized texture coordinates
        texDescr.filterMode       = cudaFilterModeLinear; // linear interpolation
        // wrap texture coordinates
        texDescr.addressMode[0] = cudaAddressModeClamp;//Wrap;
        texDescr.addressMode[1] = cudaAddressModeClamp;//Wrap;
        texDescr.addressMode[2] = cudaAddressModeClamp;//Wrap;
        texDescr.sRGB                = 0;

        // texDescr.addressMode[0]      = cudaAddressModeBorder;
        // texDescr.addressMode[1]      = cudaAddressModeBorder;
        texDescr.maxAnisotropy       = 1;
        texDescr.maxMipmapLevelClamp = 0;
        texDescr.minMipmapLevelClamp = 0;
        texDescr.mipmapFilterMode    = cudaFilterModePoint;
        texDescr.borderColor[0]      = 0.0f;
        texDescr.borderColor[1]      = 0.0f;
        texDescr.borderColor[2]      = 0.0f;
        texDescr.borderColor[3]      = 0.0f;
        texDescr.readMode = cudaReadModeElementType;
    
        cudaCreateTextureObject(&volumeTexture, &texRes, &texDescr, NULL);
        return volumeTexture;
    } 

namespace dtracker
{

  Renderer::Renderer()
  {
  }

  Renderer::~Renderer()
  {
  }

  void Renderer::Init(const unsigned int mode, bool autoSetCamera)
  {
    if(umeshPtrs.size() == 0 && rawPtrs.size() == 0)
    {
      LOG_ERROR("No data source specified!");
      return;
    }
    this->mode = (Mode)mode;
    // Init owl
    LOG("Initializing owl...");
    context = owlContextCreate(nullptr, 1);
    module = owlModuleCreate(context, deviceCode_ptx);
    owlContextSetRayTypeCount(context, 1);

    LOG("Creating programs...");
      rayGen = owlRayGenCreate(context, module, "mainRG",
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

    if(umeshPtrs.size() > 0)
    {
      meshType = MeshType::UMESH;
      tetrahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtrs[0]->tets.size() * 4, nullptr);
      pyramidsData = owlDeviceBufferCreate(context, OWL_INT, umeshPtrs[0]->pyrs.size() * 5, nullptr);
      wedgesData = owlDeviceBufferCreate(context, OWL_INT, umeshPtrs[0]->wedges.size() * 6, nullptr);
      hexahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtrs[0]->hexes.size() * 8, nullptr);
      verticesData = owlDeviceBufferCreate(context, OWL_FLOAT3, umeshPtrs[0]->vertices.size(), nullptr);
      scalarData[0] = owlDeviceBufferCreate(context, OWL_FLOAT, umeshPtrs[0]->perVertex->values.size(), nullptr);//!!
      owlBufferUpload(tetrahedraData, umeshPtrs[0]->tets.data());
      owlBufferUpload(pyramidsData, umeshPtrs[0]->pyrs.data());
      owlBufferUpload(wedgesData, umeshPtrs[0]->wedges.data());
      owlBufferUpload(hexahedraData, umeshPtrs[0]->hexes.data());
      owlBufferUpload(verticesData, umeshPtrs[0]->vertices.data());
      owlBufferUpload(scalarData[0], umeshPtrs[0]->perVertex->values.data());//!!

      if(macrocellDims.x == 0 || macrocellDims.y == 0 || macrocellDims.z == 0)
      {
        auto tmp = CalculateMCGridDims(8);
        for(int i = 0; i < 3; i++)
          if(macrocellDims[i] == 0)
            macrocellDims[i] = tmp[i];
      }

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
          { "maxima", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, maxima)},
          { "bboxes", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, bboxes)},
          {/* sentinel to mark end of list */}};

      OWLVarDecl triangleVars[] = {
          {"triVertices", OWL_BUFPTR, OWL_OFFSETOF(TriangleData, vertices)},
          {"indices", OWL_BUFPTR, OWL_OFFSETOF(TriangleData, indices)},
          {"color", OWL_FLOAT3, OWL_OFFSETOF(TriangleData, color)},
          {/* sentinel to mark end of list */}};

      OWLVarDecl macrocellVars[] = {
          {"bboxes", OWL_BUFPTR, OWL_OFFSETOF(MacrocellData, bboxes)},
          {"maxima", OWL_BUFPTR, OWL_OFFSETOF(MacrocellData, maxima)},
          {"offset", OWL_INT, OWL_OFFSETOF(MacrocellData, offset)},
          {/* sentinel to mark end of list */}};

      // Declare the geometry types
      macrocellType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(MacrocellData), macrocellVars, -1);
      tetrahedraType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
      pyramidType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
      wedgeType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
      hexahedraType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
      triangleType = owlGeomTypeCreate(context, OWL_GEOMETRY_TRIANGLES, sizeof(TriangleData), triangleVars, -1);


      // Set intersection programs
      //owlGeomTypeSetIntersectProg(macrocellType, /*ray type */ 1, module, "MacrocellIntersection");
      owlGeomTypeSetIntersectProg(macrocellType, /*ray type */ 0, module, "volumeIntersection");
      owlGeomTypeSetIntersectProg(tetrahedraType, /*ray type */ 0, module, "tetrahedraPointQuery");
      owlGeomTypeSetIntersectProg(pyramidType, /*ray type */ 0, module, "pyramidPointQuery");
      owlGeomTypeSetIntersectProg(wedgeType, /*ray type */ 0, module, "wedgePointQuery");
      owlGeomTypeSetIntersectProg(hexahedraType, /*ray type */ 0, module, "hexahedraPointQuery");

      // Set boundary programs
      owlGeomTypeSetBoundsProg(tetrahedraType, module, "tetrahedraBounds");
      owlGeomTypeSetBoundsProg(pyramidType, module, "pyramidBounds");
      owlGeomTypeSetBoundsProg(wedgeType, module, "wedgeBounds");
      owlGeomTypeSetBoundsProg(hexahedraType, module, "hexahedraBounds");
      owlGeomTypeSetBoundsProg(macrocellType, module, "macrocellBounds");

      owlGeomTypeSetClosestHit(triangleType, /*ray type */ 0, module, "triangleCH");
      owlGeomTypeSetClosestHit(macrocellType, /*ray type*/ 0, module, "adaptiveDTCH");
      
      owlBuildPrograms(context);

      tfdatas.push_back(TFData());// push one empty transfer function
      tfdatas[0].volDomain = interval<float>({umeshPtrs[0]->getBounds4f().lower.w, umeshPtrs[0]->getBounds4f().upper.w});
      printf("volume domain: %f %f\n", tfdatas[0].volDomain.lower, tfdatas[0].volDomain.upper);
      owlParamsSet4f(lp, "volume.globalBoundsLo",
                    owl4f{umeshPtrs[0]->getBounds4f().lower.x, umeshPtrs[0]->getBounds4f().lower.y,
                          umeshPtrs[0]->getBounds4f().lower.z, umeshPtrs[0]->getBounds4f().lower.w});
      owlParamsSet4f(lp, "volume.globalBoundsHi",
                    owl4f{umeshPtrs[0]->getBounds4f().upper.x, umeshPtrs[0]->getBounds4f().upper.y,
                          umeshPtrs[0]->getBounds4f().upper.z, umeshPtrs[0]->getBounds4f().upper.w});

      // camera
      if(autoSetCamera)
      {
        auto center = umeshPtrs[0]->getBounds().center();
        vec3f eye = vec3f(center.x, center.y, center.z + 2.5f * (umeshPtrs[0]->getBounds().upper.z - umeshPtrs[0]->getBounds().lower.z));
        camera.setOrientation(eye, vec3f(center.x, center.y, center.z), vec3f(0, 1, 0), 45.0f);
        camera.setFocalDistance(umesh::length(umeshPtrs[0]->getBounds().size()) / 2.f);
        UpdateCamera();
      }

      LOG("Setting buffers ...");
      // Allocate buffers for volume data
      tetrahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtrs[0]->tets.size() * 4, nullptr);
      pyramidsData = owlDeviceBufferCreate(context, OWL_INT, umeshPtrs[0]->pyrs.size() * 5, nullptr);
      wedgesData = owlDeviceBufferCreate(context, OWL_INT, umeshPtrs[0]->wedges.size() * 6, nullptr);
      hexahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtrs[0]->hexes.size() * 8, nullptr);
      verticesData = owlDeviceBufferCreate(context, OWL_FLOAT3, umeshPtrs[0]->vertices.size(), nullptr);
      scalarData[0] = owlDeviceBufferCreate(context, OWL_FLOAT, umeshPtrs[0]->perVertex->values.size(), nullptr);//!!

      // Upload data
      owlBufferUpload(tetrahedraData, umeshPtrs[0]->tets.data());
      owlBufferUpload(pyramidsData, umeshPtrs[0]->pyrs.data());
      owlBufferUpload(wedgesData, umeshPtrs[0]->wedges.data());
      owlBufferUpload(hexahedraData, umeshPtrs[0]->hexes.data());
      owlBufferUpload(verticesData, umeshPtrs[0]->vertices.data());
      owlBufferUpload(scalarData[0], umeshPtrs[0]->perVertex->values.data());//!!

      indexBuffer = owlDeviceBufferCreate(context, OWL_INT3, umeshPtrs[0]->triangles.size() * 3, nullptr);
      vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, umeshPtrs[0]->vertices.size(), nullptr);

      // Macrocell data
      int numMacrocells = 1;
      std::vector<box4f> bboxes;
      bboxes.resize(numMacrocells);
      auto bb = umeshPtrs[0]->getBounds4f();
      bboxes[0] = box4f(vec4f(bb.lower.x, bb.lower.y, bb.lower.z, bb.lower.w), vec4f(bb.upper.x, bb.upper.y, bb.upper.z, bb.upper.w));

      size_t gMaximaBufSize = macrocellDims.x*macrocellDims.y*macrocellDims.z * 
          ( mode <= Mode::MULTI_ALT ? rawPtrs.size() : mode <= Mode::MIX ? 1 : 0 );
      gridMaximaBuffer = owlDeviceBufferCreate(context, OWL_FLOAT, 
          gMaximaBufSize, nullptr);
      //clusterMaximaBuffer = owlDeviceBufferCreate(context, OWL_FLOAT, numClusters, nullptr);
      owlParamsSetBuffer(lp, "volume.majorants", gridMaximaBuffer);

      rootMaximaBuffer = owlDeviceBufferCreate(context, OWL_FLOAT, numMacrocells, nullptr);
      rootBBoxBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(box4f), numMacrocells, nullptr);
      owlBufferUpload(rootBBoxBuffer, bboxes.data());
      
      box3f bounds = {
          {umeshPtrs[0]->bounds.lower.x, umeshPtrs[0]->bounds.lower.y, umeshPtrs[0]->bounds.lower.z},
          {umeshPtrs[0]->bounds.upper.x, umeshPtrs[0]->bounds.upper.y, umeshPtrs[0]->bounds.upper.z}
        };
      if(mode < Mode::MARCHER_MULTI)
        macrocellsBuffer = buildSpatialMacrocells({int(macrocellDims.x), int(macrocellDims.y), int(macrocellDims.z)}, bounds);
      owlParamsSetBuffer(lp, "volume.macrocells", macrocellsBuffer);
      //const uint3 macrocellDims = {macrocellDims, macrocellDims, macrocellDims};

      owlParamsSet3ui(lp, "volume.macrocellDims", (const owl3ui &)macrocellDims);

      LOG("Building geometries ...");

      cudaDeviceSynchronize();
      // Surface geometry
      trianglesGeom = owlGeomCreate(context, triangleType);
      if(umeshPtrs[0]->triangles.size() > 0)
      {
        owlBufferUpload(indexBuffer, umeshPtrs[0]->triangles.data());
        owlBufferUpload(vertexBuffer, umeshPtrs[0]->vertices.data());
        owlTrianglesSetIndices(trianglesGeom, indexBuffer, umeshPtrs[0]->triangles.size(), sizeof(vec3i), 0);
        owlTrianglesSetVertices(trianglesGeom, vertexBuffer, umeshPtrs[0]->triangles.size(), sizeof(vec3f), 0);
        owlGeomSetBuffer(trianglesGeom, "indices", indexBuffer);
        owlGeomSetBuffer(trianglesGeom, "triVertices", vertexBuffer);
        trianglesGroup = owlTrianglesGeomGroupCreate(context, 1, &trianglesGeom);
        owlGroupBuildAccel(trianglesGroup);
        triangleTLAS = owlInstanceGroupCreate(context, 1, &trianglesGroup);
        owlGroupBuildAccel(triangleTLAS);
      }
      owlGeomSet3f(trianglesGeom, "color", owl3f{0, 1, 1});

      // Macrocell geometry
      OWLGeom macrocellGeom = owlGeomCreate(context, macrocellType);
      owlGeomSetPrimCount(macrocellGeom, numMacrocells);
      owlGeomSet1i(macrocellGeom, "offset", 0);
      owlGeomSetBuffer(macrocellGeom, "maxima", rootMaximaBuffer);
      owlGeomSetBuffer(macrocellGeom, "bboxes", rootBBoxBuffer);
      
      rootMacrocellBLAS = owlUserGeomGroupCreate(context, numMacrocells, &macrocellGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
      owlGroupBuildAccel(rootMacrocellBLAS);
      rootMacrocellTLAS = owlInstanceGroupCreate(context, 1, nullptr, nullptr, nullptr, OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
      owlInstanceGroupSetChild(rootMacrocellTLAS, 0, rootMacrocellBLAS); 
      owlGroupBuildAccel(rootMacrocellTLAS);
      owlParamsSetGroup(lp, "volume.rootMacrocellTLAS", rootMacrocellTLAS);

      // Volume geometry
      if (umeshPtrs[0]->tets.size() > 0)
      {
        OWLGeom tetrahedraGeom = owlGeomCreate(context, tetrahedraType);
        owlGeomSetPrimCount(tetrahedraGeom, umeshPtrs[0]->tets.size() * 4);
        owlGeomSetBuffer(tetrahedraGeom, "tetrahedra", tetrahedraData);
        owlGeomSetBuffer(tetrahedraGeom, "vertices", verticesData);
        owlGeomSetBuffer(tetrahedraGeom, "scalars", scalarData[0]);
        owlGeomSet1ul(tetrahedraGeom, "offset", 0);
        owlGeomSet1ui(tetrahedraGeom, "bytesPerIndex", 4);
        owlGeomSet1ul(tetrahedraGeom, "numTetrahedra", umeshPtrs[0]->tets.size());
        owlGeomSet1ul(tetrahedraGeom, "numPyramids", umeshPtrs[0]->pyrs.size());
        owlGeomSet1ul(tetrahedraGeom, "numWedges", umeshPtrs[0]->wedges.size());
        owlGeomSet1ul(tetrahedraGeom, "numHexahedra", umeshPtrs[0]->hexes.size());
        OWLGroup tetBLAS = owlUserGeomGroupCreate(context, 1, &tetrahedraGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
        owlGroupBuildAccel(tetBLAS);
        elementBLAS.push_back(tetBLAS);
        elementGeom.push_back(tetrahedraGeom);
      }
      if (umeshPtrs[0]->pyrs.size() > 0)
      {
        OWLGeom pyramidGeom = owlGeomCreate(context, pyramidType);
        owlGeomSetPrimCount(pyramidGeom, umeshPtrs[0]->pyrs.size());
        owlGeomSetBuffer(pyramidGeom, "pyramids", pyramidsData);
        owlGeomSetBuffer(pyramidGeom, "vertices", verticesData);
        owlGeomSetBuffer(pyramidGeom, "scalars", scalarData[0]);
        owlGeomSet1ul(pyramidGeom, "offset", 0);
        owlGeomSet1ui(pyramidGeom, "bytesPerIndex", 4);
        owlGeomSet1ul(pyramidGeom, "numTetrahedra", umeshPtrs[0]->tets.size());
        owlGeomSet1ul(pyramidGeom, "numPyramids", umeshPtrs[0]->pyrs.size());
        owlGeomSet1ul(pyramidGeom, "numWedges", umeshPtrs[0]->wedges.size());
        owlGeomSet1ul(pyramidGeom, "numHexahedra", umeshPtrs[0]->hexes.size());
        OWLGroup pyramidBLAS = owlUserGeomGroupCreate(context, 1, &pyramidGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
        owlGroupBuildAccel(pyramidBLAS);
        elementBLAS.push_back(pyramidBLAS);
        elementGeom.push_back(pyramidGeom);
      }

      if (umeshPtrs[0]->wedges.size() > 0)
      {
        OWLGeom wedgeGeom = owlGeomCreate(context, wedgeType);
        owlGeomSetPrimCount(wedgeGeom, umeshPtrs[0]->wedges.size());
        owlGeomSetBuffer(wedgeGeom, "wedges", wedgesData);
        owlGeomSetBuffer(wedgeGeom, "vertices", verticesData);
        owlGeomSetBuffer(wedgeGeom, "scalars", scalarData[0]);
        owlGeomSet1ul(wedgeGeom, "offset", 0);
        owlGeomSet1ui(wedgeGeom, "bytesPerIndex", 4);
        owlGeomSet1ul(wedgeGeom, "numTetrahedra", umeshPtrs[0]->tets.size());
        owlGeomSet1ul(wedgeGeom, "numPyramids", umeshPtrs[0]->pyrs.size());
        owlGeomSet1ul(wedgeGeom, "numWedges", umeshPtrs[0]->wedges.size());
        owlGeomSet1ul(wedgeGeom, "numHexahedra", umeshPtrs[0]->hexes.size());
        OWLGroup wedgeBLAS = owlUserGeomGroupCreate(context, 1, &wedgeGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
        owlGroupBuildAccel(wedgeBLAS);
        elementBLAS.push_back(wedgeBLAS);
        elementGeom.push_back(wedgeGeom);
      }

      if (umeshPtrs[0]->hexes.size() > 0)
      {
        OWLGeom hexahedraGeom = owlGeomCreate(context, hexahedraType);
        owlGeomSetPrimCount(hexahedraGeom, umeshPtrs[0]->hexes.size());
        owlGeomSetBuffer(hexahedraGeom, "hexahedra", hexahedraData);
        owlGeomSetBuffer(hexahedraGeom, "vertices", verticesData);
        owlGeomSetBuffer(hexahedraGeom, "scalars", scalarData[0]);
        owlGeomSet1ul(hexahedraGeom, "offset", 0);
        owlGeomSet1ui(hexahedraGeom, "bytesPerIndex", 4);
        owlGeomSet1ul(hexahedraGeom, "numTetrahedra", umeshPtrs[0]->tets.size());
        owlGeomSet1ul(hexahedraGeom, "numPyramids", umeshPtrs[0]->pyrs.size());
        owlGeomSet1ul(hexahedraGeom, "numWedges", umeshPtrs[0]->wedges.size());
        owlGeomSet1ul(hexahedraGeom, "numHexahedra", umeshPtrs[0]->hexes.size());
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
    }
    //================================================================================================
    else if(rawPtrs.size() > 0)
    {
      meshType = MeshType::RAW;
      if (macrocellDims.x == 0 || macrocellDims.y == 0 || macrocellDims.z == 0)
      {
        auto tmp = CalculateMCGridDims(4);
        for (int i = 0; i < 3; i++)
          if (macrocellDims[i] == 0)
            macrocellDims[i] = tmp[i];
      }

      OWLVarDecl triangleVars[] = {
          {"triVertices", OWL_BUFPTR, OWL_OFFSETOF(TriangleData, vertices)},
          {"indices", OWL_BUFPTR, OWL_OFFSETOF(TriangleData, indices)},
          {"color", OWL_FLOAT3, OWL_OFFSETOF(TriangleData, color)},
          {/* sentinel to mark end of list */}};

      OWLVarDecl macrocellVars[] = {
          {"bboxes", OWL_BUFPTR, OWL_OFFSETOF(MacrocellData, bboxes)},
          {"maxima", OWL_BUFPTR, OWL_OFFSETOF(MacrocellData, maxima)},
          {"offset", OWL_INT, OWL_OFFSETOF(MacrocellData, offset)},
          {/* sentinel to mark end of list */}};

      // Declare the geometry types
      macrocellType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(MacrocellData), macrocellVars, -1);
      triangleType = owlGeomTypeCreate(context, OWL_GEOMETRY_TRIANGLES, sizeof(TriangleData), triangleVars, -1);

      // Set intersection programs
      owlGeomTypeSetIntersectProg(macrocellType, /*ray type */ 0, module, "volumeIntersection");

      // Set boundary programs
      owlGeomTypeSetBoundsProg(macrocellType, module, "macrocellBounds");

      owlGeomTypeSetClosestHit(triangleType, /*ray type */ 0, module, "triangleCH");
      if(mode == Mode::BASELINE)
        owlGeomTypeSetClosestHit(macrocellType, /*ray type*/ 0, module, "baseLineDTCH");
      else if(mode == Mode::MULTI)
        owlGeomTypeSetClosestHit(macrocellType, /*ray type*/ 0, module, "multiMajDTCH");
      else if(mode == Mode::MULTI_ALT)
        owlGeomTypeSetClosestHit(macrocellType, /*ray type*/ 0, module, "altMultiMajDTCH");
      else if(mode == Mode::CUMMULATIVE)
        owlGeomTypeSetClosestHit(macrocellType, /*ray type*/ 0, module, "cummilativeDTCH");
      else if(mode <= Mode::MIX)
        owlGeomTypeSetClosestHit(macrocellType, /*ray type*/ 0, module, "blendDTCH");
      else if(mode == Mode::MARCHER_COMP)
        owlGeomTypeSetClosestHit(macrocellType, /*ray type*/ 0, module, "compRayMarcherCH");
      else
        owlGeomTypeSetClosestHit(macrocellType, /*ray type*/ 0, module, "rayMarcherCH");
      
      owlBuildPrograms(context);
      LOG("Setting buffers ...");
      
      for (size_t i = 0; i < rawPtrs.size(); i++)
      {
        scalarData[i] = owlDeviceBufferCreate(context, OWL_FLOAT, rawPtrs[i]->getDims().x * rawPtrs[i]->getDims().y * rawPtrs[i]->getDims().z, nullptr);
        //get data as void pointer and create vector of floats
        auto data = rawPtrs[i]->getDataVector();
        owlBufferUpload(scalarData[i], data.data());
      }
      // Macrocell data
      int numMacrocells = 1;
      // Although there is only one macrocell, we still use a vector since bufferUpload expects a pointer
      std::vector<box4f> bboxes; 
      bboxes.resize(numMacrocells);
      bboxes[0] = box4f();
      for (size_t i = 0; i < rawPtrs.size(); i++)
        bboxes[0].extend(rawPtrs[i]->getBounds4f());//Extend the bounding box to include all meshes

      size_t gMaximaBufSize = macrocellDims.x*macrocellDims.y*macrocellDims.z * 
          ( mode <= Mode::MULTI_ALT ? rawPtrs.size() : mode <= Mode::MIX ? 1 : 0 );
      gridMaximaBuffer = owlDeviceBufferCreate(context, OWL_FLOAT, 
          gMaximaBufSize, nullptr);
      //clusterMaximaBuffer = owlDeviceBufferCreate(context, OWL_FLOAT, numClusters, nullptr);
      owlParamsSetBuffer(lp, "volume.majorants", gridMaximaBuffer);


      rootMaximaBuffer = owlDeviceBufferCreate(context, OWL_FLOAT, numMacrocells, nullptr);
      rootBBoxBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(box4f), numMacrocells, nullptr);
      owlBufferUpload(rootBBoxBuffer, bboxes.data());
      
      box3f bounds = {
          {bboxes[0].lower.x, bboxes[0].lower.y, bboxes[0].lower.z},
          {bboxes[0].upper.x, bboxes[0].upper.y, bboxes[0].upper.z}
        };

      printf("Cummulative Bounds of %d meshes: %f %f %f %f %f %f\n", rawPtrs.size(), 
              bounds.lower.x, bounds.lower.y, bounds.lower.z, 
              bounds.upper.x, bounds.upper.y, bounds.upper.z);
      if(mode < Mode::MARCHER_MULTI)
      {
        macrocellsBuffer = buildSpatialMacrocells(
            {int(macrocellDims.x), int(macrocellDims.y), int(macrocellDims.z)},
            bounds);
        //macrocellsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(float2), macrocellsPerSide*macrocellsPerSide*macrocellsPerSide, nullptr);
        owlParamsSetBuffer(lp, "volume.macrocells", macrocellsBuffer);
        //const uint3 macrocellDims = {macrocellDims, macrocellDims, macrocellDims};

        owlParamsSet3ui(lp, "volume.macrocellDims", (const owl3ui &)macrocellDims);
      }
      //delete scalar buffers since we don't need them anymore
      for (size_t i = 0; i < rawPtrs.size(); i++)
        owlBufferDestroy(scalarData[i]);

      for (size_t i = 0; i < rawPtrs.size(); i++)
      {
        OWL_CUDA_SYNC_CHECK();
        tfdatas.push_back(TFData());// push one empty transfer function
        tfdatas[i].volDomain = interval<float>({rawPtrs[i]->getBounds4f().lower.w, rawPtrs[i]->getBounds4f().upper.w});
        printf("volume domain: %f %f\n", tfdatas[i].volDomain.lower, tfdatas[i].volDomain.upper);
        owlParamsSet4f(lp, "volume.globalBoundsLo",
                      owl4f{rawPtrs[i]->getBounds4f().lower.x, rawPtrs[i]->getBounds4f().lower.y,
                            rawPtrs[i]->getBounds4f().lower.z, rawPtrs[i]->getBounds4f().lower.w});
        owlParamsSet4f(lp, "volume.globalBoundsHi",
                      owl4f{rawPtrs[i]->getBounds4f().upper.x, rawPtrs[i]->getBounds4f().upper.y,
                            rawPtrs[i]->getBounds4f().upper.z, rawPtrs[i]->getBounds4f().upper.w});
        
        //find the maximum extent of the data and split it into two halves geometrically
        //to create two separate textures
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        const owl3l dims = {static_cast<unsigned long>(rawPtrs[i]->getDims().x),
                              static_cast<unsigned long>(rawPtrs[i]->getDims().y),
                              static_cast<unsigned long>(rawPtrs[i]->getDims().z)};
        //find the maximum extent
        const int maxAxis= dims.x > dims.y ? (dims.x > dims.z ? 0 : 2) : (dims.y > dims.z ? 1 : 2);
        if (dims.x > prop.maxTexture3D[0] || dims.y > prop.maxTexture3D[1] || dims.z > prop.maxTexture3D[2])
        {
          printf("Creating chunked two texture objects...");
          // create two vectors to split the data into two halves
          auto data = rawPtrs[i]->getDataVector();
          std::vector<float> data1, data2;
          cudaTextureObject_t volumeTextChunk1, volumeTextChunk2;
          // split the data into two halves according to the maximum maxAxis
          if (maxAxis == 0)
          {
            data1.resize((dims.x+1)/2  * dims.y * dims.z);
            data2.resize(dims.x/ 2 * dims.y * dims.z);
            for (size_t z = 0; z < dims.z; z++)
              for (size_t y = 0; y < dims.y; y++)
                for (size_t x = 0; x < dims.x; x++)
                {
                  if (x < dims.x / 2)
                    data1[z * dims.y * (dims.x+1) / 2 + y * (dims.x+1) / 2 + x] = data[z * dims.y * dims.x + y * dims.x + x];
                  else
                    data2[z * dims.y * (dims.x / 2) + y * (dims.x / 2) + (x - (dims.x+1) / 2)] = data[z * dims.y * dims.x + y * dims.x + x];
                }
            owlParamsSet1ui(lp, ("volume.sGrid[" + std::to_string(i) + "].splitPos").c_str(), dims.x / 2);
            volumeTextChunk1 = create3DTexture(data1.data(), vec3i((dims.x+1) / 2, dims.y, dims.z));
            volumeTextChunk2 = create3DTexture(data2.data(), vec3i(dims.x / 2, dims.y, dims.z));
          }
          else if (maxAxis == 1)
          {
            data1.resize(dims.x * (dims.y +1) / 2 * dims.z );
            data2.resize(dims.x * dims.y / 2 * dims.z );
            for (size_t z = 0; z < dims.z; z++)
              for (size_t y = 0; y < dims.y; y++)
                for (size_t x = 0; x < dims.x; x++)
                {
                 if (y < dims.y / 2)
                   data1[z * (dims.y + 1) / 2 * dims.x + y * dims.x + x] = data[z * dims.y * dims.x + y * dims.x + x];
                 else
                   data2[z * (dims.y / 2) * dims.x + (y - (dims.y+1) / 2) * dims.x + x] = data[z * dims.y * dims.x + y * dims.x + x];
                }
            owlParamsSet1ui(lp, ("volume.sGrid[" + std::to_string(i) + "].splitPos").c_str(), dims.y / 2);
            volumeTextChunk1 = create3DTexture(data1.data(), vec3i(dims.x, (dims.y + 1) / 2, dims.z));
            volumeTextChunk2 = create3DTexture(data2.data(), vec3i(dims.x, dims.y / 2, dims.z));
          }
          else
          {
            data1.resize(dims.x * dims.y * (dims.z + 1) / 2);
            data2.resize(dims.x * dims.y * dims.z / 2);
            for (size_t z = 0; z < dims.z; z++)
              for (size_t y = 0; y < dims.y; y++)
                for (size_t x = 0; x < dims.x; x++)
                {
                  if (z < dims.z / 2)
                  {
                    data1[z * dims.y * dims.x + y * dims.x + x] = data[z * dims.y * dims.x + y * dims.x + x];
                    //print if data1 size is less than z * dims.y * dims.x + y * dims.x + x
                    if (z * dims.y * dims.x + y * dims.x + x >= data1.size())
                      printf("z: %ld, y: %ld, x: %ld, z * dims.y * dims.x + y * dims.x + x: %ld, data1.size(): %ld\n", z, y, x, z * dims.y * dims.x + y * dims.x + x, data1.size());
                  }
                  else
                    data2[(z - (dims.z+1) / 2) * dims.y * dims.x + y * dims.x + x] = data[z * dims.y * dims.x + y * dims.x + x];
                }
            owlParamsSet1ui(lp, ("volume.sGrid[" + std::to_string(i) + "].splitPos").c_str(), dims.z / 2);
            volumeTextChunk1 = create3DTexture(data1.data(), vec3i(dims.x, dims.y, (dims.z + 1) / 2));
            volumeTextChunk2 = create3DTexture(data2.data(), vec3i(dims.x, dims.y, dims.z / 2));
          }
          printf("Done \n");

          owlParamsSet1ui(lp, ("volume.sGrid[" + std::to_string(i) + "].splitAxis").c_str(), maxAxis);
          owlParamsSetRaw(lp, ("volume.sGrid[" + std::to_string(i) + "].scalarTex[0]").c_str(), &volumeTextChunk1);
          owlParamsSetRaw(lp, ("volume.sGrid[" + std::to_string(i) + "].scalarTex[1]").c_str(), &volumeTextChunk2);
        }
        else
        {
          // get data as void pointer and create vector of floats
          auto data = rawPtrs[i]->getDataVector();
          // owlBufferUpload(scalarData[i], data.data());
          printf("Creating 3D texture object ...");
          cudaTextureObject_t volumeTexture = create3DTexture(data.data(), vec3i(dims.x, dims.y, dims.z));
          printf("Done \n");

          owlParamsSetRaw(lp, ("volume.sGrid[" + std::to_string(i) + "].scalarTex[0]").c_str(), &volumeTexture);
          owlParamsSet1ui(lp, ("volume.sGrid[" + std::to_string(i) + "].splitAxis").c_str(), 3);
        }
        // structured grid data
        owlParamsSet3ui(lp, ("volume.sGrid[" + std::to_string(i) + "].dims").c_str(), {static_cast<unsigned int>(dims.x),
                                                                                      static_cast<unsigned int>(dims.y), 
                                                                                      static_cast<unsigned int>(dims.z)});
      }
      

      // camera
      if(autoSetCamera)
      {
        vec3f center = {0.5f, 0.5f, 0.5f};
        vec3f eye = vec3f(center.x, center.y, center.z + 2.5f);
        camera.setOrientation(eye, vec3f(center.x, center.y, center.z), vec3f(0, 1, 0), 45.0f);
        camera.setFocalDistance(3.f);
        UpdateCamera();
      }

      // indexBuffer = owlDeviceBufferCreate(context, OWL_INT3, umeshPtrs[0]->triangles.size() * 3, nullptr);
      // vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, umeshPtrs[0]->vertices.size(), nullptr);
      
      
      LOG("Building geometries ...");

      cudaDeviceSynchronize();
      // Surface geometry
      trianglesGeom = owlGeomCreate(context, triangleType);
      // if(umeshPtrs[0]->triangles.size() > 0)
      // {
      //   owlBufferUpload(indexBuffer, umeshPtrs[0]->triangles.data());
      //   owlBufferUpload(vertexBuffer, umeshPtrs[0]->vertices.data());
      //   owlTrianglesSetIndices(trianglesGeom, indexBuffer, umeshPtrs[0]->triangles.size(), sizeof(vec3i), 0);
      //   owlTrianglesSetVertices(trianglesGeom, vertexBuffer, umeshPtrs[0]->triangles.size(), sizeof(vec3f), 0);
      //   owlGeomSetBuffer(trianglesGeom, "indices", indexBuffer);
      //   owlGeomSetBuffer(trianglesGeom, "triVertices", vertexBuffer);
      //   owlGeomSet3f(trianglesGeom, "color", owl3f{0, 1, 1});

      //   trianglesGroup = owlTrianglesGeomGroupCreate(context, 1, &trianglesGeom);
      //   owlGroupBuildAccel(trianglesGroup);
      //   triangleTLAS = owlInstanceGroupCreate(context, 1, &trianglesGroup);
      //   owlGroupBuildAccel(triangleTLAS);
      // }

      // Macrocell geometry
      OWLGeom macrocellGeom = owlGeomCreate(context, macrocellType);
      owlGeomSetPrimCount(macrocellGeom, numMacrocells);
      owlGeomSet1i(macrocellGeom, "offset", 0);
      owlGeomSetBuffer(macrocellGeom, "maxima", rootMaximaBuffer);
      owlGeomSetBuffer(macrocellGeom, "bboxes", rootBBoxBuffer);
      
      rootMacrocellBLAS = owlUserGeomGroupCreate(context, numMacrocells, &macrocellGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
      owlGroupBuildAccel(rootMacrocellBLAS);
      rootMacrocellTLAS = owlInstanceGroupCreate(context, 1, nullptr, nullptr, nullptr, OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
      owlInstanceGroupSetChild(rootMacrocellTLAS, 0, rootMacrocellBLAS); 
      owlGroupBuildAccel(rootMacrocellTLAS);
      owlParamsSetGroup(lp, "volume.rootMacrocellTLAS", rootMacrocellTLAS);

      if(varyingDim!=-1)
      {
        std::vector<float> varyingDimsPrefixSum(varyingDims.size(), 0.0f);
        varyingDimsPrefixSum[0] = varyingDims[0];
        for(int i = 1; i < varyingDims.size(); i++)
          varyingDimsPrefixSum[i] = varyingDimsPrefixSum[i-1] + varyingDims[i];
        prefixSummedVariableDims = owlDeviceBufferCreate(context, OWL_FLOAT, varyingDimsPrefixSum.size(), varyingDimsPrefixSum.data());
        //set buffer to the varyingDimsPrefixSum
        owlParamsSetBuffer(lp, "volume.sGrid[0].varyingDims", prefixSummedVariableDims);
        owlParamsSet1i(lp, "volume.sGrid[0].varyingDim", varyingDim);
      }
      else
      {
        owlParamsSet1i(lp, "volume.sGrid[0].varyingDim", -1);
      }
    }

    owlParamsSet1i(lp, "volume.meshType", meshType); //mesh mode

    //framebuffer
    frameBuffer = owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y);
    if (!accumBuffer)
      accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(accumBuffer, fbSize.x * fbSize.y);
    owlParamsSetBuffer(lp, "accumBuffer", accumBuffer);

    frameID = 0;
    owlParamsSet1i(lp, "frameID", frameID);
    ResetAccumulation();

    owlParamsSetBuffer(lp, "fbPtr", frameBuffer);
    owlParamsSet2i(lp, "fbSize", (const owl2i &)fbSize);
    owlParamsSet3f(lp, "bgColor", (const owl3f &)bgColor);

    // light
    owlParamsSet3f(lp, "lightDir", owl3f{lightDir.x, lightDir.y, lightDir.z});
    owlParamsSet1f(lp, "lightIntensity", lightIntensity);
    owlParamsSet1f(lp, "ambientIntensity", ambient);

    // transfer function
    for(size_t i = 0; i < tfdatas.size(); ++i)
      owlParamsSet2f(lp, std::string("transferFunction[" + std::to_string(i) + "].volumeDomain").c_str(),
          owl2f{tfdatas[i].volDomain.lower, tfdatas[i].volDomain.upper});

    owlParamsSet1i(lp, "mode", this->mode);

    Resetdt();

    owlParamsSet1i(lp, "volume.numChannels", meshType == MeshType::UMESH ? umeshPtrs.size() : rawPtrs.size());

    LOG("Building programs...");
    owlBuildPipeline(context);
    owlBuildSBT(context);
  }

  void Renderer::Render(short heatMapMode)
  {
    owlBuildSBT(context);

    auto start = std::chrono::system_clock::now();
    owlLaunch2D(rayGen, fbSize.x, fbSize.y, lp);
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    totalTime += elapsed_seconds.count();
    avgTime = totalTime / (accumID + 1);
    minTime = std::min(minTime, (float)elapsed_seconds.count());

    owlParamsSet1i(lp, "accumID", accumID++);
    owlParamsSet1i(lp, "frameID", frameID++);
    owlParamsSet1s(lp, "heatMapMode", heatMapMode);
    owlParamsSet1f(lp, "heatMapScale", heatMapScale);
  }

  void Renderer::Update()
  {
    auto glfw = GLFWHandler::getInstance();
#if !OFFLINE_VIEWER
    if (glfw->getWindowSize() != fbSize)
      Resize(glfw->getWindowSize());
#endif
    owlParamsSet1f(lp, "volume.dt", dt);
    owlParamsSet3f(lp, "lightDir", (const owl3f &)lightDir);
    owlParamsSet1b(lp, "enableShadows", enableShadows);
    owlParamsSet1b(lp, "enableAccumulation", enableAccumulation);
    owlParamsSet1b(lp, "enableGradientShading", enableGradientShading);
  }

  void Renderer::Terminate()
  {
    LOG("Terminating...\n");
    owlContextDestroy(context);
  }

  bool Renderer::PushMesh(std::shared_ptr<umesh::UMesh> mesh)
  {
    if(umeshPtrs.size() <= MAX_CHANNELS)
    {
      LOG("Pushing mesh...\n");
      umeshPtrs.push_back(mesh);
      return true;
    }
    else
    {
      LOG_ERROR("Cannot push mesh\n");
      return false;
    }
  }

  bool Renderer::PushMesh(std::shared_ptr<raw::RawR> mesh)
  {
    if(rawPtrs.size() <= MAX_CHANNELS)
    {
      LOG("Pushing mesh...\n");
      rawPtrs.push_back(mesh);
      return true;
    }
    else
    {
      LOG_ERROR("Cannot push mesh\n");
      return false;
    }
  }

  void Renderer::SetVaryingDims(const std::vector<float>& _varyingDims, int dimIndex)
  {
    if (_varyingDims.size() <= 0 || dimIndex < 0 || meshType == MeshType::UMESH)
      return;

    varyingDim = dimIndex;

    //resize varyingDims to the size of the raw data at dimIndex and set it to all 1.0f
    int datasizeOnDim = rawPtrs[0]->getDims()[dimIndex];
    varyingDims.resize(datasizeOnDim, 1.0f);
    //print a warning if the size of the _varyingDims is not equal to the size varyingDims
    if(_varyingDims.size() != varyingDims.size())
      printf("Warning: The size of the varying dimensions is not equal to the size of the raw data at the specified dimension index\n");

    //use memcpy to copy the data
    memcpy(varyingDims.data(), _varyingDims.data(), _varyingDims.size() * sizeof(float));

    if(rawPtrs.size() == 0)
      return;
    //adjust the global bounds to the new varying dimension it
    vec3f extendMultiplier(1.0f);
    float dimExtend = 0.0f;
      for(int j =0; j < rawPtrs[0]->getDims()[dimIndex]; j++)
        dimExtend += varyingDims[j];
    dimExtend /= rawPtrs[0]->getDims()[dimIndex];
    extendMultiplier[dimIndex] = dimExtend;
    rawPtrs[0]->reshapeBounds(extendMultiplier);
    //set the structured volume channel' new varying dimesions after creating an OWL buffer
    //prefix sum the varyingDims to get the varyingDimsPrefixSum
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
    if(meshType == MeshType::UMESH)
      camera.motionSpeed = umesh::length(umeshPtrs[0]->getBounds().size()) / 50.f;
    else if(meshType == MeshType::RAW)
      camera.motionSpeed = owl::length(rawPtrs[0]->getBounds().size()) / 50.f;


    // ----------- set variables  ----------------------------
    owlParamsSetGroup(lp, "triangleTLAS", triangleTLAS);
    owlParamsSet3f(lp, "camera.org", (const owl3f &)origin);
    owlParamsSet3f(lp, "camera.llc", (const owl3f &)lower_left_corner);
    owlParamsSet3f(lp, "camera.horiz", (const owl3f &)horizontal);
    owlParamsSet3f(lp, "camera.vert", (const owl3f &)vertical);
    ResetAccumulation();
  }

  void Renderer::SetXFColormap(std::vector<vec4f> newCM, size_t tfID)
  {
    if (tfID >= tfdatas.size())
      return; // safely return if this does not exist
    for (uint32_t i = 0; i < newCM.size(); ++i)
    {
      newCM[i].w = powf(newCM[i].w, 3.f);
    }

    tfdatas[tfID].colorMap = newCM;
    if (!tfdatas[tfID].colorMapBuffer)
      tfdatas[tfID].colorMapBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4,
                                             newCM.size(), nullptr);
    owlBufferUpload(tfdatas[tfID].colorMapBuffer, newCM.data());

    if (tfdatas[tfID].colorMapTexture != 0)
    {
      (cudaDestroyTextureObject(tfdatas[tfID].colorMapTexture));
      tfdatas[tfID].colorMapTexture = 0;
    }

    tfdatas[tfID].numTexels= tfdatas[tfID].colorMap.size();

    cudaResourceDesc res_desc = {};
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();

    // cudaArray_t   voxelArray;
    if (tfdatas[tfID].colorMapArray == 0)
    {
      (cudaMallocArray(&tfdatas[tfID].colorMapArray,
                       &channel_desc,
                       newCM.size(), 1));
    }

    int pitch = newCM.size() * sizeof(newCM[0]);
    (cudaMemcpy2DToArray(tfdatas[tfID].colorMapArray,
                         /* offset */ 0, 0,
                         newCM.data(),
                         pitch, pitch, 1,
                         cudaMemcpyHostToDevice));

    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = tfdatas[tfID].colorMapArray;

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
    (cudaCreateTextureObject(&tfdatas[tfID].colorMapTexture, &res_desc, &tex_desc,
                             nullptr));

    // OWLTexture xfTexture
    //   = owlTexture2DCreate(owl,OWL_TEXEL_FORMAT_RGBA32F,
    //                        colorMap.size(),1,
    //                        colorMap.data());
    owlParamsSetRaw(lp, 
      std::string("transferFunction[" + std::to_string(tfID) + "].xf").c_str(),
      &tfdatas[tfID].colorMapTexture);
    ResetAccumulation();
    if(mode < Mode::MARCHER_MULTI)
      RecalculateDensityRanges();
  }

  void Renderer::SetXFOpacityScale(float newOpacityScale, size_t tfID)
  {
    if (tfID >= tfdatas.size())
      return; // safely return if this does not exist
    tfdatas[tfID].opacityScale = newOpacityScale;
    owlParamsSet1f(lp, 
      std::string("transferFunction[" + std::to_string(tfID) + "].opacityScale").c_str(),
      tfdatas[tfID].opacityScale);
    ResetAccumulation();

    if(mode < Mode::MARCHER_MULTI)
      RecalculateDensityRanges();
  }


  void Renderer::SetXFRange(const vec2f newRange, size_t tfID)
  {
    if (tfID >= tfdatas.size())
      return; // safely return if this does not exist
    tfdatas[tfID].xfDomain = interval<float>(newRange.x, newRange.y);
    owlParamsSet2f(lp,
      std::string("transferFunction[" + std::to_string(tfID) + "].xfDomain").c_str(),
      (const owl2f &)tfdatas[tfID].xfDomain);
    ResetAccumulation();
    if(mode < Mode::MARCHER_MULTI)
      RecalculateDensityRanges();
  }

  vec3ui Renderer::CalculateMCGridDims(int estimatedElementPerMc)
  {
    if(meshType == MeshType::UMESH)
    {
      auto avgDims = vec3ui(0);
      for(auto mesh : umeshPtrs)
      {
        //get total number of elements
        size_t numElements = mesh->numVolumeElements();
        //lets assume elements are distributed uniformly in space
        //calculate number of elements per dimension
        //since dimensions are not necessarily cubic, we need to calculate the volume of the bounding box
        //and then calculate the number of elements per dimension
        auto size = mesh->getBounds().size();
        float volume = size.x * size.y * size.z;
        float NumElementsToVolumeRatio = (float)numElements / volume;
        float NumElementsPerDim = powf(NumElementsToVolumeRatio, 1.f / 3.f);
        vec3ui dims(size.x * NumElementsPerDim, size.y * NumElementsPerDim, size.z * NumElementsPerDim);
        avgDims += dims/(float)estimatedElementPerMc;
      }
      avgDims /= (float)umeshPtrs.size();
      printf("Calculated MC Grid Dims: %d %d %d\n", avgDims.x, avgDims.y, avgDims.z);
      return avgDims;

    }
    else if(meshType == MeshType::RAW)
    {
      auto avgDims = vec3ui(0);
      for(auto mesh : rawPtrs)
      {
        vec3ui dims = {static_cast<unsigned int>(mesh->getDims().x),
                      static_cast<unsigned int>(mesh->getDims().y),
                      static_cast<unsigned int>(mesh->getDims().z)};
        avgDims += dims/(float)estimatedElementPerMc;
      }
      avgDims /= (float)rawPtrs.size();
      printf("Calculated MC Grid Dims: %d %d %d\n", avgDims.x, avgDims.y, avgDims.z);
      return avgDims;
    }
    else
    {
      LOG_ERROR("Mesh type not supported\n");
      return vec3ui(0);
    }
  }

  void Renderer::Resetdt()
  {
    float minSpan = std::numeric_limits<float>::max();
    if (meshType == MeshType::UMESH)
    {
      //go over all elements calculate bounding boxes and find avg of spans
      for (int i = 0; i < umeshPtrs[0]->tets.size(); ++i)
      {
        auto tet = umeshPtrs[0]->tets[i];
        auto v0 = umeshPtrs[0]->vertices[tet[0]];
        auto v1 = umeshPtrs[0]->vertices[tet[1]];
        auto v2 = umeshPtrs[0]->vertices[tet[2]];
        auto v3 = umeshPtrs[0]->vertices[tet[3]];
        auto bb = box4f(vec4f(v0.x, v0.y, v0.z, umeshPtrs[0]->perVertex->values[tet[0]]),
                        vec4f(v1.x, v1.y, v1.z, umeshPtrs[0]->perVertex->values[tet[1]]));
        bb.extend(vec4f(v2.x, v2.y, v2.z, umeshPtrs[0]->perVertex->values[tet[2]]));
        bb.extend(vec4f(v3.x, v3.y, v3.z, umeshPtrs[0]->perVertex->values[tet[3]]));
        //calculate length of span
        minSpan = min(minSpan,max(length(vec3f(bb.span())), 0.05f));
      }
      //same for pyramids
      for (int i = 0; i < umeshPtrs[0]->pyrs.size(); ++i)
      {
        auto pyr = umeshPtrs[0]->pyrs[i];
        auto v0 = umeshPtrs[0]->vertices[pyr[0]];
        auto v1 = umeshPtrs[0]->vertices[pyr[1]];
        auto v2 = umeshPtrs[0]->vertices[pyr[2]];
        auto v3 = umeshPtrs[0]->vertices[pyr[3]];
        auto v4 = umeshPtrs[0]->vertices[pyr[4]];
        auto bb = box4f(vec4f(v0.x, v0.y, v0.z, umeshPtrs[0]->perVertex->values[pyr[0]]),
                        vec4f(v1.x, v1.y, v1.z, umeshPtrs[0]->perVertex->values[pyr[1]]));
        bb.extend(vec4f(v2.x, v2.y, v2.z, umeshPtrs[0]->perVertex->values[pyr[2]]));
        bb.extend(vec4f(v3.x, v3.y, v3.z, umeshPtrs[0]->perVertex->values[pyr[3]]));
        bb.extend(vec4f(v4.x, v4.y, v4.z, umeshPtrs[0]->perVertex->values[pyr[4]]));
        //calculate length of span
        minSpan = min(minSpan,max(length(vec3f(bb.span())), 0.05f));
      }
      //same for wedges
      for (int i = 0; i < umeshPtrs[0]->wedges.size(); ++i)
      {
        auto wedge = umeshPtrs[0]->wedges[i];
        auto v0 = umeshPtrs[0]->vertices[wedge[0]];
        auto v1 = umeshPtrs[0]->vertices[wedge[1]];
        auto v2 = umeshPtrs[0]->vertices[wedge[2]];
        auto v3 = umeshPtrs[0]->vertices[wedge[3]];
        auto v4 = umeshPtrs[0]->vertices[wedge[4]];
        auto v5 = umeshPtrs[0]->vertices[wedge[5]];
        auto bb = box4f(vec4f(v0.x, v0.y, v0.z, umeshPtrs[0]->perVertex->values[wedge[0]]),
                        vec4f(v1.x, v1.y, v1.z, umeshPtrs[0]->perVertex->values[wedge[1]]));
        bb.extend(vec4f(v2.x, v2.y, v2.z, umeshPtrs[0]->perVertex->values[wedge[2]]));
        bb.extend(vec4f(v3.x, v3.y, v3.z, umeshPtrs[0]->perVertex->values[wedge[3]]));
        bb.extend(vec4f(v4.x, v4.y, v4.z, umeshPtrs[0]->perVertex->values[wedge[4]]));
        bb.extend(vec4f(v5.x, v5.y, v5.z, umeshPtrs[0]->perVertex->values[wedge[5]]));
        //calculate length of span
        minSpan = min(minSpan,max(length(vec3f(bb.span())), 0.05f));
      }
      //same for hexes
      for (int i = 0; i < umeshPtrs[0]->hexes.size(); ++i)
      {
        auto hex = umeshPtrs[0]->hexes[i];
        auto v0 = umeshPtrs[0]->vertices[hex[0]];
        auto v1 = umeshPtrs[0]->vertices[hex[1]];
        auto v2 = umeshPtrs[0]->vertices[hex[2]];
        auto v3 = umeshPtrs[0]->vertices[hex[3]];
        auto v4 = umeshPtrs[0]->vertices[hex[4]];
        auto v5 = umeshPtrs[0]->vertices[hex[5]];
        auto v6 = umeshPtrs[0]->vertices[hex[6]];
        auto v7 = umeshPtrs[0]->vertices[hex[7]];
        auto bb = box4f(vec4f(v0.x, v0.y, v0.z, umeshPtrs[0]->perVertex->values[hex[0]]),
                        vec4f(v1.x, v1.y, v1.z, umeshPtrs[0]->perVertex->values[hex[1]]));
        bb.extend(vec4f(v2.x, v2.y, v2.z, umeshPtrs[0]->perVertex->values[hex[2]]));
        bb.extend(vec4f(v3.x, v3.y, v3.z, umeshPtrs[0]->perVertex->values[hex[3]]));
        bb.extend(vec4f(v4.x, v4.y, v4.z, umeshPtrs[0]->perVertex->values[hex[4]]));
        bb.extend(vec4f(v5.x, v5.y, v5.z, umeshPtrs[0]->perVertex->values[hex[5]]));
        bb.extend(vec4f(v6.x, v6.y, v6.z, umeshPtrs[0]->perVertex->values[hex[6]]));
        bb.extend(vec4f(v7.x, v7.y, v7.z, umeshPtrs[0]->perVertex->values[hex[7]]));
        //calculate length of span
        minSpan = min(minSpan,max(length(vec3f(bb.span())), 0.05f));
      }
      Setdt(minSpan * 0.5f);
    }
    else if (meshType == MeshType::RAW)
    {
      float minVoxelSideLength = std::numeric_limits<float>::max();
      for (int i = 0; i < rawPtrs.size(); ++i)
      {
        const auto& span = rawPtrs[i]->getBounds().span();
        const auto& dims = rawPtrs[i]->getDims();
        minVoxelSideLength = min(span.x/(float)dims.x, min(span.y/(float)dims.y, span.z/(float)dims.z));
      }
      Setdt(minVoxelSideLength * 0.5f);
    }
    else
      exit(1);
    
  }

  void Renderer::Setdt(float newDt)
  {
    dt = max(newDt, 1e-4f);
    printf("Set dt to : %f\n", dt);
    owlParamsSet1f(lp, "volume.dt", dt);
    ResetAccumulation();
  }

  void Renderer::SetLightDirection(const vec3f newLightDir)
  {
    //if any of the components is 0, set it to a small value
    lightDir = newLightDir;
    owlParamsSet3f(lp, "lightDir", (const owl3f &)lightDir);
    ResetAccumulation();
  }

  void Renderer::SetLightIntensity(float lightIntensity)
  {
    owlParamsSet1f(lp, "lightIntensity", lightIntensity);
    ResetAccumulation();
  }

  void Renderer::SetAmbient(float ambientIntensity)
  {
    ambient = ambientIntensity;
    owlParamsSet1f(lp, "ambientIntensity", ambient);
    ResetAccumulation();
  }
  void Renderer::ResetAccumulation()
  {
    accumID = 0;
    owlParamsSet1i(lp, "accumID", accumID);
    totalTime = 0;
    minTime = std::numeric_limits<float>::max();
    avgTime = 0;
  }

} // namespace dtracker
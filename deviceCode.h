#pragma once

#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include "owl/common/math/random.h"
#include <cuda_runtime.h>

#include "cuda_fp16.h"

using namespace owl;

#define NUM_BINS 32

/* variables for the triangle mesh geometry */
struct TriangleData
{
  /*! base color we use for the entire mesh */
  vec3f color;
  /*! array/buffer of vertex indices */
  vec3i *indices;
  /*! array/buffer of vertex positions */
  vec3f *vertices;
};

struct UnstructuredElementData
{
  unsigned int *tetrahedra;
  unsigned int *pyramids;
  unsigned int *hexahedra;
  unsigned int *wedges;
  uint32_t bytesPerIndex;
  vec3f *vertices;
  float *scalars;
  uint64_t offset; // for pre-split geom
  uint64_t numTetrahedra;
  uint64_t numPyramids;
  uint64_t numWedges;
  uint64_t numHexahedra;
  uint8_t *maxima;
  half *bboxes;
};

struct MacrocellData {
    float* maxima;        
    box4f* bboxes;
    int offset; // for pre-split geom
};


/* variables for the ray generation program */
struct LaunchParams
{
  uint32_t *fbPtr;
  vec2i fbSize;
  vec4f *accumBuffer;
  uint32_t frameID;
  uint32_t accumID;

  OptixTraversableHandle triangleTLAS;
  bool shadows;
  vec3f lightDir;

  struct
  {
    //interval<float> domain;
    OptixTraversableHandle rootMacrocellTLAS;
    OptixTraversableHandle elementTLAS;
    OptixTraversableHandle macrocellTLAS;

    int numModes;
    int mode;
    int numAdaptiveSamplingRays;
    float dt;

    vec3i macrocellDims;

    float4 globalBoundsLo;
    float4 globalBoundsHi;

  } volume;

  struct
  {
    cudaTextureObject_t xf;
    // int numTexels;
    float2 volumeDomain;
    float2 xfDomain;
    float opacityScale;
  } transferFunction;

  struct
  {
    vec3f origin;
    vec3f lower_left_corner;
    vec3f horizontal;
    vec3f vertical;
  } camera;
};

struct RayGenData
{};

struct RayPayload
{
  float t0;
  float t1;
  owl::common::LCG<4> rng; // random number generator
  vec4f rgba;
  float dataMax;
  float dataMin;
  float dataAvg;
  float dataValue;
  float tHit;
  int samples;
  float maxima[NUM_BINS];
  bool shadowRay;
  bool missed;
  bool debug;
};

/* variables for the miss program */
struct MissProgData
{
  vec3f color0;
  vec3f color1;
};
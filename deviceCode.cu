#include "deviceCode.h"
#include <optix_device.h>
#include "renderer.h"
#include "unstructuredElementHelper.h"

using namespace owl;
using namespace dtracker;

extern "C" __constant__ static LaunchParams optixLaunchParams;

#define DEBUG 0
// create a debug function macro that gets called only for center pixel
inline __device__ bool dbg()
{
    auto lp = optixLaunchParams;
#if DEBUG
    return false;
#else
    auto pixelID = vec2i(owl::getLaunchIndex()[0], owl::getLaunchIndex()[1]);
    return (lp.fbSize.x / 2 == pixelID.x) &&
           (lp.fbSize.y / 2 == pixelID.y);
#define ACTIVATE_CROSSHAIRS
#endif
}

inline __both__ 
float4 transferFunction(float f)
{
  auto &lp = optixLaunchParams;
  if (f < lp.transferFunction.volumeDomain.x ||
      f > lp.transferFunction.volumeDomain.y)
  {
    return make_float4(1.f, 0.f, 1.f, 0.0f);
  }
  float remapped = (f - lp.transferFunction.volumeDomain.x) /
                   (lp.transferFunction.volumeDomain.y - lp.transferFunction.volumeDomain.x);

  float4 xf = tex2D<float4>(lp.transferFunction.xf, remapped, 0.5f);
  xf.w *= lp.transferFunction.opacityScale;

  return xf;
}

inline __device__ vec3f over(vec3f Cin, vec3f Cx, float Ain, float Ax)
{
  return Cin + Cx * Ax * (1.f - Ain);
}

inline __device__ float over(const float Ain, const float Ax)
{
  return Ain + (1.f - Ain) * Ax;
}

inline __device__ vec4f over(const vec4f &in, const vec4f &x)
{
  auto c = over(vec3f(in), vec3f(x), in.w, x.w);
  auto a = over(in.w, x.w);
  return vec4f(c, a);
}

inline __device__ void generateRay(const vec2f screen, owl::Ray &ray)
{
    auto &lp = optixLaunchParams;
    ray.origin = lp.camera.origin;
    vec3f direction = lp.camera.lower_left_corner +
                      screen.u * lp.camera.horizontal +
                      screen.v * lp.camera.vertical;
    // direction = normalize(direction);
    if (fabs(direction.x) < 1e-5f)
        direction.x = 1e-5f;
    if (fabs(direction.y) < 1e-5f)
        direction.y = 1e-5f;
    if (fabs(direction.z) < 1e-5f)
        direction.z = 1e-5f;
    ray.direction = normalize(direction - ray.origin);
}

inline __device__
vec3f missCheckerBoard(const vec3f& color0 = vec3f(.2f, .2f, .26f), 
    const vec3f& color1 = vec3f(.1f, .1f, .16f), int gap = 25)
{
    const vec2i pixelID = owl::getLaunchIndex();

    int pattern = (pixelID.x / gap) ^ (pixelID.y / gap);
    vec3f color = (pattern & 1) ? color1 : color0;
    return color;
}

// Simple raygen that creates a checker-board pattern
OPTIX_RAYGEN_PROGRAM(testRayGen)
()
{
    auto &lp = optixLaunchParams;
    const vec2i pixelID = owl::getLaunchIndex();
    const int fbOfs = pixelID.x + lp.fbSize.x * pixelID.y;

    int seed = owl::getLaunchDims().x * owl::getLaunchDims().y * lp.frameID;
    owl::common::LCG<4> random(threadIdx.x + seed, threadIdx.y + seed);
    const vec2f screen = (vec2f(pixelID) + vec2f(.5f)) / vec2f(lp.fbSize);
    Ray ray;
    generateRay(screen, ray);

    RayPayload surfPrd;
    const MissProgData &missData = owl::getProgramData<MissProgData>();
    vec4f finalColor = vec4f(missCheckerBoard(), 1.0f);
    vec4f color = vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    traceRay(lp.triangleTLAS, ray, surfPrd); //surface
    if (!surfPrd.missed)
         color = surfPrd.rgba;

    const float tMax = surfPrd.missed ? 1000.f : surfPrd.tHit;
    int numSteps = 0;
    for(float t = 0.f; t < tMax || numSteps < 10; t+=5.f )
    {
        RayPayload volPrd;
        traceRay(lp.volume.elementTLAS, ray, volPrd); //volume
        if(!volPrd.missed)
        {
            color = over(transferFunction(volPrd.dataValue), color);
            //do opacity correction using exp and step size
            color.w *= 1.f - exp(-color.w * 5.f);
            color.x *= color.w;
            color.y *= color.w;
            color.z *= color.w;
            //clamp color
            color =clamp(color, vec4f(0.f), vec4f(1.f));
        }
        if(color.w > 0.99f)
            break;
        ray.origin += ray.direction * 5.f;
        numSteps++;
    }

    // if(!volPrd.missed)
    //     finalColor = 0.3f * vec3f(1.0f,0.0f,0.0f) + 0.7f * finalColor;
    finalColor = over(color, finalColor);
    
    vec4f oldColor = lp.accumBuffer[fbOfs];
    vec4f newColor = (vec4f(finalColor) + float(lp.accumID) * oldColor) / float(lp.accumID + 1);
    lp.fbPtr[fbOfs] = make_rgba(vec4f(newColor));
    lp.accumBuffer[fbOfs] = vec4f(newColor);
}

OPTIX_CLOSEST_HIT_PROGRAM(triangle_test)
()
{
    // get light direction and do a simple lambert shading

    const TriangleData &self = owl::getProgramData<TriangleData>();

    // compute normal:
    const int primID = optixGetPrimitiveIndex();
    const vec3i index = self.indices[primID];
    const vec3f &A = self.vertices[index.x];
    const vec3f &B = self.vertices[index.y];
    const vec3f &C = self.vertices[index.z];
    const vec3f Ng = normalize(cross(B - A, C - A));
    const vec2f bary = optixGetTriangleBarycentrics();
    const vec3f P = bary.x * A + bary.y * B + (1.f - bary.x - bary.y) * C;

    const vec3f rayDir = optixGetWorldRayDirection();
    RayPayload &prd = owl::getPRD<RayPayload>();
    prd.tHit = length(P - vec3f(optixGetWorldRayOrigin()));
    prd.missed = false;
    prd.rgba = vec4f((.2f + .8f * fabs(dot(rayDir, Ng))) * self.color, 1);
}

OPTIX_MISS_PROGRAM(miss)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    prd.missed = true;
}

// ------------------------------------------------------------------
// Bounds programs for volume elements
// ------------------------------------------------------------------

// OPTIX_BOUNDS_PROGRAM(MacrocellBounds)
// (
//     const void *geomData,
//     owl::common::box3f &primBounds,
//     const int primID)
// {
//     const MacrocellData &self = *(const MacrocellData *)geomData;
//     // if (self.maxima[primID] <= 0.f) {
//     //    primBounds = box3f(); // empty box
//     //  }
//     //  else
//     {
//         primBounds = box3f();
//         primBounds = primBounds.including(vec3f(self.bboxes[(primID * 2 + 0)].x,
//                                                 self.bboxes[(primID * 2 + 0)].y,
//                                                 self.bboxes[(primID * 2 + 0)].z));
//         primBounds = primBounds.including(vec3f(self.bboxes[(primID * 2 + 1)].x,
//                                                 self.bboxes[(primID * 2 + 1)].y,
//                                                 self.bboxes[(primID * 2 + 1)].z));
//         // primBounds.lower.x = self.bboxes[(primID * 2 + 0)].x;
//         // primBounds.lower.y = self.bboxes[(primID * 2 + 0)].y;
//         // primBounds.lower.z = self.bboxes[(primID * 2 + 0)].z;
//         // primBounds.upper.x = self.bboxes[(primID * 2 + 1)].x;
//         // primBounds.upper.y = self.bboxes[(primID * 2 + 1)].y;
//         // primBounds.upper.z = self.bboxes[(primID * 2 + 1)].z;
//     }
// }

OPTIX_BOUNDS_PROGRAM(TetrahedraBounds)
(
    const void *geomData,
    owl::common::box3f &primBounds,
    const int primID)
{
    const UnstructuredElementData &self = *(const UnstructuredElementData *)geomData;
    primBounds = box3f();
    unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;
    if (ID >= self.numTetrahedra)
        return;

    unsigned int *tets = (unsigned int *)self.tetrahedra;
    uint64_t i0 = tets[ID * 4 + 0];
    uint64_t i1 = tets[ID * 4 + 1];
    uint64_t i2 = tets[ID * 4 + 2];
    uint64_t i3 = tets[ID * 4 + 3];

    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];

    primBounds = primBounds.including(P0)
                     .including(P1)
                     .including(P2)
                     .including(P3);
}

OPTIX_BOUNDS_PROGRAM(PyramidBounds)
(
    const void *geomData,
    owl::common::box3f &primBounds,
    const int primID)

{
    const UnstructuredElementData &self = *(const UnstructuredElementData *)geomData;
    primBounds = box3f();
    unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;
    if (ID >= self.numPyramids)
        return;

    unsigned int *pyrs = (unsigned int *)self.pyramids;
    uint64_t i0 = pyrs[ID * 5 + 0];
    uint64_t i1 = pyrs[ID * 5 + 1];
    uint64_t i2 = pyrs[ID * 5 + 2];
    uint64_t i3 = pyrs[ID * 5 + 3];
    uint64_t i4 = pyrs[ID * 5 + 4];

    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];
    vec3f P4 = self.vertices[i4];

    primBounds = primBounds.including(P0)
                     .including(P1)
                     .including(P2)
                     .including(P3)
                     .including(P4);
}

OPTIX_BOUNDS_PROGRAM(WedgeBounds)
(
    const void *geomData,
    owl::common::box3f &primBounds,
    const int primID)
{
    const UnstructuredElementData &self = *(const UnstructuredElementData *)geomData;
    primBounds = box3f();
    unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;
    if (ID >= self.numWedges)
        return;

    unsigned int *weds = (unsigned int *)self.wedges;
    uint64_t i0 = weds[ID * 6 + 0];
    uint64_t i1 = weds[ID * 6 + 1];
    uint64_t i2 = weds[ID * 6 + 2];
    uint64_t i3 = weds[ID * 6 + 3];
    uint64_t i4 = weds[ID * 6 + 4];
    uint64_t i5 = weds[ID * 6 + 5];

    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];
    vec3f P4 = self.vertices[i4];
    vec3f P5 = self.vertices[i5];

    primBounds = primBounds.including(P0)
                     .including(P1)
                     .including(P2)
                     .including(P3)
                     .including(P4)
                     .including(P5);
}

OPTIX_BOUNDS_PROGRAM(HexahedraBounds)
(
    const void *geomData,
    owl::common::box3f &primBounds,
    const int primID)
{
    const UnstructuredElementData &self = *(const UnstructuredElementData *)geomData;
    primBounds = box3f();
    unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;
    if (ID >= self.numHexahedra)
        return;

    unsigned int *hexes = (unsigned int *)self.hexahedra;
    uint64_t i0 = hexes[ID * 8 + 0];
    uint64_t i1 = hexes[ID * 8 + 1];
    uint64_t i2 = hexes[ID * 8 + 2];
    uint64_t i3 = hexes[ID * 8 + 3];
    uint64_t i4 = hexes[ID * 8 + 4];
    uint64_t i5 = hexes[ID * 8 + 5];
    uint64_t i6 = hexes[ID * 8 + 6];
    uint64_t i7 = hexes[ID * 8 + 7];

    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];
    vec3f P4 = self.vertices[i4];
    vec3f P5 = self.vertices[i5];
    vec3f P6 = self.vertices[i6];
    vec3f P7 = self.vertices[i7];
    primBounds.extend(P0)
        .extend(P1)
        .extend(P2)
        .extend(P3)
        .extend(P4)
        .extend(P5)
        .extend(P6)
        .extend(P7);
    // primBounds.extend(P7); // wtf??!
}

// ------------------------------------------------------------------
// intersection programs
// ------------------------------------------------------------------
OPTIX_INTERSECT_PROGRAM(TetrahedraPointQuery)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    const auto &self = owl::getProgramData<UnstructuredElementData>();
    unsigned int primID = optixGetPrimitiveIndex(); //+ self.offset;
    float3 origin = optixGetObjectRayOrigin();

    // for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
    //   uint32_t ID = primID * ELEMENTS_PER_BOX + i;
    if (primID >= self.numTetrahedra)
        return;

    // printf("TetrahedraPointQuery: primID = %d\\n", primID);

    unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;

    vec3f P = {origin.x, origin.y, origin.z};

    // unsigned int i0, i1, i2, i3;
    uint32_t *tets = (uint32_t *)self.tetrahedra;
    uint64_t i0 = tets[ID * 4 + 0];
    uint64_t i1 = tets[ID * 4 + 1];
    uint64_t i2 = tets[ID * 4 + 2];
    uint64_t i3 = tets[ID * 4 + 3];

    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];

    float S0 = self.scalars[i0];
    float S1 = self.scalars[i1];
    float S2 = self.scalars[i2];
    float S3 = self.scalars[i3];

    // prd.missed = false;              // for
    // prd.dataValue = S0;              // testing
    // optixReportIntersection(0.f, 0); // please
    // return;                          // remove

    if (interpolateTetrahedra(P, P0, P1, P2, P3, S0, S1, S2, S3, prd.dataValue))
    {
        optixReportIntersection(0.f, 0);
        prd.missed = false;
        return;
    }
}

OPTIX_INTERSECT_PROGRAM(PyramidPointQuery)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    const auto &self = owl::getProgramData<UnstructuredElementData>();
    unsigned int primID = optixGetPrimitiveIndex(); //+ self.offset;
    float3 origin = optixGetObjectRayOrigin();

    // for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
    //   uint32_t ID = primID * ELEMENTS_PER_BOX + i;
    if (primID >= self.numPyramids)
        return;

    // printf("TetrahedraPointQuery: primID = %d\\n", primID);

    unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;

    vec3f P = {origin.x, origin.y, origin.z};

    // unsigned int i0, i1, i2, i3;
    uint32_t *pyrs = (uint32_t *)self.pyramids;
    uint64_t i0 = pyrs[ID * 5 + 0];
    uint64_t i1 = pyrs[ID * 5 + 1];
    uint64_t i2 = pyrs[ID * 5 + 2];
    uint64_t i3 = pyrs[ID * 5 + 3];
    uint64_t i4 = pyrs[ID * 5 + 4];

    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];
    vec3f P4 = self.vertices[i4];

    float S0 = self.scalars[i0];
    float S1 = self.scalars[i1];
    float S2 = self.scalars[i2];
    float S3 = self.scalars[i3];
    float S4 = self.scalars[i4];

    // prd.missed = false;              // for
    // prd.dataValue = S0;              // testing
    // optixReportIntersection(0.f, 0); // please
    // return;                          // remove

    if (interpolatePyramid(P, P0, P1, P2, P3, P4, S0, S1, S2, S3, S4, prd.dataValue))
    {
        optixReportIntersection(0.f, 0);
        prd.missed = false;
        return;
    }
}

OPTIX_INTERSECT_PROGRAM(WedgePointQuery)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    const auto &self = owl::getProgramData<UnstructuredElementData>();
    unsigned int primID = optixGetPrimitiveIndex(); //+ self.offset;
    float3 origin = optixGetObjectRayOrigin();

    // for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
    //   uint32_t ID = primID * ELEMENTS_PER_BOX + i;
    if (primID >= self.numWedges)
        return;

    // printf("TetrahedraPointQuery: primID = %d\\n", primID);

    unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;

    vec3f P = {origin.x, origin.y, origin.z};

    // unsigned int i0, i1, i2, i3;
    uint32_t *weds = (uint32_t *)self.wedges;
    uint64_t i0 = weds[ID * 6 + 0];
    uint64_t i1 = weds[ID * 6 + 1];
    uint64_t i2 = weds[ID * 6 + 2];
    uint64_t i3 = weds[ID * 6 + 3];
    uint64_t i4 = weds[ID * 6 + 4];
    uint64_t i5 = weds[ID * 6 + 5];

    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];
    vec3f P4 = self.vertices[i4];
    vec3f P5 = self.vertices[i5];

    float S0 = self.scalars[i0];
    float S1 = self.scalars[i1];
    float S2 = self.scalars[i2];
    float S3 = self.scalars[i3];
    float S4 = self.scalars[i4];
    float S5 = self.scalars[i5];

    // prd.missed = false;              // for
    // prd.dataValue = S0;              // testing
    // optixReportIntersection(0.f, 0); // please
    // return;                          // remove

    if (interpolateWedge(P, P0, P1, P2, P3, P4, P5, S0, S1, S2, S3, S4, S5, prd.dataValue))
    {
        optixReportIntersection(0.f, 0);
        prd.missed = false;
        return;
    }
}

OPTIX_INTERSECT_PROGRAM(HexahedraPointQuery)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    const auto &self = owl::getProgramData<UnstructuredElementData>();
    unsigned int primID = optixGetPrimitiveIndex(); //+ self.offset;
    float3 origin = optixGetObjectRayOrigin();

    // for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
    //   uint32_t ID = primID * ELEMENTS_PER_BOX + i;
    if (primID >= self.numHexahedra)
        return;

    // printf("TetrahedraPointQuery: primID = %d\\n", primID);

    unsigned int ID = (uint32_t(primID)) /* ELEMENTS_PER_BOX*/;

    vec3f P = {origin.x, origin.y, origin.z};

    // unsigned int i0, i1, i2, i3;
    uint32_t *hexes = (uint32_t *)self.hexahedra;
    uint64_t i0 = hexes[ID * 8 + 0];
    uint64_t i1 = hexes[ID * 8 + 1];
    uint64_t i2 = hexes[ID * 8 + 2];
    uint64_t i3 = hexes[ID * 8 + 3];
    uint64_t i4 = hexes[ID * 8 + 4];
    uint64_t i5 = hexes[ID * 8 + 5];
    uint64_t i6 = hexes[ID * 8 + 6];
    uint64_t i7 = hexes[ID * 8 + 7];

    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];
    vec3f P4 = self.vertices[i4];
    vec3f P5 = self.vertices[i5];
    vec3f P6 = self.vertices[i6];
    vec3f P7 = self.vertices[i7];

    float S0 = self.scalars[i0];
    float S1 = self.scalars[i1];
    float S2 = self.scalars[i2];
    float S3 = self.scalars[i3];
    float S4 = self.scalars[i4];
    float S5 = self.scalars[i5];
    float S6 = self.scalars[i6];
    float S7 = self.scalars[i7];

    // prd.missed = false;              // for
    // prd.dataValue = S0;              // testing
    // optixReportIntersection(0.f, 0); // please
    // return;                          // remove

    if (interpolateHexahedra(P, P0, P1, P2, P3, P4, P5, P6, P7,
                             S0, S1, S2, S3, S4, S5, S6, S7, prd.dataValue))
    {
        optixReportIntersection(0.f, 0);
        prd.missed = false;
        return;
    }
}
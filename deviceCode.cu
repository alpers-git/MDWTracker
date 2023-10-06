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
    vec3f
    missColor(const Ray &ray)
{
    const vec2i pixelID = owl::getLaunchIndex();

    // Calculate the intersection point in world coordinates
    vec3f intersectionPoint = ray.origin + ray.direction * 1e20f;

    // Calculate the grid pattern based on the intersection point
    int pattern = ((int)intersectionPoint.x / 18) ^ ((int)intersectionPoint.y / 18);

    vec3f color = (pattern & 1) ? vec3f(.2f, .2f, .26f) : vec3f(.1f, .1f, .16f);
    return color;
}

// Simple raygen that creates a checker-board pattern
OPTIX_RAYGEN_PROGRAM(testRayGen)
()
{
    auto &lp = optixLaunchParams;
    const vec2i pixelID = owl::getLaunchIndex();
    const int fbOfs = pixelID.x + lp.fbSize.x * pixelID.y;

    const vec2f screen = (vec2f(pixelID) + vec2f(0.5f)) / vec2f(lp.fbSize);
    Ray ray;
    generateRay(screen, ray);

    RayPayload prd;

    traceRay(/*accel to trace against*/ lp.triangleTLAS,
             /*the ray to trace*/ ray,
             /*prd*/ prd);

    // prd.rgba = vec4f(missColor(ray), 1);
    // Choose the appropriate color based on the checkerboard pattern
    lp.fbPtr[fbOfs] = owl::make_rgba(prd.rgba);
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

    const vec3f rayDir = optixGetWorldRayDirection();
    RayPayload &prd = owl::getPRD<RayPayload>();
    prd.rgba = vec4f((.2f + .8f * fabs(dot(rayDir, Ng))) * self.color, 1);
}

OPTIX_MISS_PROGRAM(miss)
()
{
    const vec2i pixelID = owl::getLaunchIndex();

    const MissProgData &self = owl::getProgramData<MissProgData>();

    RayPayload &prd = owl::getPRD<RayPayload>();
    int pattern = (pixelID.x / 18) ^ (pixelID.y / 18);
    prd.rgba = (pattern & 1) ? vec4f(self.color1, 1) : vec4f(self.color0, 1);
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
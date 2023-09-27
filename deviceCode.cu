#include "deviceCode.h"
#include <optix_device.h>
#include "renderer.h"

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

    traceRay(/*accel to trace against*/ lp.volume.elementTLAS,
             /*the ray to trace*/ ray,
             /*prd*/ prd);

    // Choose the appropriate color based on the checkerboard pattern
    lp.fbPtr[fbOfs] = owl::make_rgba(prd.rgba);
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
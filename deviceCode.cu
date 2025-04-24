#include "deviceCode.h"
#include "dda.h"
#include <optix_device.h>
#include "renderer.h"
#include "unstructuredElementHelper.h"

using namespace owl;
using namespace dtracker;

extern "C" __constant__ LaunchParams optixLaunchParams;

#define DEBUG 1
// create a debug function macro that gets called only for center pixel
inline __device__ bool dbg()
{
#if DEBUG
    return false;
#else
    auto lp = optixLaunchParams;
    auto pixelID = vec2i(owl::getLaunchIndex()[0], owl::getLaunchIndex()[1]);
    return (lp.fbSize.x / 2 == pixelID.x) &&
           (lp.fbSize.y / 2 == pixelID.y);
#define ACTIVATE_CROSSHAIRS
#endif
}

inline __both__ float4 transferFunction(float f, size_t tfID = 0)
{
    auto &lp = optixLaunchParams;
    if (f < lp.transferFunction[tfID].volumeDomain.x ||
        f > lp.transferFunction[tfID].volumeDomain.y)
    {
        return make_float4(1.f, 0.f, 1.f, 0.0f);
    }
    float remapped1 = (f - lp.transferFunction[tfID].volumeDomain.x) / (lp.transferFunction[tfID].volumeDomain.y - lp.transferFunction[tfID].volumeDomain.x);
    float remapped2 = (remapped1 - lp.transferFunction[tfID].xfDomain.x) / (lp.transferFunction[tfID].xfDomain.y - lp.transferFunction[tfID].xfDomain.x);
    
    float4 xf = tex2D<float4>(lp.transferFunction[tfID].xf, remapped2, 0.5f);
    xf.w *= lp.transferFunction[tfID].opacityScale;

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

inline __device__
float sampleVolumeTexture(const vec3f& normalizedPos, const int channelID = 0)
{
    if (normalizedPos.x < 0.0f || normalizedPos.x > 1.0f || normalizedPos.y < 0.0f || normalizedPos.y > 1.0f || normalizedPos.z < 0.0f || normalizedPos.z > 1.0f)
        return NAN;
    auto &lp = optixLaunchParams;
    float value = tex3D<float>(lp.volume.sGrid[channelID].scalarTex, 
            normalizedPos.x, normalizedPos.y,normalizedPos.z);
                    
    // Sample scalar field
    return value;
}

inline __device__
float sampleVolume(const vec3f& pos, const int channelID = 0)
{
    auto &lp = optixLaunchParams;
    if(lp.volume.meshType == 1)//Query unstructred mesh
    {
        //create a ray with zero lenght and origin at pos
        Ray ray;
        ray.origin = pos;
        ray.direction = vec3f(1.0f, 1.0f, 1.0f);
        ray.tmin = 0.0f;
        ray.tmax = 0.0f;
        ray.time = 0.0f;
        RayPayload prd;
        prd.debug = dbg();

        owl::traceRay(lp.volume.elementTLAS, ray, prd);
        if (prd.missed)
            return NAN;
        else
            return prd.dataValue;
    }
    else if(lp.volume.meshType == 2)//Query structured mesh
    {
        //normalize pos to [0,1] using bounds of voxel grid
        vec3f normalizedPos = (pos - vec3f(lp.volume.globalBoundsLo)) / 
            (vec3f(lp.volume.globalBoundsHi) - vec3f(lp.volume.globalBoundsLo)); 
        // Sample scalar field
        return sampleVolumeTexture(normalizedPos, channelID);
    }
}
//max opacity blending func
__forceinline__ __device__
static vec4f maxOpacityBlend(vec3f texSpacePos)
{
    //find the mesh with the highest opacity
    vec4f maxOpacitySample = vec4f(0.0f,0.0f,0.0f,0.0f);
    for(int channelID = 0; channelID < optixLaunchParams.volume.numChannels; channelID++)
    {
        vec4f sample = transferFunction(
            sampleVolumeTexture(texSpacePos, channelID), channelID);
        if(sample.w > maxOpacitySample.w)
            maxOpacitySample = sample;
    }
    return maxOpacitySample;
}

//mix with equal weights blending func
__forceinline__ __device__
    static vec4f mixBlend(vec3f texSpacePos)
{
    vec3f color = vec3f(0.0f,0.0f,0.0f);
    float maxOpacity = 0.0f;
    for(int channelID = 0; channelID < optixLaunchParams.volume.numChannels; channelID++)
    {
        vec4f curColor = transferFunction(
            sampleVolumeTexture(texSpacePos, channelID), channelID);
        color += vec3f(curColor) * curColor.w;
        if(curColor.w > maxOpacity)
            maxOpacity = curColor.w;
    }
    if(maxOpacity <= 0.0f)
        return vec4f(0.0f,0.0f,0.0f,0.0f);
    color /= maxOpacity;
    return vec4f(color, maxOpacity);
}

//majorant based blending func
__forceinline__ __device__
static vec4f densityBlend(vec3f texSpacePos)
{
    LaunchParams &lp = optixLaunchParams;
    float densitySum = 0.0f;
    //weighted average of samples from each mesh using majorants as weights
    vec4f color = vec4f(0.0f,0.0f,0.0f,0.0f);
    for (int i = 0; i < lp.volume.numChannels; i++)
    {
        vec4f sample = transferFunction(
            sampleVolumeTexture(texSpacePos, i), i);
        color += sample * sample.w;
        densitySum += sample.w;
    }
    if(densitySum <= 0.0f)
        return vec4f(0.0f,0.0f,0.0f,0.0f);
    color /= densitySum;
    return color;
}
__device__ 
vec4f blendChannels(const vec3f texSpacePos)
{
    if (texSpacePos.x < 0.0f || texSpacePos.x > 1.0f ||
     texSpacePos.y < 0.0f || texSpacePos.y > 1.0f ||
     texSpacePos.z < 0.0f || texSpacePos.z > 1.0f)
        return vec4f(0.0f,0.0f,0.0f,0.0f);// empty sample for boundary condition
    switch (optixLaunchParams.mode)
    {
    case Mode::MARCHER_MULTI: // majorant based blending
        return densityBlend(texSpacePos);
    case Mode::MAX:
    case Mode::MARCHER_MAX: // max opacity blending
        return maxOpacityBlend(texSpacePos);
    case Mode::MIX:
    case Mode::MARCHER_MIX: // mix blending with equal weights
        return mixBlend(texSpacePos);
    default:
        break;
    }
}


OPTIX_RAYGEN_PROGRAM(mainRG)
()
{
    auto &lp = optixLaunchParams;
    const vec2i pixelID = owl::getLaunchIndex();
    const int fbOfs = pixelID.x + lp.fbSize.x * pixelID.y;

    RayPayload volumePrd;
    uint64_t start, end;
    vec4f finalColor = vec4f(0.f,0.f,0.f,1.f);
    for (int sampleID = 0; sampleID < lp.spp; sampleID++)
    {
        // generate ray
        int seed = owl::getLaunchDims().x * owl::getLaunchDims().y * (lp.frameID+1) * (sampleID+1);
        owl::common::LCG<4> random(threadIdx.x + seed, threadIdx.y + seed); // jittered sampling
        const vec2f screen = (vec2f(pixelID) + random()) / vec2f(lp.fbSize);
        Ray ray;
        generateRay(screen, ray);

        // test surface intersections first
        RayPayload surfPrd;
         // vec4f(missCheckerBoard(), 1.0f);
        vec4f color = vec4f(0.0f, 0.0f, 0.0f, 0.0f);
        traceRay(lp.triangleTLAS, ray, surfPrd, OPTIX_RAY_FLAG_DISABLE_ANYHIT); // surface
        if (!surfPrd.missed)
            color = surfPrd.rgba;

        const float tMax = surfPrd.missed ? 1e20 : surfPrd.tHit;

        // test for root macrocell intersection
        //===== TIMING =====
        if (lp.heatMapMode == 3)
            start = clock();
        //==================
        volumePrd.debug = dbg();
        volumePrd.t0 = 0.f;
        volumePrd.t1 = tMax;
        volumePrd.rng = random;
        traceRay(lp.volume.rootMacrocellTLAS, ray, volumePrd, OPTIX_RAY_FLAG_DISABLE_ANYHIT); // root macrocell to initiate dda traversal
        if (!volumePrd.missed)
        {
            vec3f albedo = vec3f(volumePrd.rgba);
            float transparency = volumePrd.rgba.w;
            if (lp.enableShadows && (lp.mode < Mode::MARCHER_MULTI))
            {
                // trace shadow rays
                RayPayload shadowbyVolPrd;
                shadowbyVolPrd.debug = dbg();
                shadowbyVolPrd.t0 = 0.f;
                shadowbyVolPrd.t1 = 1e20f; // todo fix this
                int seed = owl::getLaunchDims().x * owl::getLaunchDims().y * lp.frameID;
                shadowbyVolPrd.rng = owl::common::LCG<4>(threadIdx.x + seed + 1, threadIdx.y + seed + 1);

                Ray shadowRay;
                shadowRay.origin = ray.origin + volumePrd.tHit * ray.direction;
                shadowRay.direction = -lp.lightDir;
                shadowRay.tmin = 0.00f;
                shadowRay.tmax = 1e20f;

                traceRay(lp.volume.rootMacrocellTLAS, shadowRay, shadowbyVolPrd, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
                vec3f shadow((1.f - lp.ambient) * (1.f - shadowbyVolPrd.rgba.w) + lp.ambient);
                color = vec4f(albedo * shadow * lp.lightIntensity, transparency);

                volumePrd.samples += shadowbyVolPrd.samples;       // for heatmap
                volumePrd.rejections += shadowbyVolPrd.rejections; // for heatmap
            }
            else
                color = vec4f(albedo * lp.lightIntensity, transparency);
            if (dbg())
                printf("col: %f %f %f %f\n", color.x, color.y, color.z, color.w);
        }
        //===== TIMING=====
        if (lp.heatMapMode == 3)
            end = clock();
        //==================
        finalColor += over(color, vec4f(lp.bgColor,1.f));
    }
    finalColor /= float(lp.spp);
    if (lp.heatMapMode == 1) // samples heatmap
    {
        int samples = volumePrd.samples * (lp.volume.meshType == 0 ? 1 : 10 / lp.volume.numChannels);
        lp.fbPtr[fbOfs] = make_rgba(vec4f(samples / lp.heatMapScale, samples / lp.heatMapScale, samples / lp.heatMapScale, 1.f));
    }
    else if (lp.heatMapMode == 2) // rejections heatmap
    {
        int rejections = volumePrd.rejections * (lp.volume.meshType == 0 ? 1 : 10 / lp.volume.numChannels);
        lp.fbPtr[fbOfs] = make_rgba(vec4f(rejections / lp.heatMapScale, rejections / lp.heatMapScale, rejections / lp.heatMapScale, 1.f));
    }
    else if (lp.heatMapMode == 3) // timers heatmap
    {
        float time = (end - start) / lp.heatMapScale;
        const vec4f heatColor = vec4f(time, time, time, 1.f);
        if (lp.enableAccumulation)
        {
            const vec4f accumColor = lp.accumBuffer[fbOfs];
            vec4f heatColor = vec4f(time, time, time, 1.f);
            heatColor = (heatColor + float(lp.accumID) * accumColor) / float(lp.accumID + 1);
            lp.fbPtr[fbOfs] = make_rgba(heatColor);
            lp.accumBuffer[fbOfs] = vec4f(heatColor);
        }
        else
            lp.fbPtr[fbOfs] = make_rgba(heatColor);
    }
    else
    {
        if (lp.enableAccumulation)
        {
            const vec4f accumColor = lp.accumBuffer[fbOfs];
            finalColor = (vec4f(finalColor) + float(lp.accumID) * accumColor) / float(lp.accumID + 1);
            lp.fbPtr[fbOfs] = make_rgba(vec4f(finalColor));
            lp.accumBuffer[fbOfs] = vec4f(finalColor);
        }
        else
            lp.fbPtr[fbOfs] = make_rgba(vec4f(finalColor));
#ifdef ACTIVATE_CROSSHAIRS
        if (pixelID.x == lp.fbSize.x / 2 || pixelID.y == lp.fbSize.y / 2 ||
            pixelID.x == lp.fbSize.x / 2 + 1 || pixelID.y == lp.fbSize.y / 2 + 1 ||
            pixelID.x == lp.fbSize.x / 2 - 1 || pixelID.y == lp.fbSize.y / 2 - 1)
            lp.fbPtr[fbOfs] = make_rgba(vec4f(1.0f - finalColor.x, 1.0f - finalColor.y, 1.0f - finalColor.y, 1.f));
#endif
    }
}

OPTIX_CLOSEST_HIT_PROGRAM(triangleCH)
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


//define an enum for different volume rendering events
// enum VolumeEvent
// {
//     ABSORPTION,
//     SCATTERING,
//     NULL_COLLISION // also used as no collision
// };
OPTIX_CLOSEST_HIT_PROGRAM(cummilativeDTCH)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    auto &lp = optixLaunchParams;
    const MacrocellData &self = owl::getProgramData<MacrocellData>();
    prd.missed = true;
    prd.rgba = vec4f(0.0f, 0.0f, 0.0f, 0.0f);

    float unit = lp.volume.dt;

    vec3f worldOrg = optixGetWorldRayOrigin();
    vec3f org = optixGetWorldRayOrigin();
    vec3f worldDir = optixGetWorldRayDirection();

    // assuming ray is already in voxel space
    box3f worlddim = {{lp.volume.globalBoundsLo.x, lp.volume.globalBoundsLo.y, lp.volume.globalBoundsLo.z},
                      {lp.volume.globalBoundsHi.x, lp.volume.globalBoundsHi.y, lp.volume.globalBoundsHi.z}};
    vec3ui mcDim = lp.volume.macrocellDims;

    org = org - worlddim.lower;

    const vec3f worldToUnit = 1.f / (worlddim.upper - worlddim.lower);
    const vec3f unitToGrid = vec3f(mcDim.x, mcDim.y, mcDim.z);
    
    org = unitToGrid *worldToUnit * org;
    vec3f dir = worldToUnit * unitToGrid * worldDir;
    const float worldToGridT = length(dir);
    const float gridToWorldT = 1.f / worldToGridT;
    dir = normalize(dir);

    //VolumeEvent event = NULL_COLLISION;
    auto lambda = [&](const vec3i &cellIdx, float t0, float t1) -> bool
    {
        const int cellID = cellIdx.x + cellIdx.y * mcDim.x + cellIdx.z * mcDim.x * mcDim.y;
        float majorant = lp.volume.majorants[cellID];

        if(prd.debug)
            printf("cellID = %d, majorant = %f\n", cellID, majorant);

        if (majorant == 0.00f)
            return true;

        float t = t0;

        // Sample free-flight distance
        while (true)
        {
            //t_{i} = t_{i-1} - ln(1-rand())/mu_{t,max}
            //NOTE: this "unit" can be considered as a global opacity scale ass it makes sampling a point
            // more/less probable by altering the length of the woodcock step size
            t = t - (log(1.0f - prd.rng()) / majorant) * unit * worldToGridT;

            // A cell boundary has been hit
            if (t >= t1){
                //event = NULL_COLLISION;
                break; // go to next cell
            }

            // Update current position
            const float tWorld = t * gridToWorldT;
            const vec3f xTexture = (worldOrg + tWorld * worldDir) * worldToUnit;
            // A world boundary has been hit
            if (tWorld >= prd.t1)
            {
                //event = NULL_COLLISION;
                prd.rejections++;
                return false; // terminate traversal
            }
            
            //density(w component of float4) at TF(ray(t)) similar to spectrum(TR * 1 - max(0, density * invMaxDensity)) in pbrt
            //get values from all meshes and decide which one the sample is gonna come from
            float nullColThreshold = prd.rng() * majorant;

            for(int channelID = 0; channelID < lp.volume.numChannels; channelID++)
            {
                const float value = sampleVolumeTexture(xTexture, channelID);
                prd.samples++;
                if(isnan(value)) // miss: this shouldnt happen in structured volumes
                {
                    //event = NULL_COLLISION;
                    continue;
                }
                const float4 curSample = transferFunction(value, channelID);
                if(curSample.w > 0.0f && nullColThreshold < curSample.w)
                {
                    //event = ABSORPTION;
                    prd.tHit = tWorld;
                    prd.rgba = curSample;
                    prd.rgba.w = 1.0f;
                    prd.missed = false;
                    return false;
                }
                nullColThreshold -= curSample.w;
            }
            //if the process survies all meshes, it is a null collision, keep going
            //event = NULL_COLLISION;
            prd.rejections++;
        }

        return true;
    };
    dda::dda3(org,dir,1e20f,mcDim,lambda,false);
}

OPTIX_CLOSEST_HIT_PROGRAM(multiMajDTCH)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    auto &lp = optixLaunchParams;
    const MacrocellData &self = owl::getProgramData<MacrocellData>();
    prd.missed = true;
    prd.rgba = vec4f(0.0f, 0.0f, 0.0f, 0.0f);

    float unit = lp.volume.dt;

    vec3f worldOrg = optixGetWorldRayOrigin();
    vec3f org = optixGetWorldRayOrigin();
    vec3f worldDir = optixGetWorldRayDirection();

    // assuming ray is already in voxel space
    box3f worlddim = {{lp.volume.globalBoundsLo.x, lp.volume.globalBoundsLo.y, lp.volume.globalBoundsLo.z},
                      {lp.volume.globalBoundsHi.x, lp.volume.globalBoundsHi.y, lp.volume.globalBoundsHi.z}};
    vec3ui mcDim = lp.volume.macrocellDims;

    org = org - worlddim.lower;

    const vec3f worldToUnit = 1.f / (worlddim.upper - worlddim.lower);
    const vec3f unitToGrid = vec3f(mcDim.x, mcDim.y, mcDim.z);
    
    org = unitToGrid *worldToUnit * org;
    vec3f dir = worldToUnit * unitToGrid * worldDir;
    const float worldToGridT = length(dir);
    const float gridToWorldT = 1.f / worldToGridT;
    dir = normalize(dir);

    float majorants[MAX_CHANNELS];
    float ts[MAX_CHANNELS];
    //VolumeEvent event = NULL_COLLISION;
    auto lambda = [&](const vec3i &cellIdx, float t0, float t1) -> bool
    {
        const int cellID = cellIdx.x + cellIdx.y * mcDim.x + cellIdx.z * mcDim.x * mcDim.y;

        float majorantSum = 0.0f;

        for (int i = 0; i < lp.volume.numChannels; i++)
        {
            majorants[i] = lp.volume.majorants[cellID * lp.volume.numChannels + i];
            majorantSum += majorants[i];
            ts[i] = t0;
        }

        if(prd.debug)
            for (int i = 0; i < lp.volume.numChannels; i++)
                printf("cellID = %d, majorant = %f\n", cellID, majorants[i]);

        if (majorantSum <= 0.0000001f)
            return true;

        for (int i = 0; i < lp.volume.numChannels; i++)
        {
            if(majorants[i] > 0.0f)
                ts[i] = ts[i] - (log(1.0f - prd.rng()) / majorants[i]) * unit * worldToGridT;
            else
                ts[i] = 1e20f;
        }
        
        // Sample free-flight distance
        while (true)
        {
            //t_{i} = t_{i-1} - ln(1-rand())/mu_{t,max}
            //NOTE: this "unit" can be considered as a global opacity scale ass it makes sampling a point
            // more/less probable by altering the length of the woodcock step size
            float minT = ts[0];
            int selectedChannel = 0;
            auto rand = prd.rng();
            for (int i = 1; i < lp.volume.numChannels; i++)
            {
                if(ts[i] < minT && majorants[i] > 0.f)
                {
                    selectedChannel = i;
                    minT = ts[i];
                }
            }

            // A cell boundary has been hit
            if (minT >= t1){
                //event = NULL_COLLISION;
                break; // go to next cell
            }

            // Update current position
            const float tWorld = minT * gridToWorldT;
            const vec3f xTexture = (worldOrg + tWorld * worldDir) * worldToUnit;
            // A world boundary has been hit
            if (tWorld >= prd.t1)
            {
                //event = NULL_COLLISION;
                prd.rejections++;
                return false; // terminate traversal
            }
            
            //density(w component of float4) at TF(ray(t)) similar to spectrum(TR * 1 - max(0, density * invMaxDensity)) in pbrt
            //get values from all meshes and decide which one the sample is gonna come from
            float nullColThreshold = prd.rng() * (majorants[selectedChannel]);
            const float value = sampleVolumeTexture(xTexture, selectedChannel);
            prd.samples++;
            if(!isnan(value))//event = NULL_COLLISION
            {
                const float4 curSample = transferFunction(value, selectedChannel);
                //sample a mesh based on its opacity
                if(curSample.w > 0.0f && nullColThreshold < curSample.w)
                {
                    //event = ABSORPTION;
                    prd.tHit = tWorld;
                    prd.rgba = curSample;
                    prd.rgba.w = 1.0f;
                    prd.missed = false;
                    return false;
                }
            }
            //if the process survies all meshes, it is a null collision, keep going
            //event = NULL_COLLISION;
            prd.rejections++;
            ts[selectedChannel] = ts[selectedChannel] - (log(1.0f - rand) / majorants[selectedChannel]) * unit * worldToGridT;
        }

        return true;
    };
    dda::dda3(org,dir,1e20f,mcDim,lambda,false);
}

OPTIX_CLOSEST_HIT_PROGRAM(altMultiMajDTCH)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    auto &lp = optixLaunchParams;
    const MacrocellData &self = owl::getProgramData<MacrocellData>();
    prd.missed = true;
    prd.rgba = vec4f(0.0f, 0.0f, 0.0f, 0.0f);

    float unit = lp.volume.dt;

    vec3f worldOrg = optixGetWorldRayOrigin();
    vec3f org = optixGetWorldRayOrigin();
    vec3f worldDir = optixGetWorldRayDirection();

    // assuming ray is already in voxel space
    box3f worlddim = {{lp.volume.globalBoundsLo.x, lp.volume.globalBoundsLo.y, lp.volume.globalBoundsLo.z},
                      {lp.volume.globalBoundsHi.x, lp.volume.globalBoundsHi.y, lp.volume.globalBoundsHi.z}};
    vec3ui mcDim = lp.volume.macrocellDims;

    org = org - worlddim.lower;

    const vec3f worldToUnit = 1.f / (worlddim.upper - worlddim.lower);
    const vec3f unitToGrid = vec3f(mcDim.x, mcDim.y, mcDim.z);
    
    org = unitToGrid *worldToUnit * org;
    vec3f dir = worldToUnit * unitToGrid * worldDir;
    const float worldToGridT = length(dir);
    const float gridToWorldT = 1.f / worldToGridT;
    dir = normalize(dir);

    float majorant;
    float t;
    //VolumeEvent event = NULL_COLLISION;
    auto lambda = [&](const vec3i &cellIdx, float t0, float t1) -> bool
    {
        const int cellID = cellIdx.x + cellIdx.y * mcDim.x + cellIdx.z * mcDim.x * mcDim.y;
        float minT = t1; // distance to the closest cell boundary

        for (int channelID = 0; channelID < lp.volume.numChannels; channelID++)
        {
            majorant = lp.volume.majorants[cellID * lp.volume.numChannels + channelID];
            t = t0;

            if(prd.debug)
                    printf("cellID = %d, channel: %d majorant = %f\n", cellID, channelID, majorant);

            if (majorant == 0.0f)
            {
                continue;
            }

            // Sample free-flight distance
            while (true)
            {
                //t_{i} = t_{i-1} - ln(1-rand())/mu_{t,max}
                //NOTE: this "unit" can be considered as a global opacity scale ass it makes sampling a point
                // more/less probable by altering the length of the woodcock step size
                t = t - (log(1.0f - prd.rng()) / majorant) * unit * worldToGridT;

                if(t >= minT)
                    break; //this channel is not the closest sample

                // Update current position
                const float tWorld = t * gridToWorldT;
                const vec3f xTexture = (worldOrg + tWorld * worldDir) * worldToUnit;
                // A world boundary has been hit
                if (tWorld >= prd.t1)
                {
                    //event = NULL_COLLISION;
                    prd.rejections++;
                    if(channelID == lp.volume.numChannels -1)
                        return false; // terminate traversal
                    else
                        break;
                }

                
                //density(w component of float4) at TF(ray(t)) similar to spectrum(TR * 1 - max(0, density * invMaxDensity)) in pbrt
                //get values from all meshes and decide which one the sample is gonna come from
                float nullColThreshold = prd.rng() * majorant;
                const float value = sampleVolumeTexture(xTexture, channelID);
                prd.samples++;
                if(!isnan(value))//event = NULL_COLLISION
                {
                    const float4 curSample = transferFunction(value, channelID);
                    //sample a mesh based on its opacity
                    if(curSample.w > 0.0f && nullColThreshold < curSample.w)
                    {
                        //event = ABSORPTION;
                        prd.tHit = tWorld;
                        prd.rgba = curSample;
                        prd.rgba.w = 1.0f;
                        prd.missed = false;
                        minT = min(t, minT);
                    }
                }
                //if the process survies all meshes, it is a null collision, keep going
                //event = NULL_COLLISION;
                prd.rejections++;
            }
        }
        if (prd.missed) // no hits within the cell
            return true; // go to next cell
        else
            return false; // terminate traversal
    };
    dda::dda3(org,dir,1e20f,mcDim,lambda,false);
}

OPTIX_CLOSEST_HIT_PROGRAM(baseLineDTCH)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    auto &lp = optixLaunchParams;
    const MacrocellData &self = owl::getProgramData<MacrocellData>();
    prd.missed = true;
    prd.rgba = vec4f(0.0f, 0.0f, 0.0f, 0.0f);

    float unit = lp.volume.dt;

    vec3f worldOrg = optixGetWorldRayOrigin();
    vec3f org = optixGetWorldRayOrigin();
    vec3f worldDir = optixGetWorldRayDirection();

    // assuming ray is already in voxel space
    box3f worlddim = {{lp.volume.globalBoundsLo.x, lp.volume.globalBoundsLo.y, lp.volume.globalBoundsLo.z},
                      {lp.volume.globalBoundsHi.x, lp.volume.globalBoundsHi.y, lp.volume.globalBoundsHi.z}};
    vec3ui mcDim = lp.volume.macrocellDims;

    org = org - worlddim.lower;

    const vec3f worldToUnit = 1.f / (worlddim.upper - worlddim.lower);
    const vec3f unitToGrid = vec3f(mcDim.x, mcDim.y, mcDim.z);
    
    org = unitToGrid *worldToUnit * org;
    vec3f dir = worldToUnit * unitToGrid * worldDir;
    const float worldToGridT = length(dir);
    const float gridToWorldT = 1.f / worldToGridT;
    //const float worldToUnitT = owl::length(mcDim) / length(worlddim.upper - worlddim.lower);
    dir = normalize(dir);

    int curMesh = 0;
    float tMax = 1e20f;
    
    //VolumeEvent event = NULL_COLLISION;
    auto lambda = [&](const vec3i &cellIdx, float t0, float t1) -> bool
    {
        const int cellID = cellIdx.x + cellIdx.y * mcDim.x + cellIdx.z * mcDim.x * mcDim.y;
        float majorant = lp.volume.majorants[cellID * lp.volume.numChannels + curMesh];

        if(prd.debug)
            printf("cellID = %d, majorant = %f\n", cellID, majorant);

        if (majorant == 0.00f)
            return true;

        float t = t0;

        // Sample free-flight distance
        while (true)
        {
            //t_{i} = t_{i-1} - ln(1-rand())/mu_{t,max}
            //NOTE: this "unit" can be considered as a global opacity scale ass it makes sampling a point
            // more/less probable by altering the length of the woodcock step size
            t = t - (log(1.0f - prd.rng()) / majorant) * unit * worldToGridT;

            // A cell boundary has been hit
            if (t >= t1){
                //event = NULL_COLLISION;
                break; // go to next cell
            }

            if(tMax < t)
                return false;

            // Update current position
            const float tWorld = t * gridToWorldT;
            const vec3f xTexture = (worldOrg + tWorld * worldDir) * worldToUnit;
            // A world boundary has been hit
            if (tWorld >= prd.t1)
            {
                //event = NULL_COLLISION;
                prd.rejections++;
                return false; // terminate traversal
            }
            
            //density(w component of float4) at TF(ray(t)) similar to spectrum(TR * 1 - max(0, density * invMaxDensity)) in pbrt
            //get values from all meshes and decide which one the sample is gonna come from
            float nullColThreshold = prd.rng() * majorant;
            const float value = sampleVolumeTexture(xTexture, curMesh);
            prd.samples++;
            if(isnan(value)) // miss: this shouldnt happen in structured volumes
            {
                //event = NULL_COLLISION;
                continue;
            }
            const float4 curSample = transferFunction(value, curMesh);
            //sample a mesh based on its opacity
            if(curSample.w > 0.0f && nullColThreshold < curSample.w)
            {
                //event = ABSORPTION;
                prd.tHit = tWorld;
                prd.rgba = curSample;
                prd.rgba.w = 1.0f;
                prd.missed = false;
                tMax = min(tMax, t);
                return false;
            }
            //if the process survies all meshes, it is a null collision, keep going
            //event = NULL_COLLISION;
            prd.rejections++;
        }

        return true;
    };
    for (int i = 0; i < lp.volume.numChannels; i++)
    {
        curMesh = i;
        dda::dda3(org,dir,1e20,mcDim,lambda,false);
    }
}

OPTIX_CLOSEST_HIT_PROGRAM(blendDTCH)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    auto &lp = optixLaunchParams;
    const MacrocellData &self = owl::getProgramData<MacrocellData>();
    prd.missed = true;
    prd.rgba = vec4f(0.0f, 0.0f, 0.0f, 0.0f);

    float unit = lp.volume.dt;

    vec3f worldOrg = optixGetWorldRayOrigin();
    vec3f org = optixGetWorldRayOrigin();
    vec3f worldDir = optixGetWorldRayDirection();

    // assuming ray is already in voxel space
    box3f worlddim = {{lp.volume.globalBoundsLo.x, lp.volume.globalBoundsLo.y, lp.volume.globalBoundsLo.z},
                      {lp.volume.globalBoundsHi.x, lp.volume.globalBoundsHi.y, lp.volume.globalBoundsHi.z}};
    vec3ui mcDim = lp.volume.macrocellDims;

    org = org - worlddim.lower;

    const vec3f worldToUnit = 1.f / (worlddim.upper - worlddim.lower);
    const vec3f unitToGrid = vec3f(mcDim.x, mcDim.y, mcDim.z);
    
    org = unitToGrid *worldToUnit * org;
    vec3f dir = worldToUnit * unitToGrid * worldDir;
    const float worldToGridT = length(dir);
    const float gridToWorldT = 1.f / worldToGridT;
    dir = normalize(dir);

    //VolumeEvent event = NULL_COLLISION;
    auto lambda = [&](const vec3i &cellIdx, float t0, float t1) -> bool
    {
        const int cellID = cellIdx.x + cellIdx.y * mcDim.x + cellIdx.z * mcDim.x * mcDim.y;
        float majorant = lp.volume.majorants[cellID];

        if(prd.debug)
            printf("cellID = %d, majorant = %f\n", cellID, majorant);

        if (majorant == 0.00f)
            return true;

        float t = t0;

        // Sample free-flight distance
        while (true)
        {
            //t_{i} = t_{i-1} - ln(1-rand())/mu_{t,max}
            //NOTE: this "unit" can be considered as a global opacity scale ass it makes sampling a point
            // more/less probable by altering the length of the woodcock step size
            t = t - (log(1.0f - prd.rng()) / majorant) * unit * worldToGridT;

            // A cell boundary has been hit
            if (t >= t1){
                //event = NULL_COLLISION;
                break; // go to next cell
            }

            // Update current position
            const float tWorld = t * gridToWorldT;
            const vec3f xTexture = (worldOrg + tWorld * worldDir) * worldToUnit;
            // A world boundary has been hit
            if (tWorld >= prd.t1)
            {
                //event = NULL_COLLISION;
                prd.rejections++;
                return false; // terminate traversal
            }
            
            //density(w component of float4) at TF(ray(t)) similar to spectrum(TR * 1 - max(0, density * invMaxDensity)) in pbrt
            //get values from all meshes and decide which one the sample is gonna come from
            vec4f blendedSample = blendChannels(xTexture);
            prd.samples++;
            
            if(blendedSample.w > 0.0f && prd.rng() * majorant < blendedSample.w)
            {
                //event = ABSORPTION;
                prd.tHit = tWorld;
                prd.rgba = blendedSample;
                prd.rgba.w = 1.0f;
                prd.missed = false;
                return false;
            }
            
            //if the process survies all meshes, it is a null collision, keep going
            //event = NULL_COLLISION;
            prd.rejections++;
        }

        return true;
    };
    dda::dda3(org,dir,1e20f,mcDim,lambda,false);
}

OPTIX_CLOSEST_HIT_PROGRAM(rayMarcherCH)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    auto &lp = optixLaunchParams;
    const MacrocellData &self = owl::getProgramData<MacrocellData>();
    prd.missed = false;
    prd.rgba = vec4f(0.0f, 0.0f, 0.0f, 0.0f);


    box3f worlddim = {{lp.volume.globalBoundsLo.x, lp.volume.globalBoundsLo.y, lp.volume.globalBoundsLo.z},
                      {lp.volume.globalBoundsHi.x, lp.volume.globalBoundsHi.y, lp.volume.globalBoundsHi.z}};
    const vec3f worldToUnit = 1.f / (worlddim.upper - worlddim.lower);

    float dt = lp.volume.dt;

    //implement a ray marcher that leaps dt step at a time
    // and takes samples at given points until opacity reaches 1.0
    vec3f org = optixGetWorldRayOrigin();
    vec3f dir = optixGetWorldRayDirection();
    vec3f pos = org;
    dir = normalize(dir);

    float alpha = 0.0f;
    vec3f color(0.0f,0.0f,0.0f);
    float highestContribution = 0.0f;
    float highestContribDistance = 0.0f;

    for(float t = prd.t0; (t < prd.t1) && (alpha < 0.99f); t += dt)
    {
        //vec3f shadow = 1.0f;
        const vec3f posTex = pos * worldToUnit;
        vec4f blendedColor = blendChannels(posTex);
        prd.samples += lp.volume.numChannels;
        // blendedColor.w = 1.f - pow(1.f-blendedColor.w,dt);
        color = over(color, vec3f(blendedColor) /** shadow*/, alpha, blendedColor.w);
        alpha = over(alpha, blendedColor.w);

        if(lp.enableShadows && highestContribution < blendedColor.w)
        {
            highestContribution = blendedColor.w;
            highestContribDistance = t;
        }

        if(prd.debug)
            printf("rgba %f %f %f %f pos %f %f %f\n", 
                color.x, color.y, color.z, 
                alpha, posTex.x, posTex.y, posTex.z);

        pos = org + dir * t;//take a step
    }
    vec3f shadow = 1.0f;
    if(lp.enableShadows && highestContribution > 0.0f)
    {
        const vec3f orgSh = org + dir * highestContribDistance;
        vec3f dirSh = -lp.lightDir;
        float alphaSh = 0.f;
        vec3f posSh = orgSh + dirSh;

        // typical ray AABB intersection test
        vec3f dirfrac = 1.f/dirSh;

        // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
        // origin is origin of ray
        const float t1 = (worlddim.lower.x - orgSh.x)*dirfrac.x;
        const float t2 = (worlddim.upper.x - orgSh.x)*dirfrac.x;
        const float t3 = (worlddim.lower.y - orgSh.y)*dirfrac.y;
        const float t4 = (worlddim.upper.y - orgSh.y)*dirfrac.y;
        const float t5 = (worlddim.lower.z - orgSh.z)*dirfrac.z;
        const float t6 = (worlddim.upper.z - orgSh.z)*dirfrac.z;

        float thit0 = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
        const float thit1 = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

        if( (thit1 >= 0) && ((thit0 < thit1)) )
        {
            thit0 = max(thit0, 0.00001f);
            if(length(dirSh) < 1e-5f)
                thit0 = thit1;
            for(float tSh = thit0; (tSh < thit1) && (alphaSh < 0.99f); tSh += dt){
                const vec3f posTexSh = posSh * worldToUnit;
                float opacitySh = blendChannels(posTexSh).w;
                alphaSh = over(alphaSh, opacitySh);
                posSh = orgSh + dirSh * tSh;//take a step
            }
            prd.samples += lp.volume.numChannels;
        }
        shadow = vec3f((1.f - lp.ambient) * (1.f - alphaSh)  + lp.ambient);
    }
    prd.rgba = vec4f(color * shadow,alpha);
}

OPTIX_CLOSEST_HIT_PROGRAM(compRayMarcherCH)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    auto &lp = optixLaunchParams;
    const MacrocellData &self = owl::getProgramData<MacrocellData>();
    prd.missed = false;
    prd.rgba = vec4f(0.0f, 0.0f, 0.0f, 0.0f);


    box3f worlddim = {{lp.volume.globalBoundsLo.x, lp.volume.globalBoundsLo.y, lp.volume.globalBoundsLo.z},
                      {lp.volume.globalBoundsHi.x, lp.volume.globalBoundsHi.y, lp.volume.globalBoundsHi.z}};
    const vec3f worldToUnit = 1.f / (worlddim.upper - worlddim.lower);

    float dt = lp.volume.dt;

    //implement a ray marcher that leaps dt step at a time
    // and takes samples at given points until opacity reaches 1.0
    vec3f org = optixGetWorldRayOrigin();
    vec3f dir = optixGetWorldRayDirection();
    vec3f pos = org;
    dir = normalize(dir);

    float alphas[MAX_CHANNELS];
    vec3f colors[MAX_CHANNELS];
    for(int i = 0; i < lp.volume.numChannels; i++) //init
    {
        alphas[i] = 0.0f;
        colors[i] = vec3f(0.0f,0.0f,0.0f);
    }
    for(int n = 0; n < lp.volume.numChannels; n++)
    {
        float highestContribution = 0.0f;
        float highestContribDistance = 0.0f;

        for(float t = prd.t0; (t < prd.t1) && (alphas[n] < 0.99f); t += dt)
        {
            //vec3f shadow = 1.0f;
            const vec3f posTex = pos * worldToUnit;
            float sample = sampleVolumeTexture(posTex, n);
            vec4f sampleColor;
            if (isnan(sample))
            {
                 sampleColor = vec4f(0.0f, 0.0f, 0.0f, 0.0f);
            }
            else
            {
                sampleColor = transferFunction(sample, n);
                prd.samples ++;
            }
            colors[n] = over(colors[n], vec3f(sampleColor) /** shadow*/, alphas[n], sampleColor.w);
            alphas[n] = over(alphas[n], sampleColor.w);

            if(lp.enableShadows && highestContribution < sampleColor.w)
            {
                highestContribution = sampleColor.w;
                highestContribDistance = t;
            }

            if(prd.debug)
                printf("rgba %f %f %f %f pos %f %f %f\n", 
                    colors[n].x, colors[n].y, colors[n].z, 
                    alphas[n], posTex.x, posTex.y, posTex.z);

            pos = org + dir * t;//take a step
        }
        // vec3f shadow = 1.0f;
        // if(lp.enableShadows && highestContribution > 0.0f)
        // {
        //     const vec3f orgSh = org + dir * highestContribDistance;
        //     vec3f dirSh = -lp.lightDir;
        //     float alphaSh = 0.f;
        //     vec3f posSh = orgSh + dirSh;

        //     // typical ray AABB intersection test
        //     vec3f dirfrac = 1.f/dirSh;

        //     // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
        //     // origin is origin of ray
        //     const float t1 = (worlddim.lower.x - orgSh.x)*dirfrac.x;
        //     const float t2 = (worlddim.upper.x - orgSh.x)*dirfrac.x;
        //     const float t3 = (worlddim.lower.y - orgSh.y)*dirfrac.y;
        //     const float t4 = (worlddim.upper.y - orgSh.y)*dirfrac.y;
        //     const float t5 = (worlddim.lower.z - orgSh.z)*dirfrac.z;
        //     const float t6 = (worlddim.upper.z - orgSh.z)*dirfrac.z;

        //     float thit0 = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
        //     const float thit1 = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

        //     if( (thit1 >= 0) && ((thit0 < thit1)) )
        //     {
        //         thit0 = max(thit0, 0.00001f);
        //         if(length(dirSh) < 1e-5f)
        //             thit0 = thit1;
        //         for(float tSh = thit0; (tSh < thit1) && (alphaSh < 0.99f); tSh += dt){
        //             const vec3f posTexSh = posSh * worldToUnit;
        //             float opacitySh = blendChannels(posTexSh).w;
        //             alphaSh = over(alphaSh, opacitySh);
        //             posSh = orgSh + dirSh * tSh;//take a step
        //         }
        //         prd.samples += lp.volume.numChannels;
        //     }
        //     shadow = vec3f((1.f - lp.ambient) * (1.f - alphaSh)  + lp.ambient);
        // }
    }
    //find the maximum alpha
    float alphaSum = 0.0f;
    for (int i = 0; i < lp.volume.numChannels; i++)
        alphaSum += alphas[i];
    vec3f finalColor(0.0f,0.0f,0.0f);
    if(alphaSum > 0.0f)
    {
        for (int i = 0; i < lp.volume.numChannels; i++)
            finalColor += colors[i] * alphas[i] / alphaSum;
    }
    prd.rgba = vec4f(finalColor,alphaSum/(float)lp.volume.numChannels);

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

OPTIX_BOUNDS_PROGRAM(macrocellBounds)
(
    const void *geomData,
    owl::common::box3f &primBounds,
    const int primID)
{
    const MacrocellData &self = *(const MacrocellData *)geomData;
    // if (self.maxima[primID] <= 0.f) {
    //    primBounds = box3f(); // empty box
    //  }
    //  else
    {
        primBounds = box3f();
        primBounds = primBounds.including(vec3f(self.bboxes[(primID)].lower.x,
                                                self.bboxes[(primID)].lower.y,
                                                self.bboxes[(primID)].lower.z));
        primBounds = primBounds.including(vec3f(self.bboxes[(primID)].upper.x,
                                                self.bboxes[(primID)].upper.y,
                                                self.bboxes[(primID)].upper.z));
        // primBounds.lower.x = self.bboxes[(primID * 2 + 0)].x;
        // primBounds.lower.y = self.bboxes[(primID * 2 + 0)].y;
        // primBounds.lower.z = self.bboxes[(primID * 2 + 0)].z;
        // primBounds.upper.x = self.bboxes[(primID * 2 + 1)].x;
        // primBounds.upper.y = self.bboxes[(primID * 2 + 1)].y;
        // primBounds.upper.z = self.bboxes[(primID * 2 + 1)].z;
    }
}

OPTIX_BOUNDS_PROGRAM(tetrahedraBounds)
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

OPTIX_BOUNDS_PROGRAM(pyramidBounds)
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

OPTIX_BOUNDS_PROGRAM(wedgeBounds)
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

OPTIX_BOUNDS_PROGRAM(hexahedraBounds)
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
OPTIX_INTERSECT_PROGRAM(tetrahedraPointQuery)
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

OPTIX_INTERSECT_PROGRAM(pyramidPointQuery)
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

OPTIX_INTERSECT_PROGRAM(wedgePointQuery)
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

OPTIX_INTERSECT_PROGRAM(hexahedraPointQuery)
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

  OPTIX_INTERSECT_PROGRAM(volumeIntersection)()
  {
    RayPayload &prd = owl::getPRD<RayPayload>();
    const auto &self = owl::getProgramData<MacrocellData>();
    const int primID = optixGetPrimitiveIndex() + self.offset;

    box4f bbox = self.bboxes[primID];
    float3 lb = make_float3(bbox.lower.x, bbox.lower.y, bbox.lower.z);
    float3 rt = make_float3(bbox.upper.x, bbox.upper.y, bbox.upper.z);
    float3 origin = optixGetObjectRayOrigin();

    // note, this is _not_ normalized. Useful for computing world space tmin/mmax
    float3 direction = optixGetObjectRayDirection();

    // float3 rt = make_float3(mx.x(), mx.y(), mx.z() + 1.f);

    // typical ray AABB intersection test
    float3 dirfrac;

    // direction is unit direction vector of ray
    dirfrac.x = 1.0f / direction.x;
    dirfrac.y = 1.0f / direction.y;
    dirfrac.z = 1.0f / direction.z;

    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // origin is origin of ray
    float t1 = (lb.x - origin.x)*dirfrac.x;
    float t2 = (rt.x - origin.x)*dirfrac.x;
    float t3 = (lb.y - origin.y)*dirfrac.y;
    float t4 = (rt.y - origin.y)*dirfrac.y;
    float t5 = (lb.z - origin.z)*dirfrac.z;
    float t6 = (rt.z - origin.z)*dirfrac.z;

    float thit0 = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float thit1 = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (thit1 < 0) { return; }

    // if tmin > tmax, ray doesn't intersect AABB
    if (thit0 >= thit1) { return; }

    // clip hit to near position
    thit0 = max(thit0, optixGetRayTmin());

    if (optixReportIntersection(thit0, /* hit kind */ 0)) 
    {
      prd.t0 = max(prd.t0, thit0);
      prd.t1 = min(prd.t1, thit1);
    }
  }



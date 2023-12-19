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
float sampleVolumeTexture(const vec3f& normalizedPos, const int meshID = 0)
{
    auto &lp = optixLaunchParams;
    float value = tex3D<float>(lp.volume.sGrid[meshID].scalarTex, 
            normalizedPos.x, normalizedPos.y,normalizedPos.z);
                    
    // Sample scalar field
    return value;
}

inline __device__
float sampleVolume(const vec3f& pos, const int meshID = 0)
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
        return sampleVolumeTexture(normalizedPos, meshID);
    }
}


OPTIX_RAYGEN_PROGRAM(mainRG)
()
{
    auto &lp = optixLaunchParams;
    const vec2i pixelID = owl::getLaunchIndex();
    const int fbOfs = pixelID.x + lp.fbSize.x * pixelID.y;

    //generate ray
    int seed = owl::getLaunchDims().x * owl::getLaunchDims().y * lp.frameID;
    owl::common::LCG<4> random(threadIdx.x + seed, threadIdx.y + seed);//jittered sampling
    const vec2f screen = (vec2f(pixelID) + random()) / vec2f(lp.fbSize);
    Ray ray;
    generateRay(screen, ray);

    //test surface intersections first
    RayPayload surfPrd;
    vec4f finalColor = vec4f(lp.bgColor,1.f);//vec4f(missCheckerBoard(), 1.0f);
    vec4f color = vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    traceRay(lp.triangleTLAS, ray, surfPrd, OPTIX_RAY_FLAG_DISABLE_ANYHIT); //surface
    if (!surfPrd.missed)
         color = surfPrd.rgba;

    const float tMax = surfPrd.missed ? 1e20 : surfPrd.tHit;

    //test for root macrocell intersection
    RayPayload volumePrd;
    volumePrd.debug = dbg();
    volumePrd.t0 = 0.f;
    volumePrd.t1 = tMax;
    traceRay(lp.volume.rootMacrocellTLAS, ray, volumePrd, OPTIX_RAY_FLAG_DISABLE_ANYHIT); //root macrocell to initiate dda traversal
    if(!volumePrd.missed)
    {
        vec3f albedo = vec3f(volumePrd.rgba);
        if(lp.enableShadows)
        {
            // trace shadow rays
            RayPayload shadowbyVolPrd;
            shadowbyVolPrd.debug = dbg();
            shadowbyVolPrd.t0 = 0.f;
            shadowbyVolPrd.t1 = 1e20f; //todo fix this

            Ray shadowRay;
            shadowRay.origin = ray.origin + volumePrd.tHit * ray.direction;
            shadowRay.direction = -lp.lightDir;
            shadowRay.tmin = 0.00f;
            shadowRay.tmax = 1e20f;

            traceRay(lp.volume.rootMacrocellTLAS, shadowRay, shadowbyVolPrd, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
            vec3f shadow((1.f - lp.ambient) * (1.f - shadowbyVolPrd.rgba.w)  + lp.ambient);
            color = vec4f(albedo * shadow * lp.lightIntensity, 1.0f);

            volumePrd.samples += shadowbyVolPrd.samples;// for heatmap
            volumePrd.rejections += shadowbyVolPrd.rejections;// for heatmap
        }
        else
            color = vec4f(albedo * lp.lightIntensity, 1.0f);
    }

    if(lp.heatMapMode == 1)
    {
        //heatmap
        int samples = volumePrd.samples * (lp.volume.meshType == 0 ? 1 : 10 / lp.volume.numMeshes);
        lp.fbPtr[fbOfs] = make_rgba(vec4f(samples / 250.f, samples / 250.f, samples / 250.f, 1.f));
    }
    else if(lp.heatMapMode == 2)
    {
        //heatmap
        int rejections = volumePrd.rejections * (lp.volume.meshType == 0 ? 1 : 10 / lp.volume.numMeshes);
        lp.fbPtr[fbOfs] = make_rgba(vec4f(rejections / 250.f, rejections / 250.f, rejections / 250.f, 1.f));
    }
    else
    {
        finalColor = over(color, finalColor);
        if(lp.enableAccumulation)
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
        lp.fbPtr[fbOfs] = make_rgba(vec4f(1.0f-finalColor.x, 1.0f-finalColor.y, 1.0f-finalColor.y, 1.f));
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
OPTIX_CLOSEST_HIT_PROGRAM(adaptiveDTCH)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    auto &lp = optixLaunchParams;
    const MacrocellData &self = owl::getProgramData<MacrocellData>();
    prd.missed = true;
    prd.rgba = vec4f(0.0f, 0.0f, 0.0f, 0.0f);

    float unit = lp.volume.globalOpacity;

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
    const float gridToWorldT = 1.f / length(dir);
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
            t = t - (log(1.0f - prd.rng()) / majorant) * unit;

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
            float meshSelector = prd.rng() * majorant;
            for(int meshID = 0; meshID < lp.volume.numMeshes; meshID++)
            {
                const float value = sampleVolumeTexture(xTexture, meshID);
                prd.samples++;
                if(isnan(value)) // miss: this shouldnt happen in structured volumes
                {
                    //event = NULL_COLLISION;
                    continue;
                }
                const float4 curSample = transferFunction(value, meshID);
                //sample a mesh based on its opacity
                if(curSample.w > 0.0f && meshSelector < curSample.w)
                {
                    //event = ABSORPTION;
                    prd.tHit = tWorld;
                    prd.rgba = curSample;
                    prd.rgba.w = 1.0f;
                    prd.missed = false;
                    return false;
                }
                meshSelector -= curSample.w;
            }
            //if the process survies all meshes, it is a null collision, keep going
            //event = NULL_COLLISION;
            prd.rejections++;
        }

        return true;
    };
    dda::dda3(org,dir,1e20f,mcDim,lambda,false);
}

OPTIX_CLOSEST_HIT_PROGRAM(adaptiveMMDTCH)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    auto &lp = optixLaunchParams;
    const MacrocellData &self = owl::getProgramData<MacrocellData>();
    prd.missed = true;
    prd.rgba = vec4f(0.0f, 0.0f, 0.0f, 0.0f);

    float unit = lp.volume.globalOpacity;

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
    const float gridToWorldT = 1.f / length(dir);
    dir = normalize(dir);

    float majorants[MAX_MESHES];
    float ts[MAX_MESHES];
    //VolumeEvent event = NULL_COLLISION;
    auto lambda = [&](const vec3i &cellIdx, float t0, float t1) -> bool
    {
        const int cellID = cellIdx.x + cellIdx.y * mcDim.x + cellIdx.z * mcDim.x * mcDim.y;

        float majorantSum = 0.0f;

        for (int i = 0; i < lp.volume.numMeshes; i++)
        {
            majorants[i] = lp.volume.majorants[cellID * lp.volume.numMeshes + i];
            majorantSum += majorants[i];
            ts[i] = t0;
        }

        if(prd.debug)
            for (int i = 0; i < lp.volume.numMeshes; i++)
                printf("cellID = %d, majorant = %f\n", cellID, majorants[i]);

        if (majorantSum == 0.00f)
            return true;

        for (int i = 0; i < lp.volume.numMeshes; i++)
            ts[i] = ts[i] - (log(1.0f - prd.rng()) / majorants[i]) * unit;

        // Sample free-flight distance
        while (true)
        {
            //t_{i} = t_{i-1} - ln(1-rand())/mu_{t,max}
            //NOTE: this "unit" can be considered as a global opacity scale ass it makes sampling a point
            // more/less probable by altering the length of the woodcock step size
            float minT = ts[0];
            int selectedChannel = 0;
            auto rand = prd.rng();
            for (int i = 1; i < lp.volume.numMeshes; i++)
            {
                if(ts[i] < minT)
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
            float meshSelector = prd.rng() * (majorants[selectedChannel]);
            const float value = sampleVolumeTexture(xTexture, selectedChannel);
            prd.samples++;
            if(isnan(value)) // miss: this shouldnt happen in structured volumes
            {
                //event = NULL_COLLISION;
                continue;
            }
            const float4 curSample = transferFunction(value, selectedChannel);
            //sample a mesh based on its opacity
            if(curSample.w > 0.0f && meshSelector < curSample.w)
            {
                //event = ABSORPTION;
                prd.tHit = tWorld;
                prd.rgba = curSample;
                prd.rgba.w = 1.0f;
                prd.missed = false;
                return false;
            }
            //if the process survies all meshes, it is a null collision, keep going
            //event = NULL_COLLISION;
            prd.rejections++;
            ts[selectedChannel] = ts[selectedChannel] - (log(1.0f - rand) / majorants[selectedChannel]) * unit;
        }

        return true;
    };
    dda::dda3(org,dir,1e20f,mcDim,lambda,false);
}

OPTIX_CLOSEST_HIT_PROGRAM(adaptiveBaseLineDTCH)
()
{
    RayPayload &prd = owl::getPRD<RayPayload>();
    auto &lp = optixLaunchParams;
    const MacrocellData &self = owl::getProgramData<MacrocellData>();
    prd.missed = true;
    prd.rgba = vec4f(0.0f, 0.0f, 0.0f, 0.0f);

    float unit = lp.volume.globalOpacity;

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
    const float gridToWorldT = 1.f / length(dir);
    //const float worldToUnitT = owl::length(mcDim) / length(worlddim.upper - worlddim.lower);
    dir = normalize(dir);

    float majorants[MAX_MESHES];
    float ts[MAX_MESHES];
    int curMesh = 0;
    float tMax = 1e20f;
    
    //VolumeEvent event = NULL_COLLISION;
    auto lambda = [&](const vec3i &cellIdx, float t0, float t1) -> bool
    {
        const int cellID = cellIdx.x + cellIdx.y * mcDim.x + cellIdx.z * mcDim.x * mcDim.y;
        float majorant = lp.volume.majorants[cellID * lp.volume.numMeshes + curMesh];

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
            t = t - (log(1.0f - prd.rng()) / majorant) * unit;

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
            float meshSelector = prd.rng() * majorant;
            const float value = sampleVolumeTexture(xTexture, curMesh);
            prd.samples++;
            if(isnan(value)) // miss: this shouldnt happen in structured volumes
            {
                //event = NULL_COLLISION;
                continue;
            }
            const float4 curSample = transferFunction(value, curMesh);
            //sample a mesh based on its opacity
            if(curSample.w > 0.0f && meshSelector < curSample.w)
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
    for (int i = 0; i < lp.volume.numMeshes; i++)
    {
        curMesh = i;
        dda::dda3(org,dir,1e20,mcDim,lambda,false);
    }
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



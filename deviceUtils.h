#pragma once

#include "deviceCode.h"

namespace dtracker
{

inline __both__ vec3f over(vec3f Cin, vec3f Cx, float Ain, float Ax)
{
    return Cin + Cx * Ax * (1.f - Ain);
}

inline __both__ float over(const float Ain, const float Ax)
{
    return Ain + (1.f - Ain) * Ax;
}

inline __both__ vec4f over(const vec4f &in, const vec4f &x)
{
    auto c = over(vec3f(in), vec3f(x), in.w, x.w);
    auto a = over(in.w, x.w);
    return vec4f(c, a);
}

// maps x from [domain.x, domain.y] to [0, 1], unclamped
inline __both__ float normalizeToDomain(float x, float2 domain)
{
    return (x - domain.x) / (domain.y - domain.x);
}

inline __both__ box3f volumeBounds(const LaunchParams &lp)
{
    return box3f(vec3f(lp.volume.globalBoundsLo.x, lp.volume.globalBoundsLo.y, lp.volume.globalBoundsLo.z),
                 vec3f(lp.volume.globalBoundsHi.x, lp.volume.globalBoundsHi.y, lp.volume.globalBoundsHi.z));
}

inline __both__ int linearCellIndex(const vec3i &cellIdx, const vec3ui &dims)
{
    return cellIdx.x + cellIdx.y * dims.x + cellIdx.z * dims.x * dims.y;
}

// slab test: entry and exit distances t0, t1 of org + t * dir through box
inline __device__ void intersectBox(const vec3f &org, const vec3f &dir, const box3f &box, float &t0, float &t1)
{
    const vec3f dirfrac = 1.f / dir;
    const vec3f tLo = (box.lower - org) * dirfrac;
    const vec3f tHi = (box.upper - org) * dirfrac;
    t0 = max(max(min(tLo.x, tHi.x), min(tLo.y, tHi.y)), min(tLo.z, tHi.z));
    t1 = min(min(max(tLo.x, tHi.x), max(tLo.y, tHi.y)), max(tLo.z, tHi.z));
}

// gradient of f(pos) by central differences, one-sided where a step leaves domain
template <typename SampleFn>
inline __device__ vec3f finiteDifferenceGradient(const vec3f &pos, const vec3f &step, const box3f &domain,
                                                 const SampleFn &f)
{
    vec3f gradient(0.f);
    for (int i = 0; i < 3; i++)
    {
        vec3f lo = pos, hi = pos;
        lo[i] -= step[i];
        hi[i] += step[i];
        const bool loInside = lo[i] >= domain.lower[i];
        const bool hiInside = hi[i] <= domain.upper[i];
        if (loInside && hiInside)
            gradient[i] = (f(hi) - f(lo)) / (2.f * step[i]);
        else if (hiInside)
            gradient[i] = (f(hi) - f(pos)) / step[i];
        else if (loInside)
            gradient[i] = (f(pos) - f(lo)) / step[i];
    }
    return gradient;
}

// gradient of voxel idx of a dims grid (x fastest), in data units per voxel; the stencil of scalarGradient
inline __device__ vec3f voxelGradient(const float *scalars, const vec3i &dims, const vec3i &idx)
{
    return finiteDifferenceGradient(vec3f(idx), vec3f(1.f), box3f(vec3f(0.f), vec3f(dims - vec3i(1))),
        [&](const vec3f &p) {
            return scalars[uint64_t(p.x) + uint64_t(p.y) * uint64_t(dims.x) +
                           uint64_t(p.z) * uint64_t(dims.x) * uint64_t(dims.y)];
        });
}

// [min, max] length of the vectors inside box b
inline __both__ float2 lengthRange(const box3f &b)
{
    const vec3f nearest = max(b.lower, min(vec3f(0.f), b.upper));
    const vec3f farthest = max(abs(b.lower), abs(b.upper));
    return make_float2(length(nearest), length(farthest));
}

// macrocells (inclusive index box) whose interior overlaps voxel idx's, voxels and cells tiling the same bounds
inline __both__ box3i voxelCells(const vec3i &idx, const vec3i &gridDims, const vec3i &mcDims)
{
    box3i cells;
    for (int i = 0; i < 3; i++)
    {
        cells.lower[i] = int(uint64_t(idx[i]) * uint64_t(mcDims[i]) / uint64_t(gridDims[i]));
        cells.upper[i] = int((uint64_t(idx[i] + 1) * uint64_t(mcDims[i]) + uint64_t(gridDims[i]) - 1) /
                             uint64_t(gridDims[i])) - 1;
    }
    return cells;
}

// how much longer than all voxel gradients a sample's can get from 8-bit texture filter weights (2x headroom)
constexpr float FILTER_WEIGHT_SLACK = 1.f + 8.f / 256.f;

// [min, max] gradient magnitude of samples blending the voxel gradients of a neighbourhood
struct GradientMagnitudeBounds
{
    box3f components;    // per-component bounds
    float longest = 0.f; // longest voxel gradient
    bool atFace = false; // a voxel on a volume face

    inline __device__ void extend(vec3f gradient, const vec3i &voxel, const vec3i &dims)
    {
        components.extend(gradient);
        longest = fmaxf(longest, length(gradient));
        // samples within half a voxel of a face see down to half an edge voxel's gradient
        for (int i = 0; i < 3; i++)
            if (voxel[i] == 0 || voxel[i] == dims[i] - 1)
            {
                gradient[i] *= 0.5f;
                atFace = true;
            }
        components.extend(gradient);
    }

    inline __device__ float2 range() const
    {
        float2 r = lengthRange(components); // blended gradients can cancel out
        if (!atFace) // and not outgrow the longest, except where one-sided differences weight components differently
            r.y = fminf(r.y, longest * FILTER_WEIGHT_SLACK);
        return r;
    }
};

// a world-space ray in the cell coordinates of a mcDim macrocell grid spanning bounds, for dda::dda3
struct MacrocellGridRay
{
    vec3f org;
    vec3f dir; // normalized
    vec3f worldToUnit;
    float worldToGridT;
    float gridToWorldT;

    inline __both__ MacrocellGridRay(const vec3f &worldOrg, const vec3f &worldDir,
                                     const box3f &bounds, const vec3ui &mcDim)
    {
        worldToUnit = 1.f / (bounds.upper - bounds.lower);
        const vec3f unitToGrid = vec3f(mcDim);
        org = unitToGrid * worldToUnit * (worldOrg - bounds.lower);
        dir = worldToUnit * unitToGrid * worldDir;
        worldToGridT = length(dir);
        gridToWorldT = 1.f / worldToGridT;
        dir = normalize(dir);
    }
};

} // namespace dtracker

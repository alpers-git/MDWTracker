// Times macrocell builds and majorant updates for 1D, 2D and 3D transfer functions, outside the renderer
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <argparse/argparse.hpp>
#include <nvml.h>

#include "deviceUtils.h"
#include "rawFile.h"

#define CUDA_CHECK(call)                                                                   \
  do                                                                                       \
  {                                                                                        \
    const cudaError_t rc = (call);                                                         \
    if (rc != cudaSuccess)                                                                 \
      throw std::runtime_error(std::string(#call) + " failed: " + cudaGetErrorString(rc)); \
  } while (0)

namespace tfbench
{
  using namespace dtracker;

  constexpr int MAX_DIM = 3;
  constexpr int MAX_GRID_SIZE = 1024 * 1024;

  // TransferFunctionND with room for a third axis
  struct BenchTF
  {
    cudaTextureObject_t xf;
    float2 domain[MAX_DIM];
    int numTexels[MAX_DIM];
    float opacityScale;
  };

  // the device-side fields of the 1D path's TFData
  struct LegacyTF
  {
    cudaTextureObject_t colorMapTexture;
    struct
    {
      float lo, hi;
    } volDomain, xfDomain;
    int numTexels;
    float opacityScale;
  };

  //=================== copied from renderer.cu, changes marked ===================//

  __device__ __forceinline__ float atomicMin(float *address, float val)
  {
    int ret = __float_as_int(*address);
    while (val < __int_as_float(ret))
    {
      int old = ret;
      if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
        break;
    }
    return __int_as_float(ret);
  }

  __device__ __forceinline__ float atomicMax(float *address, float val)
  {
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret))
    {
      int old = ret;
      if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
        break;
    }
    return __int_as_float(ret);
  }

  inline __device__ int project(float f,
                                const interval<float> range,
                                int dim)
  {
    return max(0, min(dim - 1, int(dim * (f - range.lower) / (range.upper - range.lower))));
  }

  inline __device__
      vec3i
      project(const vec3f &pos,
              const box3f &bounds,
              const vec3i &dims)
  {
    return vec3i(project(pos.x, {bounds.lower.x, bounds.upper.x}, dims.x),
                 project(pos.y, {bounds.lower.y, bounds.upper.y}, dims.y),
                 project(pos.z, {bounds.lower.z, bounds.upper.z}, dims.z));
  }

  __global__ void sizeMCs(float2 *d_mcGrid, const vec3i dims, const box3f worldBounds, const size_t numChannels = 1)
  {
    const vec3i cellIdx = vec3i(threadIdx) + vec3i(blockIdx) * vec3i(blockDim.x, blockDim.y, blockDim.z);
    if (cellIdx.x >= dims.x)
      return;
    if (cellIdx.y >= dims.y)
      return;
    if (cellIdx.z >= dims.z)
      return;

    const uint32_t cellID = (cellIdx.x + cellIdx.y * dims.x + cellIdx.z * dims.x * dims.y) * numChannels;

    for (size_t meshIndex = 0; meshIndex < numChannels; meshIndex++)
    {
      d_mcGrid[cellID + meshIndex] = make_float2(1e20, -1e20);
    }
  }

  inline dim3 to_dims(const vec3i v)
  {
    return {(unsigned)v.x, (unsigned)v.y, (unsigned)v.z};
  }

  // atomically widens a macrocell by a [min, max] range per TF axis; float2 is the 1D macrocell
  inline __device__ void widenMacrocell(float2 &cell, const MacrocellND<1> &ranges)
  {
    atomicMin(&cell.x, ranges.range[0].x);
    atomicMax(&cell.y, ranges.range[0].y);
  }

  template <int TF_DIM>
  inline __device__ void widenMacrocell(MacrocellND<TF_DIM> &cell, const MacrocellND<TF_DIM> &ranges)
  {
    for (int a = 0; a < TF_DIM; a++)
    {
      atomicMin(&cell.range[a].x, ranges.range[a].x);
      atomicMax(&cell.range[a].y, ranges.range[a].y);
    }
  }

  // widens macrocells cells.lower to cells.upper (inclusive) by ranges
  template <typename Macrocell, int TF_DIM>
  inline __device__ void rasterBox(Macrocell *d_mcGrid,
                                   const vec3i dims,
                                   const box3i cells,
                                   const MacrocellND<TF_DIM> &ranges,
                                   const size_t meshIndex = 0,
                                   const size_t numChannels = 1)
  {
    for (uint64_t iz = cells.lower.z; iz <= cells.upper.z; iz++)
      for (uint64_t iy = cells.lower.y; iy <= cells.upper.y; iy++)
        for (uint64_t ix = cells.lower.x; ix <= cells.upper.x; ix++)
        {
          const uint64_t cellID = (ix + iy * dims.x + iz * dims.x * dims.y) * numChannels + meshIndex;
          widenMacrocell(d_mcGrid[cellID], ranges);
        }
  }

  __global__ void rasterElements(float2 *d_mcGrid,
                                 const vec3i mcDims,
                                 const float *scalars,
                                 const vec3i gridDims,
                                 const size_t meshIndex = 0,
                                 size_t numChannels = 1)
  {
    const uint64_t blockID = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * MAX_GRID_SIZE;
    uint64_t primIdx = blockID * uint64_t(blockDim.x) + uint64_t(threadIdx.x);
    const uint64_t totalElements = uint64_t(gridDims.x) * uint64_t(gridDims.y) * uint64_t(gridDims.z);
    if (primIdx >= totalElements)
    {
      return;
    }
    vec3i primIdx3D = vec3i(primIdx % gridDims.x,
                            (primIdx / gridDims.x) % gridDims.y,
                            primIdx / (gridDims.x * gridDims.y));

    // samples inside the voxel blend voxels of its neighbourhood only
    float2 valueRange = make_float2(scalars[primIdx], scalars[primIdx]);
    for (int64_t iz = -1; iz <= 1; iz++)
      for (int64_t iy = -1; iy <= 1; iy++)
        for (int64_t ix = -1; ix <= 1; ix++)
        {
          int64_t nx = (int64_t)primIdx3D.x + ix;
          int64_t ny = (int64_t)primIdx3D.y + iy;
          int64_t nz = (int64_t)primIdx3D.z + iz;

          if (nx < 0 || nx >= gridDims.x)
            continue;
          if (ny < 0 || ny >= gridDims.y)
            continue;
          if (nz < 0 || nz >= gridDims.z)
            continue;

          const uint64_t neighborIdx = (uint64_t)nx + (uint64_t)ny * (uint64_t)gridDims.x + (uint64_t)nz * (uint64_t)gridDims.x * (uint64_t)gridDims.y;
          valueRange.x = fminf(valueRange.x, scalars[neighborIdx]);
          valueRange.y = fmaxf(valueRange.y, scalars[neighborIdx]);
        }
    const MacrocellND<1> ranges = {{valueRange}};
    rasterBox(d_mcGrid, mcDims, voxelCells(primIdx3D, gridDims, mcDims), ranges, meshIndex, numChannels);
  }

  // change: LegacyTF in place of dtracker::TFData
  __global__ void _recalculateDensityRangesMM(
      int numPrims, const float2 *macrocells,
      const LegacyTF *tf, size_t numChannels, float *maxima)
  {
    int nodeID = (blockIdx.x * blockDim.x + threadIdx.x);
    if (nodeID >= numPrims)
      return;

    for (size_t tfID = 0; tfID < numChannels; tfID++)
    {
      maxima[nodeID * numChannels + tfID] = 0.f;
      float mn = macrocells[nodeID * numChannels + tfID].x;
      float mx = macrocells[nodeID * numChannels + tfID].y;

      if (mx < mn)
      {
        continue;
      }

      float remappedMin1 = (mn - tf[tfID].volDomain.lo) / (tf[tfID].volDomain.hi - tf[tfID].volDomain.lo);
      float remappedMin = (remappedMin1 - tf[tfID].xfDomain.lo) / (tf[tfID].xfDomain.hi - tf[tfID].xfDomain.lo);
      float remappedMax1 = (mx - tf[tfID].volDomain.lo) / (tf[tfID].volDomain.hi - tf[tfID].volDomain.lo);
      float remappedMax = (remappedMax1 - tf[tfID].xfDomain.lo) / (tf[tfID].xfDomain.hi - tf[tfID].xfDomain.lo);
      float addr1 = remappedMin * (tf[tfID].numTexels);
      float addr2 = remappedMax * (tf[tfID].numTexels);

      int addrMin = min(max(int(min(floor(addr1), floor(addr2))), 0), tf[tfID].numTexels - 1);
      int addrMax = min(max(int(max(ceil(addr1), ceil(addr2))), 0), tf[tfID].numTexels - 1);

      float maxDensityForVolume = 0.f;
      for (int i = addrMin; i <= addrMax; ++i)
      {
        float density = tex2D<float4>(tf[tfID].colorMapTexture, (float(i) + 0.5f) / tf[tfID].numTexels, 0.5f).w * tf[tfID].opacityScale;
        if (i == addrMin)
          maxDensityForVolume = density;
        else
          maxDensityForVolume = max(maxDensityForVolume, density);
      }
      float density = tex2D<float4>(tf[tfID].colorMapTexture, (float(addr1) + 0.5f) / tf[tfID].numTexels, 0.5f).w * tf[tfID].opacityScale;
      maxDensityForVolume = max(maxDensityForVolume, density);
      density = tex2D<float4>(tf[tfID].colorMapTexture, (float(addr2) + 0.5f) / tf[tfID].numTexels, 0.5f).w * tf[tfID].opacityScale;
      maxDensityForVolume = max(maxDensityForVolume, density);
      maxima[nodeID * numChannels + tfID] = maxDensityForVolume;
    }
  }

  //================ third TF axis: second derivative along the gradient ================//

  // second differences of voxel idx as (xx, yy, zz, xy, xz, yz), stencils moved inward at the faces
  inline __device__ void voxelHessian(const float *scalars, const vec3i &dims, const vec3i &idx, float (&h)[6])
  {
    auto f = [&](const vec3i &p)
    {
      return scalars[uint64_t(p.x) + uint64_t(p.y) * uint64_t(dims.x) + uint64_t(p.z) * uint64_t(dims.x) * uint64_t(dims.y)];
    };
    vec3i center = idx;
    for (int i = 0; i < 3; i++)
      if (dims[i] >= 3)
        center[i] = min(max(idx[i], 1), dims[i] - 2);
    for (int i = 0; i < 3; i++)
    {
      vec3i q = idx, e(0);
      q[i] = center[i];
      e[i] = 1;
      h[i] = dims[i] < 3 ? 0.f : f(q + e) - 2.f * f(q) + f(q - e);
    }
    const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int k = 0; k < 3; k++)
    {
      const int i = pairs[k][0], j = pairs[k][1];
      vec3i q = idx, ei(0), ej(0);
      q[i] = center[i];
      q[j] = center[j];
      ei[i] = 1;
      ej[j] = 1;
      h[3 + k] = dims[i] < 3 || dims[j] < 3 ? 0.f : 0.25f * (f(q + ei + ej) - f(q + ei - ej) - f(q - ei + ej) + f(q - ei - ej));
    }
  }

  // g^T H g / |g|^2, 0 where the gradient vanishes
  inline __device__ float secondDirectionalDerivative(const vec3f &g, const float (&h)[6])
  {
    const float g2 = dot(g, g);
    if (g2 == 0.f)
      return 0.f;
    return (g.x * g.x * h[0] + g.y * g.y * h[1] + g.z * g.z * h[2] +
            2.f * (g.x * g.y * h[3] + g.x * g.z * h[4] + g.y * g.z * h[5])) /
           g2;
  }

  // [min, max] of u^T H u over unit u and every Hessian inside the component bounds hb (Gershgorin discs)
  inline __device__ float2 quadraticFormRange(const float2 (&hb)[6])
  {
    float offDiagonal[3]; // largest |xy|, |xz|, |yz|
    for (int k = 0; k < 3; k++)
      offDiagonal[k] = fmaxf(fabsf(hb[3 + k].x), fabsf(hb[3 + k].y));
    const float radius[3] = {offDiagonal[0] + offDiagonal[1], offDiagonal[0] + offDiagonal[2],
                             offDiagonal[1] + offDiagonal[2]};
    float2 range = make_float2(1e20f, -1e20f);
    for (int i = 0; i < 3; i++)
    {
      range.x = fminf(range.x, hb[i].x - radius[i]);
      range.y = fmaxf(range.y, hb[i].y + radius[i]);
    }
    return range;
  }

  //==================== build variant: integer-indexed gradient ====================//

  // voxelGradient with integer indexing: the same differences without float positions
  inline __device__ vec3f voxelGradientIndexed(const float *scalars, const vec3i &dims, const vec3i &idx)
  {
    const uint64_t stride[3] = {1, uint64_t(dims.x), uint64_t(dims.x) * uint64_t(dims.y)};
    const uint64_t i = uint64_t(idx.x) + uint64_t(idx.y) * stride[1] + uint64_t(idx.z) * stride[2];
    vec3f gradient(0.f);
    for (int a = 0; a < 3; a++)
    {
      const bool loInside = idx[a] > 0, hiInside = idx[a] < dims[a] - 1;
      if (loInside && hiInside)
        gradient[a] = (scalars[i + stride[a]] - scalars[i - stride[a]]) / 2.f;
      else if (hiInside)
        gradient[a] = scalars[i + stride[a]] - scalars[i];
      else if (loInside)
        gradient[a] = scalars[i] - scalars[i - stride[a]];
    }
    return gradient;
  }

  //====================== N-D macrocells and majorants (renderer.cu) ======================//

  // empty range (min > max) on every axis
  template <int TF_DIM>
  __global__ void clearMacrocellsND(MacrocellND<TF_DIM> *d_mcGrid, const uint64_t numEntries)
  {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numEntries)
      return;
    for (int a = 0; a < TF_DIM; ++a)
      d_mcGrid[index].range[a] = make_float2(1e20f, -1e20f);
  }

  // change: TF axis 2, the range of f'' along the gradient, and the indexed gradient as an option
  template <int TF_DIM, bool INDEXED_GRADIENT = false>
  __global__ void rasterMacrocellsND(MacrocellND<TF_DIM> *d_mcGrid,
                                     const vec3i mcDims,
                                     const float *scalars,
                                     const vec3i gridDims,
                                     const size_t channelID,
                                     const size_t numChannels)
  {
    static_assert(TF_DIM <= 3, "rasterMacrocellsND: add the range of the new TF axis");
    const uint64_t primIdx = (uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * MAX_GRID_SIZE) * blockDim.x + threadIdx.x;
    if (primIdx >= uint64_t(gridDims.x) * uint64_t(gridDims.y) * uint64_t(gridDims.z))
      return;
    GradientMagnitudeBounds gradientBounds;
    [[maybe_unused]] float2 hessianBounds[6]; // per-component bounds of the neighbourhood's voxel Hessians
    for (int k = 0; k < 6; k++)
      hessianBounds[k] = make_float2(1e20f, -1e20f);

    vec3i primIdx3D = vec3i(primIdx % gridDims.x,
                            (primIdx / gridDims.x) % gridDims.y,
                            primIdx / (gridDims.x * gridDims.y));

    // samples inside the voxel blend voxels (and voxel gradients) of its neighbourhood only
    float2 valueRange = make_float2(scalars[primIdx], scalars[primIdx]);
    for (int64_t iz = -1; iz <= 1; iz++)
      for (int64_t iy = -1; iy <= 1; iy++)
        for (int64_t ix = -1; ix <= 1; ix++)
        {
          int64_t nx = (int64_t)primIdx3D.x + ix;
          int64_t ny = (int64_t)primIdx3D.y + iy;
          int64_t nz = (int64_t)primIdx3D.z + iz;

          if (nx < 0 || nx >= gridDims.x)
            continue;
          if (ny < 0 || ny >= gridDims.y)
            continue;
          if (nz < 0 || nz >= gridDims.z)
            continue;

          const uint64_t neighborIdx = (uint64_t)nx + (uint64_t)ny * (uint64_t)gridDims.x + (uint64_t)nz * (uint64_t)gridDims.x * (uint64_t)gridDims.y;
          valueRange.x = fminf(valueRange.x, scalars[neighborIdx]);
          valueRange.y = fmaxf(valueRange.y, scalars[neighborIdx]);
          if constexpr (TF_DIM > 1)
          {
            const vec3i neighbor = vec3i(int(nx), int(ny), int(nz));
            gradientBounds.extend(INDEXED_GRADIENT ? voxelGradientIndexed(scalars, gridDims, neighbor)
                                                   : voxelGradient(scalars, gridDims, neighbor),
                                  neighbor, gridDims);
          }
          if constexpr (TF_DIM > 2)
          {
            float hessian[6];
            voxelHessian(scalars, gridDims, vec3i(int(nx), int(ny), int(nz)), hessian);
            for (int k = 0; k < 6; k++)
              hessianBounds[k] = make_float2(fminf(hessianBounds[k].x, hessian[k]), fmaxf(hessianBounds[k].y, hessian[k]));
          }
        }
    MacrocellND<TF_DIM> ranges;
    ranges.range[0] = valueRange;
    if constexpr (TF_DIM > 1)
      ranges.range[1] = gradientBounds.range();
    if constexpr (TF_DIM > 2)
    {
      ranges.range[2] = quadraticFormRange(hessianBounds);
      if (ranges.range[1].x == 0.f) // f'' is 0 where the gradient can vanish
        ranges.range[2] = make_float2(fminf(ranges.range[2].x, 0.f), fmaxf(ranges.range[2].y, 0.f));
    }
    rasterBox(d_mcGrid, mcDims, voxelCells(primIdx3D, gridDims, mcDims), ranges, channelID, numChannels);
  }

  // texels a bilinear lookup inside data range r can touch, plus one for the hardware's 8-bit filter weights
  // change: __host__ too, for the sweep statistics
  inline __host__ __device__ int2 texelRange(const float2 r, const float2 domain, const int numTexels)
  {
    const float a = normalizeToDomain(r.x, domain) * numTexels - 0.5f;
    const float b = normalizeToDomain(r.y, domain) * numTexels - 0.5f;
    if (!isfinite(a) || !isfinite(b))
      return make_int2(0, numTexels - 1);
    const float last = float(numTexels - 1);
    return make_int2(int(fminf(fmaxf(floorf(fminf(a, b)) - 1.f, 0.f), last)),
                     int(fminf(fmaxf(floorf(fmaxf(a, b)) + 2.f, 0.f), last)));
  }

  // opacity of a TF_DIM-D lookup texture at normalized coordinates; 1D textures are width x 1 like the renderer's
  template <int TF_DIM>
  inline __device__ float opacityAt(const BenchTF &tf, const float *coord)
  {
    if constexpr (TF_DIM == 1)
      return tex2D<float4>(tf.xf, coord[0], 0.5f).w;
    else if constexpr (TF_DIM == 2)
      return tex2D<float4>(tf.xf, coord[0], coord[1]).w;
    else
    {
      static_assert(TF_DIM == 3, "opacityAt: no lookup texture of this dimensionality");
      return tex3D<float4>(tf.xf, coord[0], coord[1], coord[2]).w;
    }
  }

  // opacity of texel idx of a TF_DIM-D lookup texture, read at its center
  template <int TF_DIM>
  inline __device__ float texelOpacity(const BenchTF &tf, const int (&idx)[TF_DIM])
  {
    float coord[TF_DIM];
    for (int a = 0; a < TF_DIM; a++)
      coord[a] = (idx[a] + 0.5f) / tf.numTexels[a];
    return opacityAt<TF_DIM>(tf, coord);
  }

  // max opacity each channel's transfer function can return inside a macrocell's range box,
  // one warp per macrocell whose lanes split the sweep of the texel box
  // change: BenchTF in place of TransferFunctionND
  template <int TF_DIM>
  __global__ void computeMajorantsND(const uint64_t numCells,
                                     const MacrocellND<TF_DIM> *macrocells,
                                     const BenchTF *tfs,
                                     const size_t numChannels,
                                     const bool perChannel,
                                     float *majorants)
  {
    const uint64_t cellID = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / warpSize;
    const int lane = threadIdx.x % warpSize;
    if (cellID >= numCells) // the whole warp leaves together
      return;

    float sum = 0.f;
    for (size_t c = 0; c < numChannels; c++)
    {
      const MacrocellND<TF_DIM> &cell = macrocells[cellID * numChannels + c];
      const BenchTF &tf = tfs[c];

      bool empty = false; // no data of channel c in the cell
      int first[TF_DIM], extent[TF_DIM];
      int numTexels = 1;
      for (int a = 0; a < TF_DIM; a++)
      {
        empty |= cell.range[a].x > cell.range[a].y;
        const int2 texels = texelRange(cell.range[a], tf.domain[a], tf.numTexels[a]);
        first[a] = texels.x;
        extent[a] = texels.y - texels.x + 1;
        numTexels *= extent[a];
      }

      float majorant = 0.f;
      for (int k = lane; !empty && k < numTexels; k += warpSize) // sweep the texel box, axis 0 fastest
      {
        int idx[TF_DIM];
        int rest = k;
        for (int a = 0; a < TF_DIM; a++)
        {
          idx[a] = first[a] + rest % extent[a];
          rest /= extent[a];
        }
        majorant = fmaxf(majorant, texelOpacity<TF_DIM>(tf, idx));
      }
      for (int offset = warpSize / 2; offset > 0; offset /= 2)
        majorant = fmaxf(majorant, __shfl_down_sync(0xffffffff, majorant, offset));
      majorant = __shfl_sync(0xffffffff, majorant, 0) * tf.opacityScale;

      if (perChannel && lane == 0)
        majorants[cellID * numChannels + c] = majorant;
      sum += majorant;
    }
    if (!perChannel && lane == 0)
      majorants[cellID] = sum;
  }

  //============================ benchmark-only kernels ============================//

  // value, |gradient| and |f''| along the gradient of every stride-th voxel, for the TF axis domains
  __global__ void voxelAttributeSamples(const float *scalars, const vec3i dims, const uint64_t stride,
                                        const uint64_t count, float3 *samples)
  {
    const uint64_t k = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count)
      return;
    const uint64_t v = k * stride;
    const vec3i idx(int(v % dims.x), int(v / dims.x % dims.y), int(v / (uint64_t(dims.x) * uint64_t(dims.y))));
    const vec3f g = voxelGradient(scalars, dims, idx);
    float h[6];
    voxelHessian(scalars, dims, idx, h);
    samples[k] = make_float3(scalars[v], length(g), fabsf(secondDirectionalDerivative(g, h)));
  }

  // uniform in [0, 1) from a hashed index (lowbias32)
  inline __device__ float hashToUnit(const uint64_t i)
  {
    uint32_t h = uint32_t(i) ^ (uint32_t(i >> 32) * 0x9E3779B9u);
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return float(h >> 8) * (1.f / 16777216.f);
  }

  // TF axes at unit-box position p, with value, gradient and Hessian trilinearly blended from the voxels
  template <int TF_DIM>
  inline __device__ void sampleAttributes(const float *scalars, const vec3i &dims, const vec3f &p, float (&attr)[MAX_DIM])
  {
    const vec3f x = p * vec3f(dims) - vec3f(0.5f); // voxel centers sit at integer x
    vec3i base;
    vec3f t;
    for (int a = 0; a < 3; a++)
    {
      const float xa = fminf(fmaxf(x[a], 0.f), float(dims[a] - 1));
      base[a] = min(int(xa), max(dims[a] - 2, 0));
      t[a] = xa - float(base[a]);
    }
    float value = 0.f, hessian[6] = {};
    vec3f gradient(0.f);
    for (int c = 0; c < 8; c++)
    {
      const vec3i o(c & 1, (c >> 1) & 1, c >> 2);
      const vec3i v = min(base + o, dims - vec3i(1));
      const float w = (o.x ? t.x : 1.f - t.x) * (o.y ? t.y : 1.f - t.y) * (o.z ? t.z : 1.f - t.z);
      value += w * scalars[uint64_t(v.x) + uint64_t(v.y) * uint64_t(dims.x) + uint64_t(v.z) * uint64_t(dims.x) * uint64_t(dims.y)];
      if constexpr (TF_DIM > 1)
        gradient += w * voxelGradient(scalars, dims, v);
      if constexpr (TF_DIM > 2)
      {
        float h[6];
        voxelHessian(scalars, dims, v, h);
        for (int k = 0; k < 6; k++)
          hessian[k] += w * h[k];
      }
    }
    attr[0] = value;
    attr[1] = length(gradient);
    attr[2] = secondDirectionalDerivative(gradient, hessian);
  }

  // TF axes as the renderer's classify<1> and classify<2> compute them: filtered texture lookups, scalarGradient
  template <int TF_DIM>
  inline __device__ void textureAttributes(const cudaTextureObject_t volume, const vec3i &dims, const vec3f &p,
                                           float (&attr)[MAX_DIM])
  {
    static_assert(TF_DIM <= 2, "textureAttributes: the renderer has no lookup of this dimensionality");
    auto sample = [&](const vec3f &q)
    { return tex3D<float>(volume, q.x, q.y, q.z); };
    attr[0] = sample(p);
    attr[1] = attr[2] = 0.f;
    if constexpr (TF_DIM > 1)
    {
      const vec3f voxel = 1.f / vec3f(dims);
      attr[1] = length(voxel * finiteDifferenceGradient(p, voxel, box3f(vec3f(0.f), vec3f(1.f)), sample));
    }
  }

  // counts random samples outside their cell's ranges, or above its majorant if given; sampled like the renderer or exactly
  template <int TF_DIM, bool RENDERER_SAMPLING>
  __global__ void verifySamples(const uint64_t numSamples, const float *scalars, const cudaTextureObject_t volume,
                                const vec3i dims, const vec3i mcDims, const MacrocellND<TF_DIM> *macrocells,
                                const BenchTF tf, const float *majorants, unsigned long long *violations)
  {
    const uint64_t s = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= numSamples)
      return;
    const vec3f p(hashToUnit(3 * s), hashToUnit(3 * s + 1), hashToUnit(3 * s + 2));
    float attr[MAX_DIM];
    if constexpr (RENDERER_SAMPLING)
      textureAttributes<TF_DIM>(volume, dims, p, attr);
    else
      sampleAttributes<TF_DIM>(scalars, dims, p, attr);
    const vec3i cell = project(p, box3f(vec3f(0.f), vec3f(1.f)), mcDims);
    const uint64_t cellID = uint64_t(cell.x) + uint64_t(cell.y) * uint64_t(mcDims.x) +
                            uint64_t(cell.z) * uint64_t(mcDims.x) * uint64_t(mcDims.y);
    bool violated = false;
    if (majorants)
    {
      float coord[TF_DIM];
      for (int a = 0; a < TF_DIM; a++)
        coord[a] = normalizeToDomain(attr[a], tf.domain[a]);
      violated = opacityAt<TF_DIM>(tf, coord) * tf.opacityScale > majorants[cellID] + 1e-6f;
    }
    else
      for (int a = 0; a < TF_DIM; a++)
      {
        const float2 r = macrocells[cellID].range[a];
        const float tolerance = 1e-5f * fmaxf(1.f, fmaxf(fabsf(r.x), fabsf(r.y)));
        violated |= attr[a] < r.x - tolerance || attr[a] > r.y + tolerance;
      }
    if (violated)
      atomicAdd(violations, 1ull);
  }

  // mean (x) and max (y) extinction of samplesPerCell random samples inside each macrocell, one warp per cell
  template <int TF_DIM>
  __global__ void cellExtinction(const uint64_t numCells, const vec3i mcDims, const uint32_t samplesPerCell,
                                 const float *scalars, const vec3i dims, const BenchTF tf, float2 *extinction)
  {
    const uint64_t cellID = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / warpSize;
    const int lane = threadIdx.x % warpSize;
    if (cellID >= numCells) // the whole warp leaves together
      return;
    const vec3f cell(float(cellID % mcDims.x), float(cellID / mcDims.x % mcDims.y),
                     float(cellID / (uint64_t(mcDims.x) * uint64_t(mcDims.y))));
    const vec3f cellSize = vec3f(1.f) / vec3f(mcDims);

    float sum = 0.f, peak = 0.f;
    for (uint32_t j = lane; j < samplesPerCell; j += warpSize)
    {
      const uint64_t k = 3 * (cellID * samplesPerCell + j);
      const vec3f u(hashToUnit(k), hashToUnit(k + 1), hashToUnit(k + 2));
      const vec3f p = (cell + 0.001f + 0.998f * u) * cellSize; // kept off the faces, where rounding could leave the cell
      float attr[MAX_DIM];
      sampleAttributes<TF_DIM>(scalars, dims, p, attr);
      float coord[TF_DIM];
      for (int a = 0; a < TF_DIM; a++)
        coord[a] = normalizeToDomain(attr[a], tf.domain[a]);
      const float mu = opacityAt<TF_DIM>(tf, coord) * tf.opacityScale;
      sum += mu;
      peak = fmaxf(peak, mu);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
      peak = fmaxf(peak, __shfl_down_sync(0xffffffff, peak, offset));
    }
    if (lane == 0)
      extinction[cellID] = make_float2(sum / samplesPerCell, peak);
  }

  //================================== host side ==================================//

  template <typename T>
  struct DeviceArray
  {
    T *ptr = nullptr;
    size_t count = 0;

    explicit DeviceArray(const size_t n) : count(n)
    {
      CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(n, 1) * sizeof(T)));
    }
    ~DeviceArray() { cudaFree(ptr); }
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;

    void upload(const T *data) { CUDA_CHECK(cudaMemcpy(ptr, data, count * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> download() const
    {
      std::vector<T> host(count);
      CUDA_CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
      return host;
    }
  };

  // the renderer's volume texture (create3DTexture): float, linear, clamped, normalized coordinates
  struct VolumeTexture
  {
    cudaArray_t array = nullptr;
    cudaTextureObject_t tex = 0;

    VolumeTexture(const float *data, const vec3i &dims)
    {
      const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
      const cudaExtent extent = make_cudaExtent(dims.x, dims.y, dims.z);
      CUDA_CHECK(cudaMalloc3DArray(&array, &channelDesc, extent));
      cudaMemcpy3DParms copy = {};
      copy.srcPtr = make_cudaPitchedPtr((void *)data, extent.width * sizeof(float), extent.width, extent.height);
      copy.dstArray = array;
      copy.extent = extent;
      copy.kind = cudaMemcpyHostToDevice;
      CUDA_CHECK(cudaMemcpy3D(&copy));
      cudaResourceDesc resDesc = {};
      resDesc.resType = cudaResourceTypeArray;
      resDesc.res.array.array = array;
      cudaTextureDesc texDesc = {};
      texDesc.normalizedCoords = true;
      texDesc.filterMode = cudaFilterModeLinear;
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.addressMode[2] = cudaAddressModeClamp;
      texDesc.readMode = cudaReadModeElementType;
      CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    }
    ~VolumeTexture()
    {
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
    }
    VolumeTexture(const VolumeTexture &) = delete;
    VolumeTexture &operator=(const VolumeTexture &) = delete;
  };

  struct Volume
  {
    std::string name;
    vec3i dims;
    float2 valueRange;
    std::unique_ptr<DeviceArray<float>> scalars;
    std::unique_ptr<VolumeTexture> texture; // null when it does not fit, then only the exact check runs
    uint64_t numVoxels() const { return uint64_t(dims.x) * uint64_t(dims.y) * uint64_t(dims.z); }

    void upload(const std::vector<float> &data)
    {
      scalars = std::make_unique<DeviceArray<float>>(data.size());
      scalars->upload(data.data());
      try
      {
        texture = std::make_unique<VolumeTexture>(data.data(), dims);
      }
      catch (const std::exception &e)
      {
        cudaGetLastError();
        printf("no volume texture (%s): renderer-sampling checks are off\n", e.what());
      }
    }
  };

  static Volume loadRaw(const std::string &path)
  {
    if (!std::filesystem::exists(path))
      throw std::runtime_error("no such file: " + path);
    raw::RawR file(path.c_str(), "rb");
    const auto dims = file.getDims();
    if (dims.x <= 0 || dims.y <= 0 || dims.z <= 0)
      throw std::runtime_error("no <X>x<Y>x<Z> dimensions in the file name: " + path);
    Volume vol;
    vol.name = std::filesystem::path(path).stem().string();
    vol.dims = vec3i(int(dims.x), int(dims.y), int(dims.z));
    vol.valueRange = make_float2(file.getBounds4f().lower.w, file.getBounds4f().upper.w); // as the renderer's volDomain
    vol.upload(file.getDataVector());
    return vol;
  }

  static float smoothstep(const float a, const float b, const float x)
  {
    const float t = std::clamp((x - a) / (b - a), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
  }

  // a soft-edged ball over a sine pattern, values in [0, 1000]
  static Volume syntheticVolume(const int n)
  {
    std::vector<float> data(size_t(n) * n * n);
    for (int z = 0; z < n; z++)
      for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
        {
          const vec3f p = (vec3f(float(x), float(y), float(z)) + 0.5f) / float(n);
          const float ball = 1.f - smoothstep(0.28f, 0.32f, length(p - vec3f(0.5f)));
          const float waves = 0.5f + 0.5f * std::sin(20.f * p.x) * std::sin(20.f * p.y) * std::sin(20.f * p.z);
          data[x + size_t(n) * (y + size_t(n) * z)] = 1000.f * (0.7f * ball + 0.3f * waves);
        }
    Volume vol;
    vol.name = "synthetic" + std::to_string(n);
    vol.dims = vec3i(n);
    const auto [lo, hi] = std::minmax_element(data.begin(), data.end());
    vol.valueRange = make_float2(*lo, *hi);
    vol.upload(data);
    return vol;
  }

  // the benchmark's TFs: a value curve, times a |f'| ramp from 2D on and an f'' peak at 0 in 3D
  struct TFShape
  {
    std::vector<vec2f> valuePoints; // alpha control points of a .tfn; a smoothstep ramp when empty
    float2 valueRamp = make_float2(0.05f, 0.3f);
    float2 valueWindow = make_float2(0.f, 1.f); // part of the data range the value axis spans
    float opacityScale = 1.f;
    float2 gradientRamp = make_float2(0.02f, 0.2f);
    float d2Width = 0.1f;

    // as the 1D widget's UpdateColormap: piecewise linear, scaled, clamped
    float value(const float x) const
    {
      if (valuePoints.empty())
        return smoothstep(valueRamp.x, valueRamp.y, x);
      if (x <= valuePoints.front().x)
        return std::clamp(valuePoints.front().y * opacityScale, 0.f, 1.f);
      for (size_t i = 1; i < valuePoints.size(); i++)
        if (x <= valuePoints[i].x)
        {
          const vec2f &a = valuePoints[i - 1], &b = valuePoints[i];
          const float t = b.x > a.x ? (x - a.x) / (b.x - a.x) : 1.f;
          return std::clamp(((1.f - t) * a.y + t * b.y) * opacityScale, 0.f, 1.f);
        }
      return std::clamp(valuePoints.back().y * opacityScale, 0.f, 1.f);
    }

    float opacity(const int dim, const float v, const float g, const float s) const
    {
      float a = value(v);
      if (dim > 1)
        a *= smoothstep(gradientRamp.x, gradientRamp.y, g);
      if (dim > 2)
        a *= std::exp(-((s - 0.5f) / d2Width) * ((s - 0.5f) / d2Width));
      return a;
    }
  };

  // the value curve, opacity scale and window of a 1D widget state (TransferFunctionWidget::SaveState)
  static TFShape loadTfn(const std::string &path, TFShape shape)
  {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("cannot open " + path);
    size_t colormapBytes = 0, numPoints = 0;
    in >> shape.opacityScale >> shape.valueWindow.x >> shape.valueWindow.y >> colormapBytes;
    in.ignore();
    in.ignore(std::streamsize(colormapBytes)); // binary RGBA colormap
    std::string colormapName;
    std::getline(in, colormapName);
    in >> numPoints;
    shape.valuePoints.resize(numPoints);
    for (vec2f &point : shape.valuePoints)
      in >> point.x >> point.y;
    if (!in || numPoints < 2)
      throw std::runtime_error("cannot read the alpha control points of " + path);
    return shape;
  }

  // RGBA float lookup texture set up like the renderer's (linear, clamped, normalized coordinates)
  struct LutTexture
  {
    int dim, res;
    float opacityScale;
    std::vector<float4> texels;
    cudaArray_t array = nullptr;
    cudaTextureObject_t tex = 0;

    LutTexture(const int dim, const int res, const TFShape &shape) : dim(dim), res(res), opacityScale(shape.opacityScale)
    {
      const int ny = dim > 1 ? res : 1, nz = dim > 2 ? res : 1;
      texels.resize(size_t(res) * ny * nz);
      for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
          for (int x = 0; x < res; x++)
          {
            const float u = (x + 0.5f) / res, v = (y + 0.5f) / ny, w = (z + 0.5f) / nz;
            texels[x + size_t(res) * (y + size_t(ny) * z)] = make_float4(u, v, w, shape.opacity(dim, u, v, w));
          }

      const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
      if (dim == 3)
        CUDA_CHECK(cudaMalloc3DArray(&array, &channelDesc, make_cudaExtent(res, res, res)));
      else
        CUDA_CHECK(cudaMallocArray(&array, &channelDesc, res, ny));
      upload();

      cudaResourceDesc resDesc = {};
      resDesc.resType = cudaResourceTypeArray;
      resDesc.res.array.array = array;
      cudaTextureDesc texDesc = {};
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.addressMode[2] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModeLinear;
      texDesc.readMode = cudaReadModeElementType;
      texDesc.normalizedCoords = 1;
      CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    }
    ~LutTexture()
    {
      cudaDestroyTextureObject(tex);
      cudaFreeArray(array);
    }
    LutTexture(const LutTexture &) = delete;
    LutTexture &operator=(const LutTexture &) = delete;

    uint64_t bytes() const { return texels.size() * sizeof(float4); }

    // what the renderer does on every TF edit
    void upload() const
    {
      const size_t pitch = size_t(res) * sizeof(float4);
      if (dim < 3)
        CUDA_CHECK(cudaMemcpy2DToArray(array, 0, 0, texels.data(), pitch, pitch, dim == 2 ? res : 1, cudaMemcpyHostToDevice));
      else
      {
        cudaMemcpy3DParms copy = {};
        copy.srcPtr = make_cudaPitchedPtr((void *)texels.data(), pitch, res, res);
        copy.dstArray = array;
        copy.extent = make_cudaExtent(res, res, res);
        copy.kind = cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaMemcpy3D(&copy));
      }
    }

    BenchTF deviceTF(const float2 (&domains)[MAX_DIM]) const
    {
      BenchTF tf = {};
      tf.xf = tex;
      for (int a = 0; a < MAX_DIM; a++)
      {
        tf.domain[a] = domains[a];
        tf.numTexels[a] = a < dim ? res : 1;
      }
      tf.opacityScale = opacityScale; // applied a second time, as the renderer does with the widget's scale
      return tf;
    }
  };

  // SM clock, clock event reasons and utilization through NVML; zeros when NVML is unavailable
  class GpuMonitor
  {
public:
    explicit GpuMonitor(const int cudaDevice)
    {
      initialized = nvmlInit_v2() == NVML_SUCCESS;
      char busId[32];
      ok = initialized && cudaDeviceGetPCIBusId(busId, sizeof(busId), cudaDevice) == cudaSuccess &&
           nvmlDeviceGetHandleByPciBusId_v2(busId, &device) == NVML_SUCCESS;
    }
    ~GpuMonitor()
    {
      if (initialized)
        nvmlShutdown();
    }
    GpuMonitor(const GpuMonitor &) = delete;
    GpuMonitor &operator=(const GpuMonitor &) = delete;

    bool available() const { return ok; }

    unsigned smClock() const
    {
      unsigned mhz = 0;
      if (ok)
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &mhz);
      return mhz;
    }

    unsigned long long clockEvents() const
    {
      unsigned long long reasons = 0;
      if (ok)
        nvmlDeviceGetCurrentClocksEventReasons(device, &reasons);
      return reasons;
    }

    unsigned utilization() const
    {
      nvmlUtilization_t rates = {};
      if (ok)
        nvmlDeviceGetUtilizationRates(device, &rates);
      return rates.gpu;
    }

private:
    bool initialized = false, ok = false;
    nvmlDevice_t device = nullptr;
  };

  // names of the nvmlClocksEventReason bits that hold the clock down, GPU idle left out
  static std::string clockEventNames(const unsigned long long reasons)
  {
    static const std::pair<unsigned long long, const char *> names[] = {
        {0x2, "app_clocks"}, {0x4, "sw_power_cap"}, {0x8, "hw_slowdown"}, {0x10, "sync_boost"}, {0x20, "sw_thermal"}, {0x40, "hw_thermal"}, {0x80, "hw_power_brake"}, {0x100, "display_clock"}};
    std::string text;
    for (const auto &[bit, name] : names)
      if (reasons & bit)
        text += (text.empty() ? "" : "|") + std::string(name);
    return text.empty() ? "none" : text;
  }

  static double median(std::vector<double> v)
  {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }

  // one CSV line; NaN fields do not apply to the row's phase and stay empty
  struct Row
  {
    std::string dataset, phase, variant;
    vec3i volume = vec3i(0);
    int tfDim = 0;
    double mc = NAN, tfRes = NAN, tfTexels = NAN;
    std::vector<double> ms;
    double smClockMHz = NAN;
    std::string clockEvents;
    double throughput = NAN;
    std::string throughputUnit;
    double macrocellBytes = NAN, tfBytes = NAN, nonemptyCells = NAN;
    double meanExtent[MAX_DIM] = {NAN, NAN, NAN};
    double meanTexelsPerCell = NAN, maxTexelsPerCell = NAN, texelsSwept = NAN;
    double meanMajorant = NAN, zeroMajorantFraction = NAN;
    double samples = NAN, violations = NAN, textureViolations = NAN;
    double tightnessSamples = NAN, meanExtinction = NAN, acceptance = NAN, maxTightness = NAN;
    double oracleAcceptance = NAN, oracleZeroFraction = NAN;
  };

  static const char *axisNames[MAX_DIM] = {"value", "gradmag", "d2grad"};

  class CsvWriter
  {
public:
    CsvWriter(const std::string &path, std::string gpu, const double busyAtStart)
        : out(path), gpu(std::move(gpu)), busyAtStart(busyAtStart)
    {
      if (!out)
        throw std::runtime_error("cannot write " + path);
      std::replace(this->gpu.begin(), this->gpu.end(), ',', ' ');
      out << "gpu,gpu_busy_at_start_pct,dataset,vol_x,vol_y,vol_z,voxels,tf_dim,tf_axes,mc_per_side,cells,tf_res,tf_texels,"
             "phase,variant,runs,mean_ms,median_ms,min_ms,max_ms,std_ms,sm_clock_mhz,clock_events,throughput,throughput_unit,"
             "macrocell_bytes,tf_bytes,nonempty_cells,mean_extent_value,mean_extent_gradmag,mean_extent_d2grad,"
             "mean_texels_per_cell,max_texels_per_cell,texels_swept,mean_majorant,zero_majorant_fraction,"
             "verify_samples,violations,texture_violations,tightness_samples_per_cell,mean_extinction,acceptance,max_tightness,"
             "oracle_acceptance,oracle_zero_fraction\n";
    }

    void write(const Row &r)
    {
      std::vector<double> ms = r.ms;
      std::sort(ms.begin(), ms.end());
      const bool timed = !ms.empty(); // rows computed for their tightness only have no timing
      const double mean = timed ? std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size() : NAN;
      double variance = 0.0;
      for (const double t : ms)
        variance += (t - mean) * (t - mean);
      const double stddev = ms.size() > 1 ? std::sqrt(variance / (ms.size() - 1)) : timed ? 0.0
                                                                                          : NAN;
      const double med = timed ? median(ms) : NAN;
      const double fastest = timed ? ms.front() : NAN, slowest = timed ? ms.back() : NAN;

      std::string axes;
      for (int a = 0; a < r.tfDim; a++)
        axes += (a ? "+" : "") + std::string(axisNames[a]);
      const double cells = std::isnan(r.mc) ? NAN : r.mc * r.mc * r.mc;
      const double voxels = double(r.volume.x) * r.volume.y * r.volume.z;

      out << gpu << ',' << num(busyAtStart, 0) << ',' << r.dataset << ',' << r.volume.x << ',' << r.volume.y << ',' << r.volume.z << ','
          << num(voxels, 0) << ',' << r.tfDim << ',' << axes << ',' << num(r.mc, 0) << ',' << num(cells, 0) << ','
          << num(r.tfRes, 0) << ',' << num(r.tfTexels, 0) << ',' << r.phase << ',' << r.variant << ','
          << ms.size() << ',' << num(mean, 4) << ',' << num(med, 4) << ',' << num(fastest, 4) << ','
          << num(slowest, 4) << ',' << num(stddev, 4) << ',' << num(r.smClockMHz, 0) << ',' << r.clockEvents << ','
          << num(r.throughput, 4) << ',' << r.throughputUnit << ','
          << num(r.macrocellBytes, 0) << ',' << num(r.tfBytes, 0) << ',' << num(r.nonemptyCells, 0) << ','
          << num(r.meanExtent[0], 3) << ',' << num(r.meanExtent[1], 3) << ',' << num(r.meanExtent[2], 3) << ','
          << num(r.meanTexelsPerCell, 3) << ',' << num(r.maxTexelsPerCell, 0) << ',' << num(r.texelsSwept, 0) << ','
          << num(r.meanMajorant, 6) << ',' << num(r.zeroMajorantFraction, 6) << ',' << num(r.samples, 0) << ','
          << num(r.violations, 0) << ',' << num(r.textureViolations, 0) << ',' << num(r.tightnessSamples, 0) << ',' << num(r.meanExtinction, 6) << ','
          << num(r.acceptance, 6) << ',' << num(r.maxTightness, 6) << ',' << num(r.oracleAcceptance, 6) << ','
          << num(r.oracleZeroFraction, 6) << '\n';
      out.flush();

      printf("  %-9s %-10s %dD", r.phase.c_str(), r.variant.c_str(), r.tfDim);
      printf(std::isnan(r.mc) ? "         " : "  mc %-4.0f", r.mc);
      printf(std::isnan(r.tfRes) ? "          " : "  res %-4.0f", r.tfRes);
      if (timed)
        printf("  median %10.3f ms  [%.3f, %.3f]", med, fastest, slowest);
      else
        printf("  %-38s", "not timed");
      if (!std::isnan(r.smClockMHz))
        printf("  %4.0f MHz", r.smClockMHz);
      if (!r.clockEvents.empty() && r.clockEvents != "none")
        printf(" (%s)", r.clockEvents.c_str());
      if (!std::isnan(r.throughput))
        printf("  %8.3f %s", r.throughput, r.throughputUnit.c_str());
      if (!std::isnan(r.violations))
        printf("  violations %.0f/%.0f", r.violations, r.samples);
      if (!std::isnan(r.textureViolations))
        printf(" (renderer sampling %.0f)", r.textureViolations);
      if (!std::isnan(r.acceptance))
        printf("  acceptance %.3f (bound %.3f x cells %.3f)", r.acceptance, r.maxTightness, r.oracleAcceptance);
      printf("\n");
    }

private:
    std::ofstream out;
    std::string gpu;
    double busyAtStart;

    static std::string num(const double v, const int decimals)
    {
      if (std::isnan(v))
        return "";
      char buffer[64];
      snprintf(buffer, sizeof(buffer), "%.*f", decimals, v);
      return buffer;
    }
  };

  struct Options
  {
    std::vector<int> macrocells, resolutions;
    int warmup, runs;
    uint64_t samples;
    double tightnessPerVoxel, derivativePercentile;
    const GpuMonitor *gpu;
  };

  // times runs of work with CUDA events, sampling the SM clock while each run is in flight
  template <typename Work>
  static void measure(const Options &opt, Row &row, Work &&work)
  {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < opt.warmup; i++)
      work();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    double clockSum = 0.0;
    unsigned long long events = 0;
    row.ms.clear();
    for (int i = 0; i < opt.runs; i++)
    {
      CUDA_CHECK(cudaEventRecord(start));
      work();
      CUDA_CHECK(cudaEventRecord(stop));
      clockSum += opt.gpu->smClock();
      events |= opt.gpu->clockEvents();
      CUDA_CHECK(cudaEventSynchronize(stop));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
      row.ms.push_back(t);
    }
    CUDA_CHECK(cudaGetLastError());
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (opt.gpu->available())
    {
      row.smClockMHz = clockSum / opt.runs;
      row.clockEvents = clockEventNames(events);
    }
  }

  template <int D>
  static void verify(const Volume &vol, const vec3i &mcDims, const MacrocellND<D> *d_cells, const BenchTF &tf,
                     const float *d_majorants, const uint64_t samples, Row &row)
  {
    if (samples == 0)
      return;
    DeviceArray<unsigned long long> count(2);
    CUDA_CHECK(cudaMemset(count.ptr, 0, 2 * sizeof(unsigned long long)));
    const unsigned blocks = unsigned(divRoundUp(samples, uint64_t(256)));
    verifySamples<D, false><<<blocks, 256>>>(samples, vol.scalars->ptr, 0, vol.dims, mcDims, d_cells, tf, d_majorants,
                                             count.ptr);
    if constexpr (D <= 2)
      if (vol.texture)
        verifySamples<D, true><<<blocks, 256>>>(samples, vol.scalars->ptr, vol.texture->tex, vol.dims, mcDims, d_cells,
                                                tf, d_majorants, count.ptr + 1);
    CUDA_CHECK(cudaGetLastError());
    const std::vector<unsigned long long> counts = count.download();
    row.samples = double(samples);
    row.violations = double(counts[0]);
    if (D <= 2 && vol.texture)
      row.textureViolations = double(counts[1]);
  }

  template <int D>
  static double countNonempty(const std::vector<MacrocellND<D>> &cells)
  {
    return double(std::count_if(cells.begin(), cells.end(), [](const MacrocellND<D> &c)
                                { return c.range[0].x <= c.range[0].y; }));
  }

  // texels computeMajorantsND sweeps per macrocell, from the same texelRange
  template <int D>
  static void sweepStatistics(const std::vector<MacrocellND<D>> &cells, const BenchTF &tf, Row &row)
  {
    double extentSum[MAX_DIM] = {}, total = 0.0, largest = 0.0, nonempty = 0.0;
    for (const MacrocellND<D> &cell : cells)
    {
      bool empty = false;
      double texels = 1.0;
      int extent[D];
      for (int a = 0; a < D; a++)
      {
        empty |= cell.range[a].x > cell.range[a].y;
        const int2 range = texelRange(cell.range[a], tf.domain[a], tf.numTexels[a]);
        extent[a] = range.y - range.x + 1;
        texels *= extent[a];
      }
      if (empty)
        continue;
      nonempty += 1.0;
      total += texels;
      largest = std::max(largest, texels);
      for (int a = 0; a < D; a++)
        extentSum[a] += extent[a];
    }
    for (int a = 0; a < D; a++)
      row.meanExtent[a] = nonempty > 0.0 ? extentSum[a] / nonempty : 0.0;
    row.meanTexelsPerCell = nonempty > 0.0 ? total / nonempty : 0.0;
    row.maxTexelsPerCell = largest;
    row.texelsSwept = total;
  }

  static void majorantStatistics(const std::vector<float> &majorants, Row &row)
  {
    double sum = 0.0, zeros = 0.0;
    for (const float m : majorants)
    {
      sum += m;
      zeros += m == 0.f;
    }
    row.meanMajorant = sum / majorants.size();
    row.zeroMajorantFraction = zeros / majorants.size();
  }

  // delta-tracking efficiency of the majorants against each cell's sampled extinction (cells have equal volume)
  static void tightness(const std::vector<float> &majorants, const std::vector<float2> &extinction,
                        const uint32_t samplesPerCell, Row &row)
  {
    if (extinction.empty())
      return;
    double majorantSum = 0.0, meanSum = 0.0, maxSum = 0.0, oracleZero = 0.0, exceeded = 0.0;
    for (size_t c = 0; c < majorants.size(); c++)
    {
      majorantSum += majorants[c];
      meanSum += extinction[c].x;
      maxSum += extinction[c].y;
      oracleZero += extinction[c].y == 0.f;
      exceeded += extinction[c].y > majorants[c] * (1.f + 1e-5f) + 1e-6f;
    }
    const double numCells = double(majorants.size());
    row.tightnessSamples = samplesPerCell;
    row.meanExtinction = meanSum / numCells;
    row.acceptance = majorantSum > 0.0 ? meanSum / majorantSum : NAN;  // real / tentative collisions
    row.maxTightness = majorantSum > 0.0 ? maxSum / majorantSum : NAN; // slack of the range-based bound
    row.oracleAcceptance = maxSum > 0.0 ? meanSum / maxSum : NAN;      // best possible at this cell size
    row.oracleZeroFraction = oracleZero / numCells;                    // cells a perfect majorant would skip
    if (exceeded > 0.0)
      printf("  warning: %.0f cells have samples above their majorant\n", exceeded);
  }

  template <int D>
  static void benchmarkDim(const Volume &vol, const TFShape &shape, const float2 (&domains)[MAX_DIM], const Options &opt,
                           CsvWriter &csv)
  {
    const box3f unitBox(vec3f(0.f), vec3f(1.f)); // the voxel-to-cell mapping does not depend on the world bounds
    const uint64_t numVoxels = vol.numVoxels();
    const uint64_t rasterBlocks = divRoundUp(numVoxels, uint64_t(32));
    const dim3 rasterGrid(unsigned(std::min(rasterBlocks, uint64_t(MAX_GRID_SIZE))),
                          unsigned(divRoundUp(rasterBlocks, uint64_t(MAX_GRID_SIZE))), 1);
    if (rasterGrid.y > unsigned(MAX_GRID_SIZE))
      throw std::runtime_error("volume too large for the macrocell raster launch");

    auto newRow = [&](const char *phase, const char *variant)
    {
      Row row;
      row.dataset = vol.name;
      row.volume = vol.dims;
      row.tfDim = D;
      row.phase = phase;
      row.variant = variant;
      return row;
    };

    std::vector<std::unique_ptr<LutTexture>> luts;
    for (const int res : opt.resolutions)
    {
      luts.push_back(std::make_unique<LutTexture>(D, res, shape));
      const LutTexture &lut = *luts.back();
      Row row = newRow("tf_upload", "float4");
      row.tfRes = res;
      row.tfTexels = double(lut.texels.size());
      row.tfBytes = double(lut.bytes());
      measure(opt, row, [&]
              { lut.upload(); });
      row.throughput = lut.bytes() / (median(row.ms) * 1e6);
      row.throughputUnit = "GB/s";
      csv.write(row);
    }

    for (const int mc : opt.macrocells)
    {
      const vec3i mcDims(mc);
      const uint64_t numCells = uint64_t(mc) * uint64_t(mc) * uint64_t(mc);
      DeviceArray<MacrocellND<D>> cells(numCells);
      DeviceArray<float> majorants(numCells);
      DeviceArray<float2> legacyCells(D == 1 ? numCells : 0);
      // ground-truth samples per cell: tightnessPerVoxel per voxel, in whole warps, at least one warp
      const double perCell = std::ceil(opt.tightnessPerVoxel * double(numVoxels) / double(numCells) / 32.0) * 32.0;
      const uint32_t samplesPerCell = opt.tightnessPerVoxel > 0.0 ? uint32_t(std::clamp(perCell, 32.0, 65536.0)) : 0u;

      Row build = newRow("build", "nd");
      build.mc = mc;
      build.macrocellBytes = double(numCells * sizeof(MacrocellND<D>));
      measure(opt, build, [&]
              {
        clearMacrocellsND<D><<<unsigned(divRoundUp(numCells, uint64_t(256))), 256>>>(cells.ptr, numCells);
        rasterMacrocellsND<D><<<rasterGrid, 32>>>(cells.ptr, mcDims, vol.scalars->ptr, vol.dims, 0, 1); });
      build.throughput = numVoxels / (median(build.ms) * 1e6);
      build.throughputUnit = "Gvoxel/s";
      const std::vector<MacrocellND<D>> hostCells = cells.download();
      build.nonemptyCells = countNonempty(hostCells);
      verify<D>(vol, mcDims, cells.ptr, BenchTF{}, nullptr, opt.samples, build);
      csv.write(build);

      if constexpr (D > 1) // the same build with voxelGradientIndexed
      {
        DeviceArray<MacrocellND<D>> indexedCells(numCells);
        Row indexed = newRow("build", "nd_intgrad");
        indexed.mc = mc;
        indexed.macrocellBytes = build.macrocellBytes;
        measure(opt, indexed, [&]
                {
          clearMacrocellsND<D><<<unsigned(divRoundUp(numCells, uint64_t(256))), 256>>>(indexedCells.ptr, numCells);
          rasterMacrocellsND<D, true><<<rasterGrid, 32>>>(indexedCells.ptr, mcDims, vol.scalars->ptr, vol.dims, 0, 1); });
        indexed.throughput = numVoxels / (median(indexed.ms) * 1e6);
        indexed.throughputUnit = "Gvoxel/s";
        indexed.nonemptyCells = build.nonemptyCells;
        if (std::memcmp(indexedCells.download().data(), hostCells.data(), numCells * sizeof(MacrocellND<D>)) == 0)
        {
          indexed.samples = build.samples;
          indexed.violations = build.violations;
          indexed.textureViolations = build.textureViolations;
        }
        else
        {
          printf("  warning: nd_intgrad and nd macrocells differ\n");
          verify<D>(vol, mcDims, indexedCells.ptr, BenchTF{}, nullptr, opt.samples, indexed);
        }
        csv.write(indexed);
      }

      if constexpr (D == 1) // the renderer's 1D path: float2 macrocells, thread-per-cell majorants
      {
        Row legacy = newRow("build", "legacy1d");
        legacy.mc = mc;
        legacy.macrocellBytes = double(numCells * sizeof(float2));
        const vec3i sizeBlock(4);
        measure(opt, legacy, [&]
                {
          sizeMCs<<<to_dims(divRoundUp(mcDims, sizeBlock)), to_dims(sizeBlock)>>>(legacyCells.ptr, mcDims, unitBox, 1);
          rasterElements<<<rasterGrid, 32>>>(legacyCells.ptr, mcDims, vol.scalars->ptr, vol.dims, 0, 1); });
        legacy.throughput = numVoxels / (median(legacy.ms) * 1e6);
        legacy.throughputUnit = "Gvoxel/s";
        const std::vector<float2> hostLegacy = legacyCells.download();
        legacy.nonemptyCells = build.nonemptyCells;
        verify<1>(vol, mcDims, (const MacrocellND<1> *)legacyCells.ptr, BenchTF{}, nullptr, opt.samples, legacy);
        csv.write(legacy);
        if (std::memcmp(hostLegacy.data(), hostCells.data(), numCells * sizeof(float2)) != 0)
          printf("  warning: legacy1d and nd macrocells differ\n");
      }

      for (const auto &lut : luts)
      {
        const BenchTF tf = lut->deviceTF(domains);
        DeviceArray<BenchTF> d_tf(1);
        d_tf.upload(&tf);

        std::vector<float2> extinction; // ground truth the majorants of this TF are compared against
        if (samplesPerCell > 0)
        {
          DeviceArray<float2> d_extinction(numCells);
          cellExtinction<D><<<unsigned(divRoundUp(numCells * 32, uint64_t(256))), 256>>>(
              numCells, mcDims, samplesPerCell, vol.scalars->ptr, vol.dims, tf, d_extinction.ptr);
          CUDA_CHECK(cudaGetLastError());
          extinction = d_extinction.download();
        }

        Row row = newRow("majorant", "warp");
        row.mc = mc;
        row.tfRes = lut->res;
        row.tfTexels = double(lut->texels.size());
        row.tfBytes = double(lut->bytes());
        row.nonemptyCells = build.nonemptyCells;
        measure(opt, row, [&]
                {
          cudaMemsetAsync(majorants.ptr, 0, numCells * sizeof(float));
          computeMajorantsND<D><<<unsigned(divRoundUp(numCells * 32, uint64_t(256))), 256>>>(
              numCells, cells.ptr, d_tf.ptr, 1, true, majorants.ptr); });
        sweepStatistics(hostCells, tf, row);
        const std::vector<float> hostMajorants = majorants.download();
        majorantStatistics(hostMajorants, row);
        tightness(hostMajorants, extinction, samplesPerCell, row);
        row.throughput = row.texelsSwept / (median(row.ms) * 1e6);
        row.throughputUnit = "Gtexel/s";
        verify<D>(vol, mcDims, cells.ptr, tf, majorants.ptr, opt.samples, row);
        csv.write(row);

        if constexpr (D > 1) // the same TF with only the value range per cell, as 1D macrocells would give
        {
          std::vector<MacrocellND<D>> valueOnly = hostCells;
          for (MacrocellND<D> &cell : valueOnly)
            if (cell.range[0].x <= cell.range[0].y)
              for (int a = 1; a < D; a++)
                cell.range[a] = tf.domain[a];
          DeviceArray<MacrocellND<D>> d_valueOnly(numCells);
          d_valueOnly.upload(valueOnly.data());
          CUDA_CHECK(cudaMemset(majorants.ptr, 0, numCells * sizeof(float)));
          computeMajorantsND<D><<<unsigned(divRoundUp(numCells * 32, uint64_t(256))), 256>>>(
              numCells, d_valueOnly.ptr, d_tf.ptr, 1, true, majorants.ptr);
          CUDA_CHECK(cudaGetLastError());

          Row projected = newRow("majorant", "value_only");
          projected.mc = mc;
          projected.tfRes = lut->res;
          projected.tfTexels = double(lut->texels.size());
          projected.tfBytes = double(lut->bytes());
          projected.nonemptyCells = build.nonemptyCells;
          sweepStatistics(valueOnly, tf, projected);
          const std::vector<float> projectedMajorants = majorants.download();
          majorantStatistics(projectedMajorants, projected);
          tightness(projectedMajorants, extinction, samplesPerCell, projected);
          verify<D>(vol, mcDims, d_valueOnly.ptr, tf, majorants.ptr, opt.samples, projected);
          csv.write(projected);
        }

        if constexpr (D == 1)
        {
          LegacyTF legacyTF = {};
          legacyTF.colorMapTexture = lut->tex;
          legacyTF.volDomain = {domains[0].x, domains[0].y};
          legacyTF.xfDomain = {0.f, 1.f};
          legacyTF.numTexels = lut->res;
          legacyTF.opacityScale = lut->opacityScale;
          DeviceArray<LegacyTF> d_legacyTF(1);
          d_legacyTF.upload(&legacyTF);

          Row legacy = newRow("majorant", "legacy1d");
          legacy.mc = mc;
          legacy.tfRes = lut->res;
          legacy.tfTexels = double(lut->texels.size());
          legacy.tfBytes = double(lut->bytes());
          legacy.nonemptyCells = build.nonemptyCells;
          measure(opt, legacy, [&]
                  { _recalculateDensityRangesMM<<<unsigned(divRoundUp(numCells, uint64_t(32))), 32>>>(
                        int(numCells), legacyCells.ptr, d_legacyTF.ptr, 1, majorants.ptr); });
          const std::vector<float> legacyMajorants = majorants.download();
          majorantStatistics(legacyMajorants, legacy);
          tightness(legacyMajorants, extinction, samplesPerCell, legacy);
          verify<1>(vol, mcDims, (const MacrocellND<1> *)legacyCells.ptr, tf, majorants.ptr, opt.samples, legacy);
          csv.write(legacy);
        }
      }
    }
  }

  static void benchmarkVolume(const Volume &vol, const TFShape &shape, const std::vector<int> &tfDims, const Options &opt,
                              CsvWriter &csv)
  {
    const float span = vol.valueRange.y - vol.valueRange.x;
    const float2 valueDomain = make_float2(vol.valueRange.x + shape.valueWindow.x * span,
                                           vol.valueRange.x + shape.valueWindow.y * span);

    // derivative axes span a percentile of the voxels inside the value window, at most 2^24 of them as the viewer samples
    const uint64_t stride = std::max<uint64_t>(1, divRoundUp(vol.numVoxels(), uint64_t(1) << 24));
    const uint64_t count = divRoundUp(vol.numVoxels(), stride);
    DeviceArray<float3> d_samples(count);
    voxelAttributeSamples<<<unsigned(divRoundUp(count, uint64_t(256))), 256>>>(vol.scalars->ptr, vol.dims, stride, count,
                                                                               d_samples.ptr);
    CUDA_CHECK(cudaGetLastError());
    const std::vector<float3> samples = d_samples.download();
    std::vector<float> gradients, seconds;
    for (const float3 &s : samples)
      if (s.x >= valueDomain.x && s.x <= valueDomain.y)
      {
        gradients.push_back(s.y);
        seconds.push_back(s.z);
      }
    const bool inWindow = !gradients.empty();
    if (!inWindow)
      for (const float3 &s : samples)
      {
        gradients.push_back(s.y);
        seconds.push_back(s.z);
      }
    auto percentile = [&](std::vector<float> &v)
    {
      const size_t i = std::min(v.size() - 1, size_t(opt.derivativePercentile / 100.0 * double(v.size() - 1) + 0.5));
      std::nth_element(v.begin(), v.begin() + i, v.end());
      return v[i] > 0.f ? v[i] : 1.f;
    };
    const float gradientTop = percentile(gradients), secondTop = percentile(seconds);
    const float2 domains[MAX_DIM] = {valueDomain, make_float2(0.f, gradientTop), make_float2(-secondTop, secondTop)};

    printf("\n%s: %d x %d x %d voxels, value [%g, %g]\n", vol.name.c_str(), vol.dims.x, vol.dims.y, vol.dims.z,
           vol.valueRange.x, vol.valueRange.y);
    printf("derivative axes: P%g of %zu sampled voxels %s: |f'| %g, |f''| %g\n", opt.derivativePercentile,
           gradients.size(), inWindow ? "in the value window" : "(none in the value window)", gradientTop, secondTop);
    if (shape.valuePoints.empty())
      printf("value TF: smoothstep %g..%g of [%g, %g]\n", shape.valueRamp.x, shape.valueRamp.y, valueDomain.x, valueDomain.y);
    else
      printf("value TF: %zu control points over [%g, %g], opacity scale %g\n", shape.valuePoints.size(), valueDomain.x,
             valueDomain.y, shape.opacityScale);
    printf("|f'| ramp %g..%g of [0, %g], f'' peak width %g of [%g, %g]\n", shape.gradientRamp.x, shape.gradientRamp.y,
           gradientTop, shape.d2Width, -secondTop, secondTop);
    for (const int d : tfDims)
    {
      if (d == 1)
        benchmarkDim<1>(vol, shape, domains, opt, csv);
      else if (d == 2)
        benchmarkDim<2>(vol, shape, domains, opt, csv);
      else
        benchmarkDim<3>(vol, shape, domains, opt, csv);
    }
  }

} // namespace tfbench

int main(int argc, char *argv[])
{
  using namespace tfbench;
  argparse::ArgumentParser program("mdwtTFBench");
  program.add_argument("-fr", "--raw-data")
      .help("raw volume(s) named <name>_<X>x<Y>x<Z>_<type>.raw, as for the viewer")
      .nargs(argparse::nargs_pattern::any);
  program.add_argument("--synthetic")
      .help("edge length of a generated test volume, used without -fr")
      .default_value(256)
      .scan<'i', int>();
  program.add_argument("-d", "--tf-dims")
      .help("TF dimensionalities: 1 = value, 2 = + gradient magnitude, 3 = + second derivative along the gradient")
      .nargs(argparse::nargs_pattern::at_least_one)
      .default_value(std::vector<int>{1, 2, 3})
      .scan<'i', int>();
  program.add_argument("-r", "--tf-res")
      .help("lookup texels per TF axis")
      .nargs(argparse::nargs_pattern::at_least_one)
      .default_value(std::vector<int>{64, 128, 256})
      .scan<'i', int>();
  program.add_argument("-mc", "--macrocells")
      .help("macrocells per side")
      .nargs(argparse::nargs_pattern::at_least_one)
      .default_value(std::vector<int>{16, 32, 64, 128})
      .scan<'i', int>();
  program.add_argument("--warmup")
      .help("untimed runs before each measurement")
      .default_value(1)
      .scan<'i', int>();
  program.add_argument("--runs")
      .help("timed runs per measurement")
      .default_value(5)
      .scan<'i', int>();
  program.add_argument("--verify-samples")
      .help("random samples checked against the macrocell ranges and majorants, 0 = off")
      .default_value(1 << 20)
      .scan<'i', int>();
  program.add_argument("-t", "--transfer-function")
      .help("1D widget state(s) (.tfn) for the value axis, one per -fr file as for the viewer")
      .nargs(argparse::nargs_pattern::any);
  program.add_argument("--value-ramp")
      .help("value smoothstep edges without a .tfn, as fractions of the data range")
      .nargs(2)
      .default_value(std::vector<float>{0.05f, 0.3f})
      .scan<'g', float>();
  program.add_argument("--gradient-ramp")
      .help("|f'| smoothstep edges of the 2D and 3D TFs, as fractions of the largest voxel |f'|")
      .nargs(2)
      .default_value(std::vector<float>{0.02f, 0.2f})
      .scan<'g', float>();
  program.add_argument("--derivative-percentile")
      .help("|f'| and |f''| axes end at this percentile of the voxels in the value window; 100 = the max, as the viewer")
      .default_value(99.0f)
      .scan<'g', float>();
  program.add_argument("--d2-width")
      .help("width of the 3D TF's peak at f'' = 0, as a fraction of the f'' axis")
      .default_value(0.1f)
      .scan<'g', float>();
  program.add_argument("--tightness-samples")
      .help("ground-truth samples per voxel for the majorant tightness, at least 32 per macrocell, 0 = off")
      .default_value(1.0f)
      .scan<'g', float>();
  program.add_argument("-o", "--csv")
      .help("output file")
      .default_value(std::string("tf_benchmark.csv"));

  try
  {
    program.parse_args(argc, argv);

    Options opt;
    opt.macrocells = program.get<std::vector<int>>("-mc");
    opt.resolutions = program.get<std::vector<int>>("-r");
    opt.warmup = std::max(0, program.get<int>("--warmup"));
    opt.runs = program.get<int>("--runs");
    opt.samples = uint64_t(std::max(0, program.get<int>("--verify-samples")));
    opt.tightnessPerVoxel = std::max(0.f, program.get<float>("--tightness-samples"));
    opt.derivativePercentile = std::clamp(program.get<float>("--derivative-percentile"), 0.f, 100.f);
    const std::vector<int> tfDims = program.get<std::vector<int>>("-d");

    TFShape baseShape;
    const std::vector<float> valueRamp = program.get<std::vector<float>>("--value-ramp");
    const std::vector<float> gradientRamp = program.get<std::vector<float>>("--gradient-ramp");
    baseShape.valueRamp = make_float2(valueRamp[0], valueRamp[1]);
    baseShape.gradientRamp = make_float2(gradientRamp[0], gradientRamp[1]);
    baseShape.d2Width = program.get<float>("--d2-width");
    if (!(valueRamp[0] < valueRamp[1]) || !(gradientRamp[0] < gradientRamp[1]) || !(baseShape.d2Width > 0.f))
      throw std::runtime_error("ramps need lower < upper edges and --d2-width must be positive");
    const std::vector<std::string> tfns =
        program.is_used("-t") ? program.get<std::vector<std::string>>("-t") : std::vector<std::string>{};
    auto shapeFor = [&](const size_t i)
    { return i < tfns.size() ? loadTfn(tfns[i], baseShape) : baseShape; };
    if (opt.runs < 1)
      throw std::runtime_error("--runs must be at least 1");
    for (const int d : tfDims)
      if (d < 1 || d > MAX_DIM)
        throw std::runtime_error("--tf-dims takes 1, 2 or 3");
    for (const int r : opt.resolutions)
      if (r < 2 || r > 2048)
        throw std::runtime_error("--tf-res takes 2 to 2048 texels");
    for (const int mc : opt.macrocells)
      if (mc < 1)
        throw std::runtime_error("--macrocells must be positive");

    const GpuMonitor gpu(0);
    opt.gpu = &gpu;
    double busyAtStart = NAN;
    if (gpu.available())
    {
      unsigned peak = 0;
      for (int i = 0; i < 10; i++) // before this process runs any work
      {
        peak = std::max(peak, gpu.utilization());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      busyAtStart = peak;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const std::string csvPath = program.get<std::string>("-o");
    CsvWriter csv(csvPath, prop.name, busyAtStart);
    printf("%s, %d warmup + %d timed runs per measurement\n", prop.name, opt.warmup, opt.runs);
    if (!gpu.available())
      printf("NVML unavailable: no clock and utilization columns\n");
    else if (busyAtStart > 5.0)
      printf("warning: the GPU was %.0f%% busy before the benchmark started; another process shares it and timings will vary\n",
             busyAtStart);

    if (program.is_used("-fr"))
    {
      const std::vector<std::string> paths = program.get<std::vector<std::string>>("-fr");
      for (size_t i = 0; i < paths.size(); i++)
        benchmarkVolume(loadRaw(paths[i]), shapeFor(i), tfDims, opt, csv);
    }
    else
      benchmarkVolume(syntheticVolume(program.get<int>("--synthetic")), shapeFor(0), tfDims, opt, csv);

    printf("\nwrote %s\n", std::filesystem::absolute(csvPath).string().c_str());
  }
  catch (const std::exception &e)
  {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}

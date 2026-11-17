// source:  https://gitlab.com/n8vm/owlptdvr/-/blob/master/renderer.cu?ref_type=heads
#include "renderer.h"
#include "deviceUtils.h"
#include "mathHelper.h"

#include "cuda_fp16.h"

#define CUDA_SYNC_CHECK()                                  \
  {                                                        \
    cudaDeviceSynchronize();                               \
    cudaError_t rc = cudaGetLastError();                   \
    if (rc != cudaSuccess)                                 \
    {                                                      \
      fprintf(stderr, "error (%s: line %d): %s\n",         \
              __FILE__, __LINE__, cudaGetErrorString(rc)); \
      throw std::runtime_error("fatal cuda error");        \
    }                                                      \
  }

__global__ void _recalculateDensityRanges(
    int numPrims, bool is_background, owl::box4f *bboxes, // const uint8_t* nvdbData,
    dtracker::TFData *tf, size_t numChannels,
    float *maxima)
{
  int primID = (blockIdx.x * blockDim.x + threadIdx.x);
  if (primID >= numPrims)
    return;

  // printf("primID %d level %d \n", primID, level);
  // if (level == -1)
  // printf("PrimID %d domain %f %f addr1 %f addr2 %f addrMin %d addrMax %d min %f max %f\n", primID, domain.x, domain.y, addr1, addr2, addrMin, addrMax,  minDensity, maxDensity);

  float mx = 0.f, mn = 0.f;
  if (!is_background)
  {
    mn = bboxes[primID].lower.w;
    mx = bboxes[primID].upper.w;
  }
  float maxDensity = 0.f;
  for (size_t tfID = 0; tfID < numChannels; tfID++)
  {
    // empty box
    if (mx < mn)
    {
      continue;
    }

    // transform data min max to transfer function space
    float remappedMin1 = (mn - tf[tfID].volDomain.lo) / (tf[tfID].volDomain.hi - tf[tfID].volDomain.lo);
    float remappedMin = (remappedMin1 - tf[tfID].xfDomain.lo) / (tf[tfID].xfDomain.hi - tf[tfID].xfDomain.lo);
    float remappedMax1 = (mx - tf[tfID].volDomain.lo) / (tf[tfID].volDomain.hi - tf[tfID].volDomain.lo);
    float remappedMax = (remappedMax1 - tf[tfID].xfDomain.lo) / (tf[tfID].xfDomain.hi - tf[tfID].xfDomain.lo);
    float addr1 = remappedMin * tf[tfID].numTexels;
    float addr2 = remappedMax * tf[tfID].numTexels;

    int addrMin = min(max(int(min(floor(addr1), floor(addr2))), 0), tf[tfID].numTexels - 1);
    int addrMax = min(max(int(max(ceil(addr1), ceil(addr2))), 0), tf[tfID].numTexels - 1);

    float maxDensityForVolume;
    for (int i = addrMin; i <= addrMax; ++i)
    {
      float density = tex2D<float4>(tf[tfID].colorMapTexture, float(i) / tf[tfID].numTexels, 0.5f).w * tf[tfID].opacityScale;
      if (i == addrMin)
        maxDensityForVolume = density;
      else
        maxDensityForVolume = max(maxDensityForVolume, density);
    }
    maxDensity += maxDensityForVolume;
  }
  maxima[primID] = maxDensity / (float)numChannels;
}

__global__ void _recalculateDensityRanges(
    int numPrims, const float2 *macrocells, // const uint8_t* nvdbData,
    const dtracker::TFData *tf, size_t numChannels, float *maxima)
{
  int nodeID = (blockIdx.x * blockDim.x + threadIdx.x);
  if (nodeID >= numPrims)
    return;

  float maxDensity = 0.f;
  maxima[nodeID] = 0.f;
  for (size_t tfID = 0; tfID < numChannels; tfID++)
  {
    float mn = macrocells[nodeID * numChannels + tfID].x;
    float mx = macrocells[nodeID * numChannels + tfID].y;

    // empty box
    if (mx < mn)
    {
      // maxima[nodeID] = 0.f;
      continue; // return;
    }

    // transform data min max to transfer function space
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
    maxDensity += maxDensityForVolume;
  }
  maxima[nodeID] = maxDensity;
}

__global__ void _recalculateDensityRangesMM(
    int numPrims, const float2 *macrocells,
    const dtracker::TFData *tf, size_t numChannels, float *maxima)
{
  int nodeID = (blockIdx.x * blockDim.x + threadIdx.x);
  if (nodeID >= numPrims)
    return;

  for (size_t tfID = 0; tfID < numChannels; tfID++)
  {
    maxima[nodeID * numChannels + tfID] = 0.f;
    float mn = macrocells[nodeID * numChannels + tfID].x;
    float mx = macrocells[nodeID * numChannels + tfID].y;

    // empty box
    if (mx < mn)
    {
      // maxima[nodeID] = 0.f;
      continue; // return;
    }

    // transform data min max to transfer function space
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

namespace dtracker
{
  void Renderer::RecalculateDensityRanges()
  {
    if (tfDim > 1)
    {
      RecalculateDensityRangesND();
      return;
    }
    size_t numChannels = meshType != MeshType::UNDEFINED ? (meshType == MeshType::UMESH ? umeshPtrs.size() : rawPtrs.size()) : 0;
    if (numChannels == 0)
      throw std::runtime_error("No mesh data found");

    // Create a dvice buffer of TFData
    OWLBuffer tfdataBuffer = owlDeviceBufferCreate(context,
                                                   OWL_USER_TYPE(dtracker::TFData),
                                                   numChannels, tfdatas.data());
    dtracker::TFData *d_tfdatas = (dtracker::TFData *)owlBufferGetPointer(tfdataBuffer, 0);

    float2 volumeDomain = {tfdatas[0].volDomain.lower, tfdatas[0].volDomain.upper};
    float2 tfnDomain = {tfdatas[0].xfDomain.lower, tfdatas[0].xfDomain.upper};
    float opacityScale = this->tfdatas[0].opacityScale;
    cudaTextureObject_t colorMapTexture = this->tfdatas[0].colorMapTexture;
    int colorMapSize = this->tfdatas[0].colorMap.size();
    dim3 blockSize(32);
    uint32_t numThreads;
    bool isBackground;
    dim3 gridSize;
    owl::box4f *bboxes;
    float *majorantBuffer;
    // root
    {
      bboxes = (owl::box4f *)owlBufferGetPointer(rootBBoxBuffer, 0);
      isBackground = false;
      numThreads = 1;
      gridSize = dim3((numThreads + blockSize.x - 1) / blockSize.x);
      majorantBuffer = (float *)owlBufferGetPointer(rootMaximaBuffer, 0);
      _recalculateDensityRanges<<<gridSize, blockSize>>>(
          numThreads, isBackground, bboxes,
          d_tfdatas, numChannels, majorantBuffer);

      CUDA_SYNC_CHECK();

      owlGroupBuildAccel(rootMacrocellBLAS);
      owlGroupBuildAccel(rootMacrocellTLAS);
    }
    {
      float2 *macrocells = (float2 *)owlBufferGetPointer(macrocellsBuffer, 0);
      numThreads = macrocellDims.x * macrocellDims.y * macrocellDims.z;
      gridSize = dim3((numThreads + blockSize.x - 1) / blockSize.x);
      majorantBuffer = (float *)owlBufferGetPointer(gridMaximaBuffer, 0);

      // Calculate majorants
      if (mode <= Mode::MULTI_ALT)
        _recalculateDensityRangesMM<<<gridSize, blockSize>>>(
            numThreads, macrocells, d_tfdatas, numChannels,
            majorantBuffer);
      else if (mode <= Mode::MIX)
        _recalculateDensityRanges<<<gridSize, blockSize>>>(
            numThreads, macrocells, d_tfdatas, numChannels,
            majorantBuffer);
    }
    // {
    //   bboxes = (owl::box4f*)owlBufferGetPointer(clusterBBoxBuffer, 0);
    //   isBackground = false;
    //   numThreads = numClusters;
    //   gridSize = dim3 ((numThreads + blockSize.x - 1) / blockSize.x);
    //   majorantBuffer = (float*)owlBufferGetPointer(clusterMaximaBuffer, 0);
    //   _recalculateDensityRanges<<<gridSize,blockSize>>>(
    //     numThreads, /*lvl*/ isBackground, bboxes,
    //     colorMapTexture, colorMapSize, volumeDomain, tfnDomain, opacityScale,
    //     majorantBuffer);
    // }

    CUDA_SYNC_CHECK();
  }

  /*! max macrocell grid size, in cells */
  constexpr int MAX_GRID_SIZE = 1024 * 1024;

  __global__ void sizeMCs(box4f *d_mcGrid, const vec3i dims, const box3f worldBounds)
  {
    const vec3i cellIdx = vec3i(threadIdx) + vec3i(blockIdx) * vec3i(blockDim.x, blockDim.y, blockDim.z);
    if (cellIdx.x >= dims.x)
      return;
    if (cellIdx.y >= dims.y)
      return;
    if (cellIdx.z >= dims.z)
      return;

    const uint32_t cellID = cellIdx.x + cellIdx.y * dims.x + cellIdx.z * dims.x * dims.y;

    d_mcGrid[cellID] = box4f();
    d_mcGrid[cellID].lower.x = ((worldBounds.upper.x - worldBounds.lower.x) * ((cellIdx.x + 0.f) / float(dims.x)) + worldBounds.lower.x);
    d_mcGrid[cellID].lower.y = ((worldBounds.upper.y - worldBounds.lower.y) * ((cellIdx.y + 0.f) / float(dims.y)) + worldBounds.lower.y);
    d_mcGrid[cellID].lower.z = ((worldBounds.upper.z - worldBounds.lower.z) * ((cellIdx.z + 0.f) / float(dims.z)) + worldBounds.lower.z);
    d_mcGrid[cellID].upper.x = ((worldBounds.upper.x - worldBounds.lower.x) * ((cellIdx.x + 1.f) / float(dims.x)) + worldBounds.lower.x);
    d_mcGrid[cellID].upper.y = ((worldBounds.upper.y - worldBounds.lower.y) * ((cellIdx.y + 1.f) / float(dims.y)) + worldBounds.lower.y);
    d_mcGrid[cellID].upper.z = ((worldBounds.upper.z - worldBounds.lower.z) * ((cellIdx.z + 1.f) / float(dims.z)) + worldBounds.lower.z);
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

  template <typename I>
  __host__ __device__ I div_up(I a, I b)
  {
    return (a + b - 1) / b;
  }

  //-------------------------------------------------------------------------------------------------
  // Stolen from https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
  //

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

  // widens every macrocell the world-space box pb overlaps by ranges
  template <typename Macrocell, int TF_DIM>
  inline __device__ void rasterBox(Macrocell *d_mcGrid,
                                   const vec3i dims,
                                   const box3f worldBounds,
                                   const box3f pb,
                                   const MacrocellND<TF_DIM> &ranges,
                                   const size_t meshIndex = 0,
                                   const size_t numChannels = 1)
  {
    if (pb.lower.x >= pb.upper.x)
      return;
    if (pb.lower.y >= pb.upper.y)
      return;
    if (pb.lower.z >= pb.upper.z)
      return;

    vec3i lo = project(pb.lower, worldBounds, dims);
    vec3i hi = project(pb.upper, worldBounds, dims);
    hi.x = min(hi.x, dims.x - 1);
    hi.y = min(hi.y, dims.y - 1);
    hi.z = min(hi.z, dims.z - 1);

    for (uint64_t iz = lo.z; iz <= hi.z; iz++)
      for (uint64_t iy = lo.y; iy <= hi.y; iy++)
        for (uint64_t ix = lo.x; ix <= hi.x; ix++)
        {
          const uint64_t cellID = (ix + iy * dims.x + iz * dims.x * dims.y) * numChannels + meshIndex;
          widenMacrocell(d_mcGrid[cellID], ranges);
        }
  }

  __global__ void rasterElements(float2 *d_mcGrid,
                                 const vec3i mcDims,
                                 const box3f worldBounds,
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
      // printf("blockId %ld limit %ld\n", primIdx, totalElements);
      return;
    }
    box4f primBounds4 = box4f();
    // Calculate the bounds of the voxel in world space and fetch the right scalar values for each 8 corners
    // length in each dimension of the voxels in world space
    vec3f boxLenghts = (worldBounds.size()) / vec3f(gridDims);

    // 3D index of the voxel in the grid
    vec3i primIdx3D = vec3i(primIdx % gridDims.x,
                            (primIdx / gridDims.x) % gridDims.y,
                            primIdx / (gridDims.x * gridDims.y));

    // World space coordinates of the voxel corners
    vec3f vxlLower = worldBounds.lower + vec3f(primIdx3D) * boxLenghts;

    // Initialize bounds with current voxel
    primBounds4.lower = vec4f(vxlLower, scalars[primIdx]);
    primBounds4.upper = vec4f(vxlLower + boxLenghts, scalars[primIdx]);

    // Extend bounds to include neighboring voxels for proper scalar range
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

          vec3f neighborLower = worldBounds.lower + vec3f(nx, ny, nz) * boxLenghts;
          primBounds4.extend({neighborLower.x, neighborLower.y, neighborLower.z, scalars[neighborIdx]});
        }
    // __syncthreads();
    const MacrocellND<1> ranges = {{make_float2(primBounds4.lower.w, primBounds4.upper.w)}};
    rasterBox(d_mcGrid, mcDims, worldBounds, box3f(vec3f(primBounds4.lower), vec3f(primBounds4.upper)),
              ranges, meshIndex, numChannels);
  }

  OWLBuffer Renderer::buildSpatialMacrocells(const vec3i &dims, const box3f &bounds)
  {
    if (meshType != MeshType::RAW)
      throw std::runtime_error("Spatial macrocells can only be built for structured (RAW) volumes");
    uint64_t numMacrocells = dims.x * dims.y * dims.z;
    size_t numChannels = rawPtrs.size();
    if (numChannels == 0)
      throw std::runtime_error("No mesh data found");

    OWLBuffer MacrocellBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(float2),
                                                      numMacrocells * numChannels, nullptr);
    float2 *d_mcGrid = (float2 *)owlBufferGetPointer(MacrocellBuffer, 0);

    const vec3i blockSize = vec3i(4);
    sizeMCs<<<to_dims(divRoundUp(dims, blockSize)), to_dims(blockSize)>>>(d_mcGrid, dims, bounds, numChannels);
    CUDA_SYNC_CHECK();

    printf("Building Spatial Macrocells\n");

    for (size_t i = 0; i < numChannels; i++)
    {
      const float *d_scalars = (const float *)owlBufferGetPointer(scalarData[i], 0);
      const uint64_t blockSize = 32;
      const vec3i vxlGridDims = {static_cast<int>(rawPtrs[i]->getDims().x),
                                 static_cast<int>(rawPtrs[i]->getDims().y),
                                 static_cast<int>(rawPtrs[i]->getDims().z)};
      const uint64_t elementCount = uint64_t(vxlGridDims.x) * uint64_t(vxlGridDims.y) * uint64_t(vxlGridDims.z);
      const uint64_t numBlocks = divRoundUp(elementCount, blockSize);
      vec3i grid(min(numBlocks, (uint64_t)MAX_GRID_SIZE), divRoundUp(numBlocks, (uint64_t)MAX_GRID_SIZE), 1);

      if (grid.x > MAX_GRID_SIZE || grid.y > MAX_GRID_SIZE || grid.z > MAX_GRID_SIZE)
      {
        printf("Grid size exceeds limits.\n");
        exit(1);
      }
      printf("Rastering channel %d with %ldx%ldx%ld blocks with %ld threads\n",
             i, grid.x, grid.y, grid.z, blockSize);
      rasterElements<<<to_dims(grid), blockSize>>>(d_mcGrid, dims, bounds, d_scalars, vxlGridDims, i, numChannels);
      CUDA_SYNC_CHECK();
    }
    CUDA_SYNC_CHECK();
    return MacrocellBuffer;
  }

  //============================ N-D transfer functions ============================//

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

  // grows channel channelID's macrocells by the value and gradient magnitude ranges of each voxel's neighbourhood
  template <int TF_DIM>
  __global__ void rasterMacrocellsND(MacrocellND<TF_DIM> *d_mcGrid,
                                     const vec3i mcDims,
                                     const box3f worldBounds,
                                     const float *scalars,
                                     const vec3i gridDims,
                                     const size_t channelID,
                                     const size_t numChannels)
  {
    static_assert(TF_DIM <= 2, "rasterMacrocellsND: add the range of the new TF axis");
    const uint64_t primIdx = (uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * MAX_GRID_SIZE) * blockDim.x + threadIdx.x;
    if (primIdx >= uint64_t(gridDims.x) * uint64_t(gridDims.y) * uint64_t(gridDims.z))
      return;
    box4f primBounds4 = box4f();
    box3f gradientBounds; // per-component bounds of the neighbourhood's voxel gradients

    // Calculate the bounds of the voxel in world space and fetch the right scalar values for each 8 corners
    // length in each dimension of the voxels in world space
    vec3f boxLenghts = (worldBounds.size()) / vec3f(gridDims);

    // 3D index of the voxel in the grid
    vec3i primIdx3D = vec3i(primIdx % gridDims.x,
                            (primIdx / gridDims.x) % gridDims.y,
                            primIdx / (gridDims.x * gridDims.y));

    // World space coordinates of the voxel corners
    vec3f vxlLower = worldBounds.lower + vec3f(primIdx3D) * boxLenghts;

    // Initialize bounds with current voxel
    primBounds4.lower = vec4f(vxlLower, scalars[primIdx]);
    primBounds4.upper = vec4f(vxlLower + boxLenghts, scalars[primIdx]);

    // Extend bounds to include neighboring voxels for proper scalar range
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

          vec3f neighborLower = worldBounds.lower + vec3f(nx, ny, nz) * boxLenghts;
          primBounds4.extend({neighborLower.x, neighborLower.y, neighborLower.z, scalars[neighborIdx]});
          if constexpr (TF_DIM > 1)
          {
            const vec3i neighbor = vec3i(int(nx), int(ny), int(nz));
            vec3f gradient = voxelGradient(scalars, gridDims, neighbor);
            gradientBounds.extend(gradient);
            // samples within half a voxel of a face see down to half an edge voxel's gradient
            for (int i = 0; i < 3; i++)
              if (neighbor[i] == 0 || neighbor[i] == gridDims[i] - 1)
                gradient[i] *= 0.5f;
            gradientBounds.extend(gradient);
          }
        }
    MacrocellND<TF_DIM> ranges;
    ranges.range[0] = make_float2(primBounds4.lower.w, primBounds4.upper.w);
    if constexpr (TF_DIM > 1) // samples blend neighbouring gradient vectors, which can cancel out
      ranges.range[1] = lengthRange(gradientBounds);
    rasterBox(d_mcGrid, mcDims, worldBounds, box3f(vec3f(primBounds4.lower), vec3f(primBounds4.upper)),
              ranges, channelID, numChannels);
  }

  // texels a bilinear lookup inside data range r can touch, plus one for the hardware's 8-bit filter weights
  inline __device__ int2 texelRange(const float2 r, const float2 domain, const int numTexels)
  {
    const float a = normalizeToDomain(r.x, domain) * numTexels - 0.5f;
    const float b = normalizeToDomain(r.y, domain) * numTexels - 0.5f;
    if (!isfinite(a) || !isfinite(b))
      return make_int2(0, numTexels - 1);
    const float last = float(numTexels - 1);
    return make_int2(int(fminf(fmaxf(floorf(fminf(a, b)) - 1.f, 0.f), last)),
                     int(fminf(fmaxf(floorf(fmaxf(a, b)) + 2.f, 0.f), last)));
  }

  // opacity of texel idx of a TF_DIM-D lookup texture, read at its center
  template <int TF_DIM>
  inline __device__ float texelOpacity(const TransferFunctionND &tf, const int (&idx)[TF_DIM])
  {
    float coord[TF_DIM];
    for (int a = 0; a < TF_DIM; a++)
      coord[a] = (idx[a] + 0.5f) / tf.numTexels[a];
    if constexpr (TF_DIM == 2)
      return tex2D<float4>(tf.xf, coord[0], coord[1]).w;
    else
    {
      static_assert(TF_DIM == 3, "texelOpacity: no lookup texture of this dimensionality");
      return tex3D<float4>(tf.xf, coord[0], coord[1], coord[2]).w;
    }
  }

  // max opacity each channel's transfer function can return inside a macrocell's range box,
  // one warp per macrocell whose lanes split the sweep of the texel box
  template <int TF_DIM>
  __global__ void computeMajorantsND(const uint64_t numCells,
                                     const MacrocellND<TF_DIM> *macrocells,
                                     const TransferFunctionND *tfs,
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
      const TransferFunctionND &tf = tfs[c];

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

  template <int TF_DIM>
  static OWLBuffer buildSpatialMacrocellsT(Renderer &r, const vec3i &dims, const box3f &bounds)
  {
    if (r.meshType != MeshType::RAW)
      throw std::runtime_error("Spatial macrocells can only be built for structured (RAW) volumes");
    const uint64_t numMacrocells = uint64_t(dims.x) * uint64_t(dims.y) * uint64_t(dims.z);
    const size_t numChannels = r.rawPtrs.size();
    if (numChannels == 0)
      throw std::runtime_error("No mesh data found");

    const uint64_t numEntries = numMacrocells * numChannels;
    OWLBuffer macrocellBuffer = owlDeviceBufferCreate(r.context, OWL_USER_TYPE(MacrocellND<TF_DIM>),
                                                      numEntries, nullptr);
    MacrocellND<TF_DIM> *d_mcGrid = (MacrocellND<TF_DIM> *)owlBufferGetPointer(macrocellBuffer, 0);

    const uint64_t clearBlockSize = 256;
    clearMacrocellsND<TF_DIM><<<unsigned(divRoundUp(numEntries, clearBlockSize)), unsigned(clearBlockSize)>>>(
        d_mcGrid, numEntries);
    CUDA_SYNC_CHECK();

    printf("Building %dD Spatial Macrocells\n", TF_DIM);
    for (size_t i = 0; i < numChannels; i++)
    {
      const float *d_scalars = (const float *)owlBufferGetPointer(r.scalarData[i], 0);
      const uint64_t blockSize = 32;
      const vec3i vxlGridDims = {static_cast<int>(r.rawPtrs[i]->getDims().x),
                                 static_cast<int>(r.rawPtrs[i]->getDims().y),
                                 static_cast<int>(r.rawPtrs[i]->getDims().z)};
      const uint64_t voxelCount = uint64_t(vxlGridDims.x) * uint64_t(vxlGridDims.y) * uint64_t(vxlGridDims.z);
      const uint64_t numBlocks = divRoundUp(voxelCount, blockSize);
      const vec3i grid(int(min(numBlocks, (uint64_t)MAX_GRID_SIZE)),
                       int(divRoundUp(numBlocks, (uint64_t)MAX_GRID_SIZE)), 1);
      if (grid.y > MAX_GRID_SIZE)
        throw std::runtime_error("Volume too large for the macrocell raster launch");

      rasterMacrocellsND<TF_DIM><<<to_dims(grid), unsigned(blockSize)>>>(
          d_mcGrid, dims, bounds, d_scalars, vxlGridDims, i, numChannels);
      CUDA_SYNC_CHECK();
    }
    return macrocellBuffer;
  }

  template <int TF_DIM>
  static void recalculateMajorantsT(Renderer &r)
  {
    const size_t numChannels = r.rawPtrs.size();
    // wait until every channel has its lookup table; the viewer uploads them one by one
    if (numChannels == 0 || r.tfdatasND.size() < numChannels)
      return;
    std::vector<TransferFunctionND> tfs(numChannels);
    for (size_t c = 0; c < numChannels; ++c)
    {
      if (r.tfdatasND[c].colorMapTexture == 0)
        return;
      tfs[c] = r.tfdatasND[c].toDevice();
    }

    const bool perChannel = r.mode <= Mode::MULTI_ALT;
    const uint64_t numCells = uint64_t(r.macrocellDims.x) * uint64_t(r.macrocellDims.y) * uint64_t(r.macrocellDims.z);
    const uint64_t numMajorants = perChannel ? numCells * numChannels : numCells;

    OWLBuffer tfBuffer = owlDeviceBufferCreate(r.context, OWL_USER_TYPE(TransferFunctionND),
                                               numChannels, tfs.data());
    const MacrocellND<TF_DIM> *d_macrocells = (const MacrocellND<TF_DIM> *)owlBufferGetPointer(r.macrocellsBuffer, 0);
    const TransferFunctionND *d_tfs = (const TransferFunctionND *)owlBufferGetPointer(tfBuffer, 0);
    float *d_majorants = (float *)owlBufferGetPointer(r.gridMaximaBuffer, 0);

    // zeroed so cells the kernel leaves untouched are skipped instead of read as garbage
    cudaMemset(d_majorants, 0, numMajorants * sizeof(float));
    const uint64_t blockSize = 256;
    computeMajorantsND<TF_DIM><<<unsigned(divRoundUp(numCells * 32, blockSize)), unsigned(blockSize)>>>(
        numCells, d_macrocells, d_tfs, numChannels, perChannel, d_majorants);
    CUDA_SYNC_CHECK();
    owlBufferDestroy(tfBuffer);
  }

  __global__ void voxelGradientMagnitudes(const float *scalars, const vec3i dims, const uint64_t stride,
                                          const uint64_t count, float *magnitudes)
  {
    const uint64_t k = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count)
      return;
    const uint64_t voxel = k * stride;
    const vec3i idx = vec3i(int(voxel % dims.x), int(voxel / dims.x % dims.y),
                            int(voxel / (uint64_t(dims.x) * uint64_t(dims.y))));
    magnitudes[k] = length(voxelGradient(scalars, dims, idx));
  }

  std::vector<float> Renderer::VoxelGradientMagnitudes(size_t channelID, size_t stride) const
  {
    const raw::RawR &raw = *rawPtrs[channelID];
    const std::vector<float> data = raw.getDataVector();
    const vec3i dims = vec3i(int(raw.getDims().x), int(raw.getDims().y), int(raw.getDims().z));
    stride = std::max<size_t>(stride, 1);
    const uint64_t count = divRoundUp(uint64_t(data.size()), uint64_t(stride));
    std::vector<float> magnitudes(count);
    if (count == 0)
      return magnitudes;

    float *d_scalars = nullptr, *d_magnitudes = nullptr;
    if (cudaMalloc(&d_scalars, data.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_magnitudes, count * sizeof(float)) != cudaSuccess)
    {
      cudaFree(d_scalars);
      throw std::runtime_error("Not enough device memory for the gradient magnitudes");
    }
    cudaMemcpy(d_scalars, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    const uint64_t blockSize = 256;
    voxelGradientMagnitudes<<<unsigned(divRoundUp(count, blockSize)), unsigned(blockSize)>>>(
        d_scalars, dims, uint64_t(stride), count, d_magnitudes);
    CUDA_SYNC_CHECK();
    cudaMemcpy(magnitudes.data(), d_magnitudes, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_scalars);
    cudaFree(d_magnitudes);
    return magnitudes;
  }

  OWLBuffer Renderer::buildSpatialMacrocellsND(const vec3i &dims, const box3f &bounds)
  {
    switch (tfDim)
    {
    case 2:
      return buildSpatialMacrocellsT<2>(*this, dims, bounds);
    default:
      throw std::runtime_error("No macrocell builder for a " + std::to_string(tfDim) + "D transfer function");
    }
  }

  void Renderer::RecalculateDensityRangesND()
  {
    switch (tfDim)
    {
    case 2:
      recalculateMajorantsT<2>(*this);
      break;
    default:
      throw std::runtime_error("No majorant kernel for a " + std::to_string(tfDim) + "D transfer function");
    }
  }

  static const char *humanSize(uint64_t bytes)
  {
    char *suffix[] = {"B", "KB", "MB", "GB", "TB"};
    char length = sizeof(suffix) / sizeof(suffix[0]);

    int i = 0;
    double dblBytes = bytes;

    if (bytes > 1024)
    {
      for (i = 0; (bytes / 1024) > 0 && i < length - 1; i++, bytes /= 1024)
        dblBytes = bytes / 1024.0;
    }

    static char output[200];
    sprintf(output, "%.02lf %s", dblBytes, suffix[i]);
    return output;
  }
};

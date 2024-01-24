//source:  https://gitlab.com/n8vm/owlptdvr/-/blob/master/renderer.cu?ref_type=heads 
#include "renderer.h"
#include "mathHelper.h"

#include <cub/cub.cuh>

#include "cuda_fp16.h"

#include "hilbert.h"

#define CUDA_SYNC_CHECK()                                     \
{                                                             \
  cudaDeviceSynchronize();                                    \
  cudaError_t rc = cudaGetLastError();                        \
  if (rc != cudaSuccess) {                                    \
    fprintf(stderr, "error (%s: line %d): %s\n",              \
            __FILE__, __LINE__, cudaGetErrorString(rc));      \
    throw std::runtime_error("fatal cuda error");             \
  }                                                           \
}


__global__
void _recalculateDensityRanges(
  int numPrims, bool is_background, owl::box4f *bboxes,//const uint8_t* nvdbData, 
  dtracker::TFData* tf, size_t numMeshes,
  float* maxima)
{
    int primID = (blockIdx.x * blockDim.x + threadIdx.x);
    if (primID >= numPrims) return;

    // printf("primID %d level %d \n", primID, level);
    // if (level == -1)
      // printf("PrimID %d domain %f %f addr1 %f addr2 %f addrMin %d addrMax %d min %f max %f\n", primID, domain.x, domain.y, addr1, addr2, addrMin, addrMax,  minDensity, maxDensity);

    float mx = 0.f, mn = 0.f;
    if (!is_background) {
      mn = bboxes[primID].lower.w;
      mx = bboxes[primID].upper.w;
    }
    float maxDensity = 0.f;
    for(size_t tfID=0; tfID < numMeshes; tfID++)
    {
      // empty box
      if (mx < mn) {
        continue;
      }

      // transform data min max to transfer function space
      float remappedMin1 = (mn - tf[tfID].volDomain.lo) / (tf[tfID].volDomain.hi - tf[tfID].volDomain.lo);
      float remappedMin = (remappedMin1 - tf[tfID].xfDomain.lo) / (tf[tfID].xfDomain.hi - tf[tfID].xfDomain.lo);
      float remappedMax1 = (mx - tf[tfID].volDomain.lo) / (tf[tfID].volDomain.hi - tf[tfID].volDomain.lo);
      float remappedMax = (remappedMax1 - tf[tfID].xfDomain.lo) / (tf[tfID].xfDomain.hi - tf[tfID].xfDomain.lo);
      float addr1 = remappedMin * tf[tfID].numTexels;
      float addr2 = remappedMax * tf[tfID].numTexels;

      int addrMin = min(max(int(min(floor(addr1), floor(addr2))), 0), tf[tfID].numTexels-1);
      int addrMax = min(max(int(max(ceil(addr1), ceil(addr2))), 0), tf[tfID].numTexels-1);

      float maxDensityForVolume;
      for (int i = addrMin; i <= addrMax; ++i) {
        float density = tex2D<float4>(tf[tfID].colorMapTexture, float(i)/tf[tfID].numTexels ,0.5f).w * tf[tfID].opacityScale;
        if (i == addrMin) maxDensityForVolume = density;
        else maxDensityForVolume = max(maxDensityForVolume, density);
      }
      maxDensity += maxDensityForVolume;
    }
    maxima[primID] = maxDensity/(float)numMeshes;
}

__global__
void _recalculateDensityRanges(
  uint32_t numPrims, half *bboxes,
  cudaTextureObject_t texture, int numTexels, 
  float2 volumeDomain, float2 xfDomain, float opacityScale,
  uint8_t* maxima)
{
    uint32_t primID = (blockIdx.x * blockDim.x + threadIdx.x);
    if (primID >= numPrims) return;

    float mn = __half2float(bboxes[primID * 8ull + 3ull]);
    float mx = __half2float(bboxes[primID * 8ull + 7ull]);

    // empty box
    if (mx < mn) {
      maxima[primID] = 0;
      return;
    }

    // transform data min max to transfer function space
    float remappedMin1 = (mn - volumeDomain.x) / (volumeDomain.y - volumeDomain.x);
    float remappedMin = (remappedMin1 - xfDomain.x) / (xfDomain.y - xfDomain.x);
    float remappedMax1 = (mx - volumeDomain.x) / (volumeDomain.y - volumeDomain.x);
    float remappedMax = (remappedMax1 - xfDomain.x) / (xfDomain.y - xfDomain.x);
    float addr1 = remappedMin * numTexels;
    float addr2 = remappedMax * numTexels;

    int addrMin = min(max(int(min(floor(addr1), floor(addr2))), 0), numTexels-1);
    int addrMax = min(max(int(max(ceil(addr1), ceil(addr2))), 0), numTexels-1);


    float maxDensity;
    for (int i = addrMin; i <= addrMax; ++i) {
      float density = tex2D<float4>(texture, float(i)/numTexels ,0.5f).w * opacityScale;
      if (i == addrMin) maxDensity = density;
      else maxDensity = max(maxDensity, density);
    }
    maxima[primID] = uint8_t(maxDensity * 256);    
} 

__global__
void _recalculateDensityRanges(
  int numPrims, const float2 *macrocells,//const uint8_t* nvdbData,
  const dtracker::TFData* tf, size_t numMeshes, float* maxima)
{
    int nodeID = (blockIdx.x * blockDim.x + threadIdx.x);
    if (nodeID >= numPrims) return;

    float maxDensity = 0.f;
    maxima[nodeID] = 0.f;
    for (size_t tfID = 0; tfID < numMeshes; tfID++)
    {
      float mn = macrocells[nodeID * numMeshes + tfID].x;
      float mx = macrocells[nodeID * numMeshes + tfID].y;

      // empty box
      if (mx < mn) {
        //maxima[nodeID] = 0.f;
        continue; //return;
      }

      // transform data min max to transfer function space
      float remappedMin1 = (mn - tf[tfID].volDomain.lo) / (tf[tfID].volDomain.hi - tf[tfID].volDomain.lo);
      float remappedMin = (remappedMin1 - tf[tfID].xfDomain.lo) / (tf[tfID].xfDomain.hi - tf[tfID].xfDomain.lo);
      float remappedMax1 = (mx - tf[tfID].volDomain.lo) / (tf[tfID].volDomain.hi - tf[tfID].volDomain.lo);
      float remappedMax = (remappedMax1 - tf[tfID].xfDomain.lo) / (tf[tfID].xfDomain.hi - tf[tfID].xfDomain.lo);
      float addr1 = remappedMin * (tf[tfID].numTexels);
      float addr2 = remappedMax * (tf[tfID].numTexels);


      int addrMin = min(max(int(min((addr1), (addr2))), 0), tf[tfID].numTexels);
      int addrMax = min(max(int(max((addr1), (addr2))), 0), tf[tfID].numTexels);

      float maxDensityForVolume = 0.f;
      for (int i = addrMin; i <= addrMax; ++i) {
        float density = tex2D<float4>(tf[tfID].colorMapTexture, (float(i)+0.5f)/tf[tfID].numTexels, 0.5f).w * tf[tfID].opacityScale;
        if (i == addrMin) maxDensityForVolume = density;
        else maxDensityForVolume = max(maxDensityForVolume, density);
      }
      float density = tex2D<float4>(tf[tfID].colorMapTexture, (float(addr1)+0.5f)/tf[tfID].numTexels, 0.5f).w * tf[tfID].opacityScale;
      maxDensityForVolume = max(maxDensityForVolume, density);
      density = tex2D<float4>(tf[tfID].colorMapTexture, (float(addr2)+0.5f)/tf[tfID].numTexels, 0.5f).w * tf[tfID].opacityScale;
      maxDensityForVolume = max(maxDensityForVolume, density);
      maxDensity += maxDensityForVolume;
    }
    maxima[nodeID] = maxDensity;
}

__global__
void _recalculateDensityRangesMM(
  int numPrims, const float2 *macrocells,
  const dtracker::TFData* tf, size_t numMeshes, float* maxima)
{
    int nodeID = (blockIdx.x * blockDim.x + threadIdx.x);
    if (nodeID >= numPrims) return;

    for (size_t tfID = 0; tfID < numMeshes; tfID++)
    {
      maxima[nodeID * numMeshes + tfID] = 0.f;
      float mn = macrocells[nodeID * numMeshes + tfID].x;
      float mx = macrocells[nodeID * numMeshes + tfID].y;

      // empty box
      if (mx < mn) {
        //maxima[nodeID] = 0.f;
        continue; //return;
      }

      // transform data min max to transfer function space
      float remappedMin1 = (mn - tf[tfID].volDomain.lo) / (tf[tfID].volDomain.hi - tf[tfID].volDomain.lo);
      float remappedMin = (remappedMin1 - tf[tfID].xfDomain.lo) / (tf[tfID].xfDomain.hi - tf[tfID].xfDomain.lo);
      float remappedMax1 = (mx - tf[tfID].volDomain.lo) / (tf[tfID].volDomain.hi - tf[tfID].volDomain.lo);
      float remappedMax = (remappedMax1 - tf[tfID].xfDomain.lo) / (tf[tfID].xfDomain.hi - tf[tfID].xfDomain.lo);
      float addr1 = remappedMin * (tf[tfID].numTexels);
      float addr2 = remappedMax * (tf[tfID].numTexels);


      int addrMin = min(max(int(min((addr1), (addr2))), 0), tf[tfID].numTexels);
      int addrMax = min(max(int(max((addr1), (addr2))), 0), tf[tfID].numTexels);

      float maxDensityForVolume = 0.f;
      for (int i = addrMin; i <= addrMax; ++i) {
        float density = tex2D<float4>(tf[tfID].colorMapTexture, (float(i)+0.5f)/tf[tfID].numTexels, 0.5f).w * tf[tfID].opacityScale;
        if (i == addrMin) maxDensityForVolume = density;
        else maxDensityForVolume = max(maxDensityForVolume, density);
      }
      float density = tex2D<float4>(tf[tfID].colorMapTexture, (float(addr1)+0.5f)/tf[tfID].numTexels, 0.5f).w * tf[tfID].opacityScale;
      maxDensityForVolume = max(maxDensityForVolume, density);
      density = tex2D<float4>(tf[tfID].colorMapTexture, (float(addr2)+0.5f)/tf[tfID].numTexels, 0.5f).w * tf[tfID].opacityScale;
      maxDensityForVolume = max(maxDensityForVolume, density);
      maxima[nodeID * numMeshes + tfID] = maxDensityForVolume;
    }
}

namespace dtracker {
  void Renderer::RecalculateDensityRanges()
  {
    size_t numMeshes = meshType != MeshType::UNDEFINED ? (meshType == MeshType::UMESH ?umeshPtrs.size() : rawPtrs.size()) : 0;
    if(numMeshes == 0) throw std::runtime_error("No mesh data found");

    //Create a dvice buffer of TFData
    OWLBuffer tfdataBuffer = owlDeviceBufferCreate(context, 
      OWL_USER_TYPE(dtracker::TFData),
      numMeshes,  tfdatas.data());
    dtracker::TFData* d_tfdatas = (dtracker::TFData*)owlBufferGetPointer(tfdataBuffer, 0);

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
    float* majorantBuffer;
    // root
    {
      bboxes = (owl::box4f*)owlBufferGetPointer(rootBBoxBuffer, 0);
      isBackground = false;
      numThreads = 1;
      gridSize = dim3((numThreads + blockSize.x - 1) / blockSize.x);
      majorantBuffer = (float*)owlBufferGetPointer(rootMaximaBuffer, 0);
      _recalculateDensityRanges<<<gridSize,blockSize>>>(
        numThreads, isBackground, bboxes, 
        d_tfdatas, numMeshes, majorantBuffer);
      
      CUDA_SYNC_CHECK();
      
      owlGroupBuildAccel(rootMacrocellBLAS);
      owlGroupBuildAccel(rootMacrocellTLAS);
    }
    {
      float2* macrocells = (float2*)owlBufferGetPointer(macrocellsBuffer, 0); 
      numThreads = macrocellDims.x * macrocellDims.y * macrocellDims.z;
      gridSize = dim3 ((numThreads + blockSize.x - 1) / blockSize.x);
      majorantBuffer = (float*)owlBufferGetPointer(gridMaximaBuffer, 0);
      
      //Calculate majorants
      if(mode > 0)
        _recalculateDensityRangesMM<<<gridSize,blockSize>>>(
          numThreads, macrocells, d_tfdatas, numMeshes,
          majorantBuffer);
      else
        _recalculateDensityRanges<<<gridSize,blockSize>>>(
          numThreads, macrocells, d_tfdatas, numMeshes,
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

  // For umeshes
  
  /*! max macrocell grid size, in cells */
  constexpr int MAX_GRID_SIZE = 1024*1024;

  __global__ void clearMCs(box4f *d_mcGrid, const vec3i dims)
  {
    const vec3i cellIdx
      = vec3i(threadIdx)
      + vec3i(blockIdx)*vec3i(blockDim.x,blockDim.y,blockDim.z);
    if (cellIdx.x >= dims.x) return;
    if (cellIdx.y >= dims.y) return;
    if (cellIdx.z >= dims.z) return;

    const uint32_t cellID
      = cellIdx.x
      + cellIdx.y * dims.x
      + cellIdx.z * dims.x * dims.y;
    d_mcGrid[cellID] = box4f();
  }

  __global__ void clearClusters(box4f *clusters, uint32_t numClusters)
  {
    const uint64_t blockID  
      = blockIdx.x
      + blockIdx.y * MAX_GRID_SIZE;
    uint64_t index = blockID*blockDim.x + threadIdx.x;
    if (index >= numClusters) return;
    clusters[index] = box4f();
  }

  __global__ void sizeMCs(box4f *d_mcGrid, const vec3i dims, const box3f worldBounds)
  {
    const vec3i cellIdx
      = vec3i(threadIdx)
      + vec3i(blockIdx)*vec3i(blockDim.x,blockDim.y,blockDim.z);
    if (cellIdx.x >= dims.x) return;
    if (cellIdx.y >= dims.y) return;
    if (cellIdx.z >= dims.z) return;

    const uint32_t cellID
      = cellIdx.x
      + cellIdx.y * dims.x
      + cellIdx.z * dims.x * dims.y;

    d_mcGrid[cellID] = box4f();
    d_mcGrid[cellID].lower.x = ((worldBounds.upper.x - worldBounds.lower.x) * ((cellIdx.x + 0.f) / float(dims.x)) + worldBounds.lower.x);
    d_mcGrid[cellID].lower.y = ((worldBounds.upper.y - worldBounds.lower.y) * ((cellIdx.y + 0.f) / float(dims.y)) + worldBounds.lower.y);
    d_mcGrid[cellID].lower.z = ((worldBounds.upper.z - worldBounds.lower.z) * ((cellIdx.z + 0.f) / float(dims.z)) + worldBounds.lower.z);
    d_mcGrid[cellID].upper.x = ((worldBounds.upper.x - worldBounds.lower.x) * ((cellIdx.x + 1.f) / float(dims.x)) + worldBounds.lower.x);
    d_mcGrid[cellID].upper.y = ((worldBounds.upper.y - worldBounds.lower.y) * ((cellIdx.y + 1.f) / float(dims.y)) + worldBounds.lower.y);
    d_mcGrid[cellID].upper.z = ((worldBounds.upper.z - worldBounds.lower.z) * ((cellIdx.z + 1.f) / float(dims.z)) + worldBounds.lower.z);
  }

  __global__ void sizeMCs(float2 *d_mcGrid, const vec3i dims, const box3f worldBounds, const size_t numMeshes=1)
  {
    const vec3i cellIdx
      = vec3i(threadIdx)
      + vec3i(blockIdx)*vec3i(blockDim.x,blockDim.y,blockDim.z);
    if (cellIdx.x >= dims.x) return;
    if (cellIdx.y >= dims.y) return;
    if (cellIdx.z >= dims.z) return;

    const uint32_t cellID
      = (cellIdx.x
      + cellIdx.y * dims.x
      + cellIdx.z * dims.x * dims.y) * numMeshes;

    for (size_t meshIndex=0; meshIndex < numMeshes; meshIndex++) {
      d_mcGrid[cellID + meshIndex] = make_float2(1e20, -1e20);
    }
  }

  inline dim3 to_dims(const vec3i v)
  { return { (unsigned)v.x,(unsigned)v.y,(unsigned)v.z }; }

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
      while(val < __int_as_float(ret))
      {
          int old = ret;
          if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
              break;
      }
      return __int_as_float(ret);
  }

  __device__ __forceinline__ float atomicMax(float *address, float val)
  {
      int ret = __float_as_int(*address);
      while(val > __int_as_float(ret))
      {
          int old = ret;
          if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
              break;
      }
      return __int_as_float(ret);
  }


  inline __device__
  int project(float f,
              const interval<float> range,
              int dim)
  {
    return max(0,min(dim-1,int(dim*(f-range.lower)/(range.upper-range.lower))));
  }

  inline __device__
  vec3i project(const vec3f &pos,
                const box3f &bounds,
                const vec3i &dims)
  {
    return vec3i(project(pos.x,{bounds.lower.x,bounds.upper.x},dims.x),
                 project(pos.y,{bounds.lower.y,bounds.upper.y},dims.y),
                 project(pos.z,{bounds.lower.z,bounds.upper.z},dims.z));
  }

  inline __device__
  void insertIntoBox(box4f *d_mcGrid,
                 const vec3i dims,
                 const box3f worldBounds,
                 const box4f primBounds4,
                 bool dbg=false)
  {
    box3f pb = box3f(vec3f(primBounds4.lower),
                     vec3f(primBounds4.upper));
    if (pb.lower.x >= pb.upper.x) return;
    if (pb.lower.y >= pb.upper.y) return;
    if (pb.lower.z >= pb.upper.z) return;

    vec3i lo = project(pb.lower,worldBounds,dims);
    vec3i hi = project(pb.upper,worldBounds,dims);

    for (int iz=lo.z;iz<=hi.z;iz++)
      for (int iy=lo.y;iy<=hi.y;iy++)
        for (int ix=lo.x;ix<=hi.x;ix++) {
          const uint32_t cellID
            = ix
            + iy * dims.x
            + iz * dims.x * dims.y;
          auto &cell = d_mcGrid[cellID];

          box3f cb = box3f(vec3f(ix, iy, iz),
                        vec3f(ix + 1, iy + 1, iz + 1));
          cb.lower = worldBounds.lower + (cb.lower * vec3f(1.f / dims.x, 1.f / dims.y, 1.f / dims.z)) * (worldBounds.upper - worldBounds.lower);
          cb.upper = worldBounds.lower + (cb.upper * vec3f(1.f / dims.x, 1.f / dims.y, 1.f / dims.z)) * (worldBounds.upper - worldBounds.lower);

          // cell.lower.x = cb.lower.x;
          // cell.upper.x = cb.upper.x;
          // cell.lower.y = cb.lower.y;
          // cell.upper.y = cb.upper.y;
          // cell.lower.z = cb.lower.z;
          // cell.upper.z = cb.upper.z;
          atomicMin(&cell.lower.x,primBounds4.lower.x);
          atomicMax(&cell.upper.x,primBounds4.upper.x);
          atomicMin(&cell.lower.y,primBounds4.lower.y);
          atomicMax(&cell.upper.y,primBounds4.upper.y);
          atomicMin(&cell.lower.z,primBounds4.lower.z);
          atomicMax(&cell.upper.z,primBounds4.upper.z);
          atomicMin(&cell.lower.w,primBounds4.lower.w);
          atomicMax(&cell.upper.w,primBounds4.upper.w);
        }
  }

  inline __device__
  void rasterBox(box4f *d_mcGrid,
                 const vec3i dims,
                 const box3f worldBounds,
                 const box4f primBounds4,
                 bool dbg=false)
  {
    box3f pb = box3f(vec3f(primBounds4.lower),
                     vec3f(primBounds4.upper));
    if (pb.lower.x >= pb.upper.x) return;
    if (pb.lower.y >= pb.upper.y) return;
    if (pb.lower.z >= pb.upper.z) return;

    vec3i lo = project(pb.lower,worldBounds,dims);
    vec3i hi = project(pb.upper,worldBounds,dims);

    for (int iz=lo.z;iz<=hi.z;iz++)
      for (int iy=lo.y;iy<=hi.y;iy++)
        for (int ix=lo.x;ix<=hi.x;ix++) {
          const uint32_t cellID
            = ix
            + iy * dims.x
            + iz * dims.x * dims.y;
          auto &cell = d_mcGrid[cellID];

          box3f cb = box3f(vec3f(ix, iy, iz),
                        vec3f(ix + 1, iy + 1, iz + 1));
          cb.lower = worldBounds.lower + (cb.lower * vec3f(1.f / dims.x, 1.f / dims.y, 1.f / dims.z)) * (worldBounds.upper - worldBounds.lower);
          cb.upper = worldBounds.lower + (cb.upper * vec3f(1.f / dims.x, 1.f / dims.y, 1.f / dims.z)) * (worldBounds.upper - worldBounds.lower);

          atomicMin(&cell.lower.w,primBounds4.lower.w);
          atomicMax(&cell.upper.w,primBounds4.upper.w);
        }
  }

  inline __device__
  void rasterBox(float2 *d_mcGrid,
                 const vec3i dims,
                 const box3f worldBounds,
                 const box4f primBounds4,
                 const size_t meshIndex=0,
                 const size_t numMeshes=1,
                 bool dbg=false)
  {
    box3f pb = box3f(vec3f(primBounds4.lower),
                     vec3f(primBounds4.upper));
    if (pb.lower.x >= pb.upper.x) return;
    if (pb.lower.y >= pb.upper.y) return;
    if (pb.lower.z >= pb.upper.z) return;

    vec3i lo = project(pb.lower,worldBounds,dims);
    vec3i hi = project(pb.upper,worldBounds,dims);

    for (int iz=lo.z;iz<=hi.z;iz++)
      for (int iy=lo.y;iy<=hi.y;iy++)
        for (int ix=lo.x;ix<=hi.x;ix++) {
          const uint32_t cellID
            = (ix
            + iy * dims.x
            + iz * dims.x * dims.y) * numMeshes +  meshIndex;
            
          atomicMin(&d_mcGrid[cellID].x,primBounds4.lower.w);
          atomicMax(&d_mcGrid[cellID].y,primBounds4.upper.w);
        }
  }

  __global__ void _computeBoundingBoxes(
                             const vec3f *vertices,
                             const float *scalars,
                             const int *tetrahedra,
                             const int *pyramids,
                             const int *wedges,
                             const int *hexahedra,
                             const uint64_t numTetrahedra,
                             const uint64_t numPyramids,
                             const uint64_t numWedges,
                             const uint64_t numHexahedra,
                             const uint64_t numVertices,
                             half *boundingBoxes
  ) {
    const uint64_t blockID = blockIdx.x + blockIdx.y * MAX_GRID_SIZE;
    uint64_t index = blockID*blockDim.x + threadIdx.x;
    uint64_t pyrOffset = numTetrahedra;
    uint64_t wedOffset = numTetrahedra + numPyramids;
    uint64_t hexOffset = numTetrahedra + numPyramids + numWedges;
    uint64_t totalElements = numTetrahedra + numPyramids + numWedges + numHexahedra;
    if (index >= totalElements) return;

    box4f primBounds4 = box4f();

    if (index < pyrOffset) {
      int i0 = tetrahedra[index * 4ull + 0ull];
      int i1 = tetrahedra[index * 4ull + 1ull];
      int i2 = tetrahedra[index * 4ull + 2ull];
      int i3 = tetrahedra[index * 4ull + 3ull];     
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
    } 
    else if (index < wedOffset) {
      index -= pyrOffset;
      int i0 = pyramids[index * 5ull + 0ull];
      int i1 = pyramids[index * 5ull + 1ull];
      int i2 = pyramids[index * 5ull + 2ull];
      int i3 = pyramids[index * 5ull + 3ull];
      int i4 = pyramids[index * 5ull + 4ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
    }
    else if (index < hexOffset) {
      index -= wedOffset;
      int i0 = wedges[index * 6ull + 0ull];
      int i1 = wedges[index * 6ull + 1ull];
      int i2 = wedges[index * 6ull + 2ull];
      int i3 = wedges[index * 6ull + 3ull];
      int i4 = wedges[index * 6ull + 4ull];
      int i5 = wedges[index * 6ull + 5ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
    }
    else {
      index -= hexOffset;
      int i0 = hexahedra[index * 8ull + 0ull];
      int i1 = hexahedra[index * 8ull + 1ull];
      int i2 = hexahedra[index * 8ull + 2ull];
      int i3 = hexahedra[index * 8ull + 3ull];
      int i4 = hexahedra[index * 8ull + 4ull];
      int i5 = hexahedra[index * 8ull + 5ull];
      int i6 = hexahedra[index * 8ull + 6ull];
      int i7 = hexahedra[index * 8ull + 7ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
      primBounds4 = primBounds4.extend({vertices[i6].x, vertices[i6].y, vertices[i6].z, scalars[i6]} );
      primBounds4 = primBounds4.extend({vertices[i7].x, vertices[i7].y, vertices[i7].z, scalars[i7]} );
    }

    boundingBoxes[index * 8ull + 0ull] = __float2half_rd(primBounds4.lower.x);
    boundingBoxes[index * 8ull + 1ull] = __float2half_rd(primBounds4.lower.y);
    boundingBoxes[index * 8ull + 2ull] = __float2half_rd(primBounds4.lower.z);
    boundingBoxes[index * 8ull + 3ull] = __float2half_rd(primBounds4.lower.w);
    boundingBoxes[index * 8ull + 4ull] = __float2half_ru(primBounds4.upper.x);
    boundingBoxes[index * 8ull + 5ull] = __float2half_ru(primBounds4.upper.y);
    boundingBoxes[index * 8ull + 6ull] = __float2half_ru(primBounds4.upper.z);
    boundingBoxes[index * 8ull + 7ull] = __float2half_ru(primBounds4.upper.w);
  }





  __global__ void _computeRanges(
                             const vec3f *vertices,
                             const float *scalars,
                             const int *tetrahedra,
                             const int *pyramids,
                             const int *wedges,
                             const int *hexahedra,
                             const uint64_t numTetrahedra,
                             const uint64_t numPyramids,
                             const uint64_t numWedges,
                             const uint64_t numHexahedra,
                             const uint64_t numVertices,
                             float2 *ranges
  ) {
    const uint64_t blockID = blockIdx.x + blockIdx.y * MAX_GRID_SIZE;
    uint64_t index = blockID*blockDim.x + threadIdx.x;
    uint64_t pyrOffset = numTetrahedra;
    uint64_t wedOffset = numTetrahedra + numPyramids;
    uint64_t hexOffset = numTetrahedra + numPyramids + numWedges;
    uint64_t totalElements = numTetrahedra + numPyramids + numWedges + numHexahedra;
    if (index >= totalElements) return;

    float2 primRange = make_float2(1e20f, -1e20f);

    if (index < pyrOffset) {
      int i0 = tetrahedra[index * 4ull + 0ull];
      int i1 = tetrahedra[index * 4ull + 1ull];
      int i2 = tetrahedra[index * 4ull + 2ull];
      int i3 = tetrahedra[index * 4ull + 3ull];
      primRange.x = min(primRange.x, scalars[i0]); primRange.y = max(primRange.y, scalars[i0]);
      primRange.x = min(primRange.x, scalars[i1]); primRange.y = max(primRange.y, scalars[i1]);
      primRange.x = min(primRange.x, scalars[i2]); primRange.y = max(primRange.y, scalars[i2]);
      primRange.x = min(primRange.x, scalars[i3]); primRange.y = max(primRange.y, scalars[i3]);
    } 
    else if (index < wedOffset) {
      index -= pyrOffset;
      int i0 = pyramids[index * 5ull + 0ull];
      int i1 = pyramids[index * 5ull + 1ull];
      int i2 = pyramids[index * 5ull + 2ull];
      int i3 = pyramids[index * 5ull + 3ull];
      int i4 = pyramids[index * 5ull + 4ull];
      primRange.x = min(primRange.x, scalars[i0]); primRange.y = max(primRange.y, scalars[i0]);
      primRange.x = min(primRange.x, scalars[i1]); primRange.y = max(primRange.y, scalars[i1]);
      primRange.x = min(primRange.x, scalars[i2]); primRange.y = max(primRange.y, scalars[i2]);
      primRange.x = min(primRange.x, scalars[i3]); primRange.y = max(primRange.y, scalars[i3]);
      primRange.x = min(primRange.x, scalars[i4]); primRange.y = max(primRange.y, scalars[i4]);
    }
    else if (index < hexOffset) {
      index -= wedOffset;
      int i0 = wedges[index * 6ull + 0ull];
      int i1 = wedges[index * 6ull + 1ull];
      int i2 = wedges[index * 6ull + 2ull];
      int i3 = wedges[index * 6ull + 3ull];
      int i4 = wedges[index * 6ull + 4ull];
      int i5 = wedges[index * 6ull + 5ull];
      primRange.x = min(primRange.x, scalars[i0]); primRange.y = max(primRange.y, scalars[i0]);
      primRange.x = min(primRange.x, scalars[i1]); primRange.y = max(primRange.y, scalars[i1]);
      primRange.x = min(primRange.x, scalars[i2]); primRange.y = max(primRange.y, scalars[i2]);
      primRange.x = min(primRange.x, scalars[i3]); primRange.y = max(primRange.y, scalars[i3]);
      primRange.x = min(primRange.x, scalars[i4]); primRange.y = max(primRange.y, scalars[i4]);
      primRange.x = min(primRange.x, scalars[i5]); primRange.y = max(primRange.y, scalars[i5]);
    }
    else {
      index -= hexOffset;
      int i0 = hexahedra[index * 8ull + 0ull];
      int i1 = hexahedra[index * 8ull + 1ull];
      int i2 = hexahedra[index * 8ull + 2ull];
      int i3 = hexahedra[index * 8ull + 3ull];
      int i4 = hexahedra[index * 8ull + 4ull];
      int i5 = hexahedra[index * 8ull + 5ull];
      int i6 = hexahedra[index * 8ull + 6ull];
      int i7 = hexahedra[index * 8ull + 7ull];      
      primRange.x = min(primRange.x, scalars[i0]); primRange.y = max(primRange.y, scalars[i0]);
      primRange.x = min(primRange.x, scalars[i1]); primRange.y = max(primRange.y, scalars[i1]);
      primRange.x = min(primRange.x, scalars[i2]); primRange.y = max(primRange.y, scalars[i2]);
      primRange.x = min(primRange.x, scalars[i3]); primRange.y = max(primRange.y, scalars[i3]);
      primRange.x = min(primRange.x, scalars[i4]); primRange.y = max(primRange.y, scalars[i4]);
      primRange.x = min(primRange.x, scalars[i5]); primRange.y = max(primRange.y, scalars[i5]);
      primRange.x = min(primRange.x, scalars[i6]); primRange.y = max(primRange.y, scalars[i6]);
      primRange.x = min(primRange.x, scalars[i7]); primRange.y = max(primRange.y, scalars[i7]);
    }

    ranges[index] = primRange;
  }

  void Renderer::computeRanges(OWLBuffer &ranges) {
    unsigned numThreads = 1024;
    unsigned numElements = umeshPtrs[0]->numVolumeElements();

    const vec3f *d_vertices = (const vec3f*)owlBufferGetPointer(verticesData,0);
    const float *d_scalars = (const float*)owlBufferGetPointer(scalarData[0],0);//!!
    const int *d_tetrahedra = (const int*)owlBufferGetPointer(tetrahedraData,0);
    const int *d_pyramids = (const int*)owlBufferGetPointer(pyramidsData,0);
    const int *d_wedges = (const int*)owlBufferGetPointer(wedgesData,0);
    const int *d_hexahedra = (const int*)owlBufferGetPointer(hexahedraData,0);
    float2 *d_ranges = (float2*)owlBufferGetPointer(ranges,0);
    {
      _computeRanges<<<div_up(numElements, numThreads), numThreads>>>(
        d_vertices, d_scalars, d_tetrahedra, d_pyramids, d_wedges, d_hexahedra, 
        umeshPtrs[0]->tets.size(), umeshPtrs[0]->pyrs.size(), umeshPtrs[0]->wedges.size(), umeshPtrs[0]->hexes.size(),umeshPtrs[0]->vertices.size(),
        d_ranges
      );
    }

    {
      // make the host block until the device is finished with foo
      cudaDeviceSynchronize();

      // check for error
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error(cudaGetErrorString(error));
      }
    }

  }


  __global__ void clusterElementsIntoGrid(box4f *d_mcGrid,
                             const vec3i dims,
                             const box3f worldBounds,
                             const vec3f *vertices,
                             const float *scalars,
                             const int *tetrahedra,
                             const int *pyramids,
                             const int *wedges,
                             const int *hexahedra,
                             const uint64_t numTetrahedra,
                             const uint64_t numPyramids,
                             const uint64_t numWedges,
                             const uint64_t numHexahedra,
                             const uint64_t numVertices
                             )
  {
    const uint64_t blockID
      = blockIdx.x
      + blockIdx.y * MAX_GRID_SIZE;
    uint64_t primIdx = blockID*blockDim.x + threadIdx.x;
    
    uint64_t pyrOffset = numTetrahedra;
    uint64_t wedOffset = numTetrahedra + numPyramids;
    uint64_t hexOffset = numTetrahedra + numPyramids + numWedges;
    uint64_t totalElements = numTetrahedra + numPyramids + numWedges + numHexahedra;
    if (primIdx >= totalElements) return;   // temporary 

    box4f primBounds4 = box4f();

    // if (primIdx < 0) return;
    // else 
    if (primIdx < pyrOffset) {
      int i0 = tetrahedra[primIdx * 4ull + 0ull];
      int i1 = tetrahedra[primIdx * 4ull + 1ull];
      int i2 = tetrahedra[primIdx * 4ull + 2ull];
      int i3 = tetrahedra[primIdx * 4ull + 3ull];
      // if (i0 < 0 || i0 >= numVertices) return;
      // if (i1 < 0 || i1 >= numVertices) return;
      // if (i2 < 0 || i2 >= numVertices) return;
      // if (i3 < 0 || i3 >= numVertices) return;

      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
    } 
    else if (primIdx < wedOffset) {
      primIdx -= pyrOffset;
      int i0 = pyramids[primIdx * 5ull + 0ull];
      int i1 = pyramids[primIdx * 5ull + 1ull];
      int i2 = pyramids[primIdx * 5ull + 2ull];
      int i3 = pyramids[primIdx * 5ull + 3ull];
      int i4 = pyramids[primIdx * 5ull + 4ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
    }
    else if (primIdx < hexOffset) {
      primIdx -= wedOffset;
      int i0 = wedges[primIdx * 6ull + 0ull];
      int i1 = wedges[primIdx * 6ull + 1ull];
      int i2 = wedges[primIdx * 6ull + 2ull];
      int i3 = wedges[primIdx * 6ull + 3ull];
      int i4 = wedges[primIdx * 6ull + 4ull];
      int i5 = wedges[primIdx * 6ull + 5ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
    }
    else {
      primIdx -= hexOffset;
      int i0 = hexahedra[primIdx * 8ull + 0ull];
      int i1 = hexahedra[primIdx * 8ull + 1ull];
      int i2 = hexahedra[primIdx * 8ull + 2ull];
      int i3 = hexahedra[primIdx * 8ull + 3ull];
      int i4 = hexahedra[primIdx * 8ull + 4ull];
      int i5 = hexahedra[primIdx * 8ull + 5ull];
      int i6 = hexahedra[primIdx * 8ull + 6ull];
      int i7 = hexahedra[primIdx * 8ull + 7ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
      primBounds4 = primBounds4.extend({vertices[i6].x, vertices[i6].y, vertices[i6].z, scalars[i6]} );
      primBounds4 = primBounds4.extend({vertices[i7].x, vertices[i7].y, vertices[i7].z, scalars[i7]} );
    }

    insertIntoBox(d_mcGrid,dims,worldBounds,primBounds4);
  }

  __global__ void rasterElements(float2 *d_mcGrid,
                             const vec3i mcDims,
                             const box3f worldBounds,
                             const float *scalars,
                             const vec3i gridDims,
                             const size_t meshIndex=0,
                             size_t numMeshes=1
                             )
  {
    const uint64_t blockID
      = blockIdx.x
      + blockIdx.y * MAX_GRID_SIZE;
    uint64_t primIdx = blockID*blockDim.x + threadIdx.x;
    const uint64_t totalElements = gridDims.x * gridDims.y * gridDims.z;
    if (primIdx >= totalElements) return;
    box4f primBounds4 = box4f();
    //Calculate the bounds of the voxel in world space and fetch the right scalar values for each 8 corners
    //length in each dimension of the voxels in world space
    vec3f boxLenghts = (worldBounds.size()) / vec3f(gridDims);
    
    //3D index of the voxel in the grid
    vec3i primIdx3D = vec3i(primIdx % gridDims.x, 
                        (primIdx / gridDims.x) % gridDims.y,
                         primIdx / (gridDims.x * gridDims.y));

    //World space coordinates of the voxel corners
    vec3f vxlLower = worldBounds.lower + vec3f(primIdx3D) * boxLenghts;

    primBounds4.lower = vec4f(vxlLower, scalars[primIdx]);
    primBounds4.upper = vec4f(vxlLower + boxLenghts, scalars[primIdx]);

    if(primIdx == 269279)
      printf("primIdx: %d, primBounds4: %f %f %f %f %f %f %f %f\n", primIdx, primBounds4.lower.x,
        primBounds4.lower.y, primBounds4.lower.z, primBounds4.lower.w, primBounds4.upper.x, primBounds4.upper.y,
        primBounds4.upper.z, primBounds4.upper.w);

    for (int iz=-1;iz<=1;iz++)
      for (int iy=-1;iy<=1;iy++)
        for (int ix=-1;ix<=1;ix++) {
          const uint32_t neighborIdx 
            = (primIdx3D.x + ix)
            + (primIdx3D.y + iy) * gridDims.x
            + (primIdx3D.z + iz) * gridDims.x * gridDims.y;
            if(primIdx3D.x + ix < 0 || primIdx3D.x + ix >= gridDims.x) continue;
            if(primIdx3D.y + iy < 0 || primIdx3D.y + iy >= gridDims.y) continue;
            if(primIdx3D.z + iz < 0 || primIdx3D.z + iz >= gridDims.z) continue;
            
          primBounds4.extend({vxlLower.x, vxlLower.y, vxlLower.z, scalars[neighborIdx]});
        }

    rasterBox(d_mcGrid,mcDims,worldBounds,primBounds4,meshIndex,numMeshes);
  }

  __global__ void rasterElements(box4f *d_mcGrid,
                             const vec3i dims,
                             const box3f worldBounds,
                             const vec3f *vertices,
                             const float *scalars,
                             const int *tetrahedra,
                             const int *pyramids,
                             const int *wedges,
                             const int *hexahedra,
                             const uint64_t numTetrahedra,
                             const uint64_t numPyramids,
                             const uint64_t numWedges,
                             const uint64_t numHexahedra,
                             const uint64_t numVertices
                             )
  {
    const uint64_t blockID
      = blockIdx.x
      + blockIdx.y * MAX_GRID_SIZE;
    uint64_t primIdx = blockID*blockDim.x + threadIdx.x;
    
    uint64_t pyrOffset = numTetrahedra;
    uint64_t wedOffset = numTetrahedra + numPyramids;
    uint64_t hexOffset = numTetrahedra + numPyramids + numWedges;
    uint64_t totalElements = numTetrahedra + numPyramids + numWedges + numHexahedra;
    if (primIdx >= totalElements) return;   // temporary 

    box4f primBounds4 = box4f();

    // if (primIdx < 0) return;
    // else 
    if (primIdx < pyrOffset) {
      int i0 = tetrahedra[primIdx * 4ull + 0ull];
      int i1 = tetrahedra[primIdx * 4ull + 1ull];
      int i2 = tetrahedra[primIdx * 4ull + 2ull];
      int i3 = tetrahedra[primIdx * 4ull + 3ull];
      // if (i0 < 0 || i0 >= numVertices) return;
      // if (i1 < 0 || i1 >= numVertices) return;
      // if (i2 < 0 || i2 >= numVertices) return;
      // if (i3 < 0 || i3 >= numVertices) return;

      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
    } 
    else if (primIdx < wedOffset) {
      primIdx -= pyrOffset;
      int i0 = pyramids[primIdx * 5ull + 0ull];
      int i1 = pyramids[primIdx * 5ull + 1ull];
      int i2 = pyramids[primIdx * 5ull + 2ull];
      int i3 = pyramids[primIdx * 5ull + 3ull];
      int i4 = pyramids[primIdx * 5ull + 4ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
    }
    else if (primIdx < hexOffset) {
      primIdx -= wedOffset;
      int i0 = wedges[primIdx * 6ull + 0ull];
      int i1 = wedges[primIdx * 6ull + 1ull];
      int i2 = wedges[primIdx * 6ull + 2ull];
      int i3 = wedges[primIdx * 6ull + 3ull];
      int i4 = wedges[primIdx * 6ull + 4ull];
      int i5 = wedges[primIdx * 6ull + 5ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
    }
    else {
      primIdx -= hexOffset;
      int i0 = hexahedra[primIdx * 8ull + 0ull];
      int i1 = hexahedra[primIdx * 8ull + 1ull];
      int i2 = hexahedra[primIdx * 8ull + 2ull];
      int i3 = hexahedra[primIdx * 8ull + 3ull];
      int i4 = hexahedra[primIdx * 8ull + 4ull];
      int i5 = hexahedra[primIdx * 8ull + 5ull];
      int i6 = hexahedra[primIdx * 8ull + 6ull];
      int i7 = hexahedra[primIdx * 8ull + 7ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
      primBounds4 = primBounds4.extend({vertices[i6].x, vertices[i6].y, vertices[i6].z, scalars[i6]} );
      primBounds4 = primBounds4.extend({vertices[i7].x, vertices[i7].y, vertices[i7].z, scalars[i7]} );
    }

    rasterBox(d_mcGrid,dims,worldBounds,primBounds4);
  }

  __global__ void rasterElements(float2 *d_mcGrid,
                             const vec3i dims,
                             const box3f worldBounds,
                             const vec3f *vertices,
                             const float *scalars,
                             const int *tetrahedra,
                             const int *pyramids,
                             const int *wedges,
                             const int *hexahedra,
                             const uint64_t numTetrahedra,
                             const uint64_t numPyramids,
                             const uint64_t numWedges,
                             const uint64_t numHexahedra,
                             const uint64_t numVertices,
                             const size_t numElements = 1
                             )
  {
    const uint64_t blockID
      = blockIdx.x
      + blockIdx.y * MAX_GRID_SIZE;
    uint64_t primIdx = blockID*blockDim.x + threadIdx.x;
    
    uint64_t pyrOffset = numTetrahedra;
    uint64_t wedOffset = numTetrahedra + numPyramids;
    uint64_t hexOffset = numTetrahedra + numPyramids + numWedges;
    uint64_t totalElements = numTetrahedra + numPyramids + numWedges + numHexahedra;
    if (primIdx >= totalElements) return;   // temporary 

    box4f primBounds4 = box4f();

    if (primIdx < pyrOffset) {
      int i0 = tetrahedra[primIdx * 4ull + 0ull];
      int i1 = tetrahedra[primIdx * 4ull + 1ull];
      int i2 = tetrahedra[primIdx * 4ull + 2ull];
      int i3 = tetrahedra[primIdx * 4ull + 3ull];

      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
    } 
    else if (primIdx < wedOffset) {
      primIdx -= pyrOffset;
      int i0 = pyramids[primIdx * 5ull + 0ull];
      int i1 = pyramids[primIdx * 5ull + 1ull];
      int i2 = pyramids[primIdx * 5ull + 2ull];
      int i3 = pyramids[primIdx * 5ull + 3ull];
      int i4 = pyramids[primIdx * 5ull + 4ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
    }
    else if (primIdx < hexOffset) {
      primIdx -= wedOffset;
      int i0 = wedges[primIdx * 6ull + 0ull];
      int i1 = wedges[primIdx * 6ull + 1ull];
      int i2 = wedges[primIdx * 6ull + 2ull];
      int i3 = wedges[primIdx * 6ull + 3ull];
      int i4 = wedges[primIdx * 6ull + 4ull];
      int i5 = wedges[primIdx * 6ull + 5ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
    }
    else {
      primIdx -= hexOffset;
      int i0 = hexahedra[primIdx * 8ull + 0ull];
      int i1 = hexahedra[primIdx * 8ull + 1ull];
      int i2 = hexahedra[primIdx * 8ull + 2ull];
      int i3 = hexahedra[primIdx * 8ull + 3ull];
      int i4 = hexahedra[primIdx * 8ull + 4ull];
      int i5 = hexahedra[primIdx * 8ull + 5ull];
      int i6 = hexahedra[primIdx * 8ull + 6ull];
      int i7 = hexahedra[primIdx * 8ull + 7ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
      primBounds4 = primBounds4.extend({vertices[i6].x, vertices[i6].y, vertices[i6].z, scalars[i6]} );
      primBounds4 = primBounds4.extend({vertices[i7].x, vertices[i7].y, vertices[i7].z, scalars[i7]} );
    }

    rasterBox(d_mcGrid,dims,worldBounds,primBounds4, 0, numElements);//!!
  }

  OWLBuffer Renderer::buildObjectOrientedMacrocells(const vec3i &dims, const box3f &bounds) {
    OWLBuffer BBoxBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(box4f), numClusters, nullptr);
    box4f *d_mcGrid = (box4f*)owlBufferGetPointer(BBoxBuffer, 0);

    const vec3i blockSize = vec3i(4);
    clearMCs<<<to_dims(divRoundUp(dims,blockSize)),to_dims(blockSize)>>>
      (d_mcGrid,dims);
    CUDA_SYNC_CHECK();

    const vec3f *d_vertices = (const vec3f*)owlBufferGetPointer(verticesData,0);
    const float *d_scalars = (const float*)owlBufferGetPointer(scalarData[0],0);//!!
    const int *d_tetrahedra = (const int*)owlBufferGetPointer(tetrahedraData,0);
    const int *d_pyramids = (const int*)owlBufferGetPointer(pyramidsData,0);
    const int *d_wedges = (const int*)owlBufferGetPointer(wedgesData,0);
    const int *d_hexahedra = (const int*)owlBufferGetPointer(hexahedraData,0);
    {
      const int blockSize = 32;

      const int numBlocks = divRoundUp(int(umeshPtrs[0]->numVolumeElements()), blockSize);
      vec3i grid(min(numBlocks,MAX_GRID_SIZE),
                  divRoundUp(numBlocks,MAX_GRID_SIZE),
                  1);

      clusterElementsIntoGrid<<<to_dims(grid),blockSize>>>
        (d_mcGrid,dims,bounds, d_vertices, d_scalars, d_tetrahedra, d_pyramids, d_wedges, d_hexahedra, 
          umeshPtrs[0]->tets.size(), umeshPtrs[0]->pyrs.size(), umeshPtrs[0]->wedges.size(), umeshPtrs[0]->hexes.size(), umeshPtrs[0]->vertices.size());

    }
    CUDA_SYNC_CHECK();
    return BBoxBuffer;
  }

  OWLBuffer Renderer::buildSpatialMacrocells(const vec3i &dims, const box3f &bounds) {
    uint32_t numMacrocells = dims.x * dims.y * dims.z;
    size_t numMeshes = meshType != MeshType::UNDEFINED ? (meshType == MeshType::UMESH ?umeshPtrs.size() : rawPtrs.size()) : 0;
    if(numMeshes == 0) throw std::runtime_error("No mesh data found");

    OWLBuffer MacrocellBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(float2), 
            numMacrocells*numMeshes, nullptr);
    float2 *d_mcGrid = (float2*)owlBufferGetPointer(MacrocellBuffer, 0);

    const vec3i blockSize = vec3i(4);
    sizeMCs<<<to_dims(divRoundUp(dims,blockSize)),to_dims(blockSize)>>>
      (d_mcGrid,dims,bounds,numMeshes);
    CUDA_SYNC_CHECK();

    printf("Building Spatial Macrocells\n");

    
    if(meshType == MeshType::UMESH)
    {
      const vec3f *d_vertices = (const vec3f*)owlBufferGetPointer(verticesData,0);
      const float *d_scalars = (const float*)owlBufferGetPointer(scalarData[0],0);//!!
      const int *d_tetrahedra = (const int*)owlBufferGetPointer(tetrahedraData,0);
      const int *d_pyramids = (const int*)owlBufferGetPointer(pyramidsData,0);
      const int *d_wedges = (const int*)owlBufferGetPointer(wedgesData,0);
      const int *d_hexahedra = (const int*)owlBufferGetPointer(hexahedraData,0);
      const int blockSize = 32;

      const int numBlocks = divRoundUp(int(umeshPtrs[0]->numVolumeElements()), blockSize);
      vec3i grid(min(numBlocks,MAX_GRID_SIZE),
                  divRoundUp(numBlocks,MAX_GRID_SIZE),
                  1);
      rasterElements<<<to_dims(grid),blockSize>>>
        (d_mcGrid,dims,bounds, d_vertices, d_scalars, d_tetrahedra, d_pyramids, d_wedges, d_hexahedra, 
          umeshPtrs[0]->tets.size(), umeshPtrs[0]->pyrs.size(), umeshPtrs[0]->wedges.size(), umeshPtrs[0]->hexes.size(),
          umeshPtrs[0]->vertices.size(), numMeshes);
    }
    else if (meshType == MeshType::RAW)
    {
      for (size_t i = 0; i < numMeshes; i++)
      {
        const float *d_scalars = (const float*)owlBufferGetPointer(scalarData[i],0);
        const int blockSize = 32;
        const vec3i vxlGridDims = rawPtrs[i]->getDims(); 
        const int elementCount = vxlGridDims.x * vxlGridDims.y * vxlGridDims.z;
        const int numBlocks = divRoundUp(elementCount, blockSize);
        vec3i grid(min(numBlocks,MAX_GRID_SIZE),
                    divRoundUp(numBlocks,MAX_GRID_SIZE),
                    1);

        rasterElements<<<to_dims(grid),blockSize>>>
          (d_mcGrid,dims,bounds,d_scalars,vxlGridDims,i,numMeshes);
        CUDA_SYNC_CHECK();
      }
      
    }
    CUDA_SYNC_CHECK();
    return MacrocellBuffer;
  }
  
  __global__ void computeCentroidsAndIndices(
    float4 *centroids,
    uint32_t *indices,
    const vec3f *vertices,
    const float *scalars,
    const int *tetrahedra,
    const int *pyramids,
    const int *wedges,
    const int *hexahedra,
    const uint64_t numTetrahedra,
    const uint64_t numPyramids,
    const uint64_t numWedges,
    const uint64_t numHexahedra
    )
  {
    const uint64_t blockID
      = blockIdx.x
      + blockIdx.y * MAX_GRID_SIZE;
    uint64_t index = blockID*blockDim.x + threadIdx.x;
    uint64_t primIdx = blockID*blockDim.x + threadIdx.x;
    
    uint64_t pyrOffset = numTetrahedra;
    uint64_t wedOffset = numTetrahedra + numPyramids;
    uint64_t hexOffset = numTetrahedra + numPyramids + numWedges;
    uint64_t totalElements = numTetrahedra + numPyramids + numWedges + numHexahedra;
    if (primIdx >= totalElements) return;

    box4f primBounds4 = box4f();
    if (primIdx < pyrOffset) {
      uint64_t i0 = tetrahedra[primIdx * 4ull + 0ull];
      uint64_t i1 = tetrahedra[primIdx * 4ull + 1ull];
      uint64_t i2 = tetrahedra[primIdx * 4ull + 2ull];
      uint64_t i3 = tetrahedra[primIdx * 4ull + 3ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
    } 
    else if (primIdx < wedOffset) {
      primIdx -= pyrOffset;
      uint64_t i0 = pyramids[primIdx * 5ull + 0ull];
      uint64_t i1 = pyramids[primIdx * 5ull + 1ull];
      uint64_t i2 = pyramids[primIdx * 5ull + 2ull];
      uint64_t i3 = pyramids[primIdx * 5ull + 3ull];
      uint64_t i4 = pyramids[primIdx * 5ull + 4ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
    }
    else if (primIdx < hexOffset) {
      primIdx -= wedOffset;
      uint64_t i0 = wedges[primIdx * 6ull + 0ull];
      uint64_t i1 = wedges[primIdx * 6ull + 1ull];
      uint64_t i2 = wedges[primIdx * 6ull + 2ull];
      uint64_t i3 = wedges[primIdx * 6ull + 3ull];
      uint64_t i4 = wedges[primIdx * 6ull + 4ull];
      uint64_t i5 = wedges[primIdx * 6ull + 5ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
    }
    else {
      primIdx -= hexOffset;
      uint64_t i0 = hexahedra[primIdx * 8ull + 0ull];
      uint64_t i1 = hexahedra[primIdx * 8ull + 1ull];
      uint64_t i2 = hexahedra[primIdx * 8ull + 2ull];
      uint64_t i3 = hexahedra[primIdx * 8ull + 3ull];
      uint64_t i4 = hexahedra[primIdx * 8ull + 4ull];
      uint64_t i5 = hexahedra[primIdx * 8ull + 5ull];
      uint64_t i6 = hexahedra[primIdx * 8ull + 6ull];
      uint64_t i7 = hexahedra[primIdx * 8ull + 7ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
      primBounds4 = primBounds4.extend({vertices[i6].x, vertices[i6].y, vertices[i6].z, scalars[i6]} );
      primBounds4 = primBounds4.extend({vertices[i7].x, vertices[i7].y, vertices[i7].z, scalars[i7]} );
    }

    float4 pt = make_float4(
      (primBounds4.upper.x + primBounds4.lower.x) * .5f,
      (primBounds4.upper.y + primBounds4.lower.y) * .5f,
      (primBounds4.upper.z + primBounds4.lower.z) * .5f,
      (primBounds4.upper.w + primBounds4.lower.w) * .5f);
    
    // printf("Centroid %f %f %f %f\n", pt.x, pt.y, pt.z, pt.w/*, code*/);

    centroids[index] = pt;
    indices[index] = index;
  }

  __global__ void computeCentroidBounds(const float4* centroids, uint64_t N, box4f* centroidBoundsPtr)
  {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: check if this might become a bottleneck
    if (index < N)
    {
      float4 pt = centroids[index];
      box4f& centroidBounds = *centroidBoundsPtr;

      // printf("Reading Centroid %f %f %f %f\n", pt.x, pt.y, pt.z, pt.w/*, code*/);
      // printf("Old Bounds %f %f %f %f\n", centroidBounds.upper.x, centroidBounds.upper.y, centroidBounds.upper.z, centroidBounds.upper.w/*, code*/);
      
      // this assumes for an anisotropic bound, we still want isotropic bounding boxes.
      // this should improve on surface area of the splits...
      float mn = min(pt.x, min(pt.y, pt.z));
      float mx = max(pt.x, max(pt.y, pt.z));
      atomicMin(&centroidBounds.lower.x, mn);
      atomicMin(&centroidBounds.lower.y, mn);
      atomicMin(&centroidBounds.lower.z, mn);
      atomicMin(&centroidBounds.lower.w, pt.w);
      atomicMax(&centroidBounds.upper.x, mx);
      atomicMax(&centroidBounds.upper.y, mx);
      atomicMax(&centroidBounds.upper.z, mx);
      atomicMax(&centroidBounds.upper.w, pt.w);


      // atomicMin(&centroidBounds.lower.x, pt.x);
      // atomicMax(&centroidBounds.upper.x, pt.x);
      // atomicMin(&centroidBounds.lower.y, pt.y);
      // atomicMax(&centroidBounds.upper.y, pt.y);
      // atomicMin(&centroidBounds.lower.z, pt.z);
      // atomicMax(&centroidBounds.upper.z, pt.z);
      // atomicMin(&centroidBounds.lower.w, pt.w);
      // atomicMax(&centroidBounds.upper.w, pt.w);

      // printf("New Bounds %f %f %f %f\n", centroidBounds.upper.x, centroidBounds.upper.y, centroidBounds.upper.z, centroidBounds.upper.w/*, code*/);

    }
  }

  __host__ __device__
  inline uint64_t morton64_encode4D(uint64_t x, uint64_t y, uint64_t z, uint64_t w)            
  {                                                                                          
      auto separate_bits = [](uint64_t n)                                                    
      {                                                                                      
          n &= 0b1111111111111111ull;                                                        
          n = (n ^ (n << 24)) & 0b0000000000000000000000001111111100000000000000000000000011111111ull;
          n = (n ^ (n << 12)) & 0b0000000000001111000000000000111100000000000011110000000000001111ull;
          n = (n ^ (n <<  6)) & 0b0000001100000011000000110000001100000011000000110000001100000011ull;
          n = (n ^ (n <<  3)) & 0b0001000100010001000100010001000100010001000100010001000100010001ull;
                                                                                             
          return n;                                                                          
      };                                                                                     
                                                                                             
      uint64_t xb = separate_bits(x);                                                        
      uint64_t yb = separate_bits(y) << 1;                                                   
      uint64_t zb = separate_bits(z) << 2;                                                   
      uint64_t wb = separate_bits(w) << 3;
      uint64_t code = xb | yb | zb | wb;                                                     
                                                                                             
      return code;                                                                           
  } 

  __host__ __device__
  inline uint64_t morton64_encode3D(float x, float y, float z)
  {
    x = x * (float)(1 << 16);
    y = y * (float)(1 << 16);
    z = z * (float)(1 << 16);
    auto separate_bits = [](uint64_t n)
    {
        n &= 0b1111111111111111111111ull;
        n = (n ^ (n << 32)) & 0b1111111111111111000000000000000000000000000000001111111111111111ull;
        n = (n ^ (n << 16)) & 0b0000000011111111000000000000000011111111000000000000000011111111ull;
        n = (n ^ (n <<  8)) & 0b1111000000001111000000001111000000001111000000001111000000001111ull;
        n = (n ^ (n <<  4)) & 0b0011000011000011000011000011000011000011000011000011000011000011ull;
        n = (n ^ (n <<  2)) & 0b1001001001001001001001001001001001001001001001001001001001001001ull;
        return n;
    };  

    return separate_bits(x) | (separate_bits(y) << 1) | (separate_bits(z) << 2); 
  }

    __host__ __device__
  inline uint64_t hilbert64_encode3D(float x, float y, float z)
  {
    x = x * (float)(1 << 16);
    y = y * (float)(1 << 16);
    z = z * (float)(1 << 16);
    const bitmask_t coord[3] = {bitmask_t(x), bitmask_t(y), bitmask_t(z)};
    return hilbert_c2i(3, 16, coord);
  }


  #include "owl/common/math/random.h"
  typedef owl::common::LCG<4> Random;
    __host__ __device__
  inline uint64_t random_encode3D(float x, float y, float z)
  {
    Random random(x, y * z * (1ull<<31ull));
    return random() * (1ull<<63ull);
  }


  __global__ void assignCodes(uint64_t* codes,
                                  const float4* centroids,
                                  unsigned N,
                                  box4f* centroidBoundsPtr)
  {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
    {
      const box4f& centroidBounds = *centroidBoundsPtr;

      // Project node to [0..1]
      float4 pt = centroids[index];
      float4 pt0 = pt;

      pt -= centroidBounds.center();
      pt = (pt + centroidBounds.size() * .5f) / centroidBounds.size();

      // edge case where all elements fall on the same plane, or have the same max data value
      if (centroidBounds.size().x == 0.f) pt.x = 0.f;
      if (centroidBounds.size().y == 0.f) pt.y = 0.f;
      if (centroidBounds.size().z == 0.f) pt.z = 0.f;
      if (centroidBounds.size().w == 0.f) pt.w = 0.f;

      // turns out including data value causes overlap which is really bad for performance...
      // uint64_t code = morton64_encode4D((uint64_t)pt.x, (uint64_t)pt.y, (uint64_t)pt.z, (uint64_t)pt.w);
      // uint64_t code = morton64_encode3D(pt.x, pt.y, pt.z);
      uint64_t code = hilbert64_encode3D(pt.x, pt.y, pt.z);
      // uint64_t code = random_encode3D(pt.x, pt.y, pt.z); // BAD, just a test...      
      codes[index] = code;
    }
  }

  struct VertexValue {
    uint32_t address;
    float3 position;  
    float scalar;  
  };

  __global__ void fillVertexValues(
    VertexValue* vertices,
    const vec3f* positions,
    const float* scalars,
    unsigned N)
  {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;

    vertices[index] = {
      index,
      positions[index],
      scalars[index]
    };
  }

  __global__ void fillVertexValues(
    vec3f* positions,
    float* scalars,
    const VertexValue* vertices,
    uint64_t N)
  {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;
    positions[index] = vertices[index].position;
    scalars[index] = vertices[index].scalar;
  }

  __global__ void assignVertexMortonCodes(
    uint64_t* codes,
    const VertexValue* vertices,
    unsigned N,
    box3f bounds)
  {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N) return;    

    // Project node to [0..1]
    float3 pt = vertices[index].position;

    pt -= bounds.center();
    pt = (pt + bounds.size() * .5f) / bounds.size();

    // edge case where all elements fall on the same plane, or have the same max data value
    if (bounds.size().x == 0.f) pt.x = 0.f;
    if (bounds.size().y == 0.f) pt.y = 0.f;
    if (bounds.size().z == 0.f) pt.z = 0.f;

    // Quantize to 16 bit
    pt = min(max(pt * 65536.f, make_float3(0.f)), make_float3(65535.f));
    uint64_t code = morton64_encode3D((uint64_t)pt.x, (uint64_t)pt.y, (uint64_t)pt.z);
    codes[index] = code;
  }

  __global__ void markClusters( const uint64_t* codes,
                                uint64_t N,
                                uint32_t* flags,
                                uint32_t bitsToDrop
                                )
  {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;   
    if (index >= N) return;
    uint64_t previousCode = (index == 0) ? -1 : codes[index-1];
    uint64_t baseCode = codes[index];
    // if our common prefix is less than the number of bits we're merging, 
    // this element is sufficiently different and  marks the beginning of a 
    // new cluster
    flags[index] = (index == 0) || ((baseCode >> bitsToDrop) != (previousCode >> bitsToDrop));
  }

  __global__ void mergeClusters(uint64_t numElements,
                                uint32_t numClusters,
                                uint32_t* flags,
                                const uint64_t* codes,
                                uint32_t* elementToCluster,
                                uint32_t* elementsInClusters,
                                int maxElementsPerCluster
                                )
  {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;   
    if (index >= numElements) return;

    // never clear the first cluster (should also always be an even cluster... optional)
    if (index == 0) return; 
    // if not the start of the cluster, return
    if (!flags[index]) return;
    uint32_t clusterID = elementToCluster[index];
    // return if we are not an odd cluster ID (odd's get merged into evens)
    if ((clusterID % 2) != 1) return;
    // figure out how many elements are in this cluster...
    uint32_t count = elementsInClusters[clusterID];
    // ... also how many elements were in the previous cluster...
    uint32_t prevCount = elementsInClusters[clusterID - 1];

    bool tooMany = count + prevCount > maxElementsPerCluster;
    if (tooMany) return;

    // combine this cluster with previous
    flags[index] = 0;
  }

  __global__ void setOne(uint32_t* numbers, uint64_t N)
  {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;   
    if (index >= N) return;
    numbers[index] = 1;
  }

  __global__ void subtractOne(uint32_t* numbers, uint64_t N)
  {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;   
    if (index >= N) return;
    numbers[index] = numbers[index] - 1;
  }

  __global__ void makeClusters( 
                                box4f* clusters,
                                uint32_t* elementToCluster,
                                uint32_t numClusters,
                                const uint32_t *elementIds,
                                const vec3f *vertices,
                                const float *scalars,
                                const int *tetrahedra,
                                const int *pyramids,
                                const int *wedges,
                                const int *hexahedra,
                                const uint64_t numTetrahedra,
                                const uint64_t numPyramids,
                                const uint64_t numWedges,
                                const uint64_t numHexahedra
                                )
  {
    uint32_t pyrOffset = numTetrahedra;
    uint32_t wedOffset = numTetrahedra + numPyramids;
    uint32_t hexOffset = numTetrahedra + numPyramids + numWedges;
    uint32_t totalElements = numTetrahedra + numPyramids + numWedges + numHexahedra;

    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalElements) return;

    uint32_t clusterIndex = elementToCluster[index];
    
    uint32_t primIdx = elementIds[index];
    box4f primBounds4 = box4f();
    if (primIdx < pyrOffset) {
      uint32_t i0 = tetrahedra[primIdx * 4ull + 0ull];
      uint32_t i1 = tetrahedra[primIdx * 4ull + 1ull];
      uint32_t i2 = tetrahedra[primIdx * 4ull + 2ull];
      uint32_t i3 = tetrahedra[primIdx * 4ull + 3ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
    } 
    else if (primIdx < wedOffset) {
      primIdx -= pyrOffset;
      uint32_t i0 = pyramids[primIdx * 5ull + 0ull];
      uint32_t i1 = pyramids[primIdx * 5ull + 1ull];
      uint32_t i2 = pyramids[primIdx * 5ull + 2ull];
      uint32_t i3 = pyramids[primIdx * 5ull + 3ull];
      uint32_t i4 = pyramids[primIdx * 5ull + 4ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
    }
    else if (primIdx < hexOffset) {
      primIdx -= wedOffset;
      uint32_t i0 = wedges[primIdx * 6ull + 0ull];
      uint32_t i1 = wedges[primIdx * 6ull + 1ull];
      uint32_t i2 = wedges[primIdx * 6ull + 2ull];
      uint32_t i3 = wedges[primIdx * 6ull + 3ull];
      uint32_t i4 = wedges[primIdx * 6ull + 4ull];
      uint32_t i5 = wedges[primIdx * 6ull + 5ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
    }
    else {
      primIdx -= hexOffset;
      uint32_t i0 = hexahedra[primIdx * 8ull + 0ull];
      uint32_t i1 = hexahedra[primIdx * 8ull + 1ull];
      uint32_t i2 = hexahedra[primIdx * 8ull + 2ull];
      uint32_t i3 = hexahedra[primIdx * 8ull + 3ull];
      uint32_t i4 = hexahedra[primIdx * 8ull + 4ull];
      uint32_t i5 = hexahedra[primIdx * 8ull + 5ull];
      uint32_t i6 = hexahedra[primIdx * 8ull + 6ull];
      uint32_t i7 = hexahedra[primIdx * 8ull + 7ull];
      primBounds4 = primBounds4.extend({vertices[i0].x, vertices[i0].y, vertices[i0].z, scalars[i0]} );
      primBounds4 = primBounds4.extend({vertices[i1].x, vertices[i1].y, vertices[i1].z, scalars[i1]} );
      primBounds4 = primBounds4.extend({vertices[i2].x, vertices[i2].y, vertices[i2].z, scalars[i2]} );
      primBounds4 = primBounds4.extend({vertices[i3].x, vertices[i3].y, vertices[i3].z, scalars[i3]} );
      primBounds4 = primBounds4.extend({vertices[i4].x, vertices[i4].y, vertices[i4].z, scalars[i4]} );
      primBounds4 = primBounds4.extend({vertices[i5].x, vertices[i5].y, vertices[i5].z, scalars[i5]} );
      primBounds4 = primBounds4.extend({vertices[i6].x, vertices[i6].y, vertices[i6].z, scalars[i6]} );
      primBounds4 = primBounds4.extend({vertices[i7].x, vertices[i7].y, vertices[i7].z, scalars[i7]} );
    }  

    atomicMin(&clusters[clusterIndex].lower.x,primBounds4.lower.x);
    atomicMax(&clusters[clusterIndex].upper.x,primBounds4.upper.x);
    atomicMin(&clusters[clusterIndex].lower.y,primBounds4.lower.y);
    atomicMax(&clusters[clusterIndex].upper.y,primBounds4.upper.y);
    atomicMin(&clusters[clusterIndex].lower.z,primBounds4.lower.z);
    atomicMax(&clusters[clusterIndex].upper.z,primBounds4.upper.z);
    atomicMin(&clusters[clusterIndex].lower.w,primBounds4.lower.w);
    atomicMax(&clusters[clusterIndex].upper.w,primBounds4.upper.w);
  }

  void Renderer::sortElements(uint64_t* &codesSorted, uint32_t* &elementIdsSorted)
  {
    unsigned numThreads = 1024;
    unsigned numElements = umeshPtrs[0]->numVolumeElements();

    const vec3f *d_vertices = (const vec3f*)owlBufferGetPointer(verticesData,0);
    const float *d_scalars = (const float*)owlBufferGetPointer(scalarData[0],0); //!!
    const int *d_tetrahedra = (const int*)owlBufferGetPointer(tetrahedraData,0);
    const int *d_pyramids = (const int*)owlBufferGetPointer(pyramidsData,0);
    const int *d_wedges = (const int*)owlBufferGetPointer(wedgesData,0);
    const int *d_hexahedra = (const int*)owlBufferGetPointer(hexahedraData,0);

    // one centroid per element
    float4* centroids;
    cudaMalloc((void**)&centroids, numElements * sizeof(float4));
    box4f* centroidBounds;
    box4f emptyBounds = box4f();
    cudaMalloc((void**)&centroidBounds, sizeof(box4f));
    cudaMemcpy(centroidBounds, &emptyBounds,sizeof(box4f),cudaMemcpyHostToDevice);

    {
      // make the host block until the device is finished with foo
      cudaDeviceSynchronize();

      // check for error
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error(cudaGetErrorString(error));
      }
    }

    // also a buffer of element indices we'll be sorting
    uint32_t* elementIdsUnsorted;
    cudaMalloc((void**)&elementIdsUnsorted, numElements * sizeof(uint32_t));
    cudaMalloc((void**)&elementIdsSorted, numElements * sizeof(uint32_t));

    // Compute element centroids and indices
    {
      computeCentroidsAndIndices<<<div_up(numElements, numThreads), numThreads>>>(
        centroids, elementIdsUnsorted, d_vertices, d_scalars, d_tetrahedra, d_pyramids, d_wedges, d_hexahedra, 
        umeshPtrs[0]->tets.size(), umeshPtrs[0]->pyrs.size(), umeshPtrs[0]->wedges.size(), umeshPtrs[0]->hexes.size()
      );
    }

    {
      // make the host block until the device is finished with foo
      cudaDeviceSynchronize();

      // check for error
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error(cudaGetErrorString(error));
      }
    }

    // Compute centroid bounds 
    {
      computeCentroidBounds<<<div_up(numElements, numThreads), numThreads>>>(
        centroids, numElements, centroidBounds
      );
      cudaMemcpy(&emptyBounds, centroidBounds,sizeof(box4f),cudaMemcpyDeviceToHost);
    }

    {
      // make the host block until the device is finished with foo
      cudaDeviceSynchronize();

      // check for error
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error(cudaGetErrorString(error));
      }
    }
    // std::cout<<morton64_encode4D(32017, 13940, 1081, 15089)<<std::endl;

    // Project on morton curve
    uint64_t* codesUnsorted;
    cudaMalloc((void**)&codesUnsorted, numElements * sizeof(uint64_t));
    cudaMalloc((void**)&codesSorted, numElements * sizeof(uint64_t));
    {      
      assignCodes<<<div_up(numElements, numThreads), numThreads>>>(
        codesUnsorted,
        centroids,
        numElements,
        centroidBounds);
    }

    cudaFree(centroids);
    cudaFree(centroidBounds);

    // std::vector<uint64_t> hCodesUnsorted(numElements);
    // cudaMemcpy(hCodesUnsorted.data(), codesUnsorted, sizeof(uint64_t) * numElements, cudaMemcpyDeviceToHost);

    // Determine temporary device storage requirements
    static size_t   oldN = 0;

    // Allocate temporary storage
    static void     *d_temp_storage = NULL;
    static size_t   temp_storage_bytes = 0;
    if (oldN < numElements) {
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            codesUnsorted, codesSorted, elementIdsUnsorted, elementIdsSorted, numElements);

        if (d_temp_storage != nullptr) cudaFree(d_temp_storage);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        oldN = numElements;
    }

    {
      // make the host block until the device is finished with foo
      cudaDeviceSynchronize();

      // check for error
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error(cudaGetErrorString(error));
      }
    }

    // sort on morton curve
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        codesUnsorted, codesSorted, elementIdsUnsorted, elementIdsSorted, numElements);

    cudaFree(codesUnsorted);
    cudaFree(elementIdsUnsorted);
    cudaFree(d_temp_storage);
  }

  void Renderer::buildClusters(
    const uint64_t* codesSorted, const uint32_t* elementIdsSorted,
    uint32_t maxNumClusters, uint32_t &numClusters, OWLBuffer &clustersBuffer, 
    bool returnSortedIndexToCluster, uint32_t* sortedIndexToCluster)
  {
    unsigned numThreads = 1024;
    unsigned numElements = umeshPtrs[0]->numVolumeElements();
    const vec3f *d_vertices = (const vec3f*)owlBufferGetPointer(verticesData,0);
    const float *d_scalars = (const float*)owlBufferGetPointer(scalarData[0],0);//!!
    const int *d_tetrahedra = (const int*)owlBufferGetPointer(tetrahedraData,0);
    const int *d_pyramids = (const int*)owlBufferGetPointer(pyramidsData,0);
    const int *d_wedges = (const int*)owlBufferGetPointer(wedgesData,0);
    const int *d_hexahedra = (const int*)owlBufferGetPointer(hexahedraData,0);

    // Mark cluster starts
    uint32_t* flags;
    cudaMalloc((void**)&flags, numElements * sizeof(uint32_t));
    setOne<<<div_up(numElements, numThreads), numThreads>>>(flags, numElements);
    if (!sortedIndexToCluster) cudaMalloc((void**)&sortedIndexToCluster, numElements * sizeof(uint32_t));

    uint32_t *uniqueSortedIndexToCluster;
    uint32_t *uniqueSortedIndexToClusterCount;
    uint32_t *numRuns;
    cudaMalloc((void**)&uniqueSortedIndexToCluster, numElements * sizeof(uint32_t));
    cudaMalloc((void**)&uniqueSortedIndexToClusterCount, numElements * sizeof(uint32_t));
    cudaMalloc((void**)&numRuns, sizeof(uint32_t));


    // going to try to merge clusters that are too small
    int prevNumClusters = 0;
    // for (int i = 0; i < 1000; ++i) 
    while (true)
    {

      // Postfix sum the flag and subtract one to compute cluster addresses
      {
        // Declare, allocate, and initialize device-accessible pointers for input and output
        uint64_t  num_items = numElements;      // e.g., 7
        uint32_t  *d_in = flags;          // e.g., [8, 6, 7, 5, 3, 0, 9]
        uint32_t  *d_out = sortedIndexToCluster;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
        // Determine temporary device storage requirements
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run exclusive prefix sum
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
        // subtract one to get addresses
        subtractOne<<<div_up(numElements, numThreads), numThreads>>>(sortedIndexToCluster, num_items);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);
      }

      // Compute run length encoding to determine next cluster from current
      {
        // Declare, allocate, and initialize device-accessible pointers for input and output
        uint32_t          num_items = numElements;          // e.g., 8
        uint32_t          *d_in = sortedIndexToCluster;              // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
        uint32_t          *d_unique_out = uniqueSortedIndexToCluster;      // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
        uint32_t          *d_counts_out = uniqueSortedIndexToClusterCount;      // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
        uint32_t          *d_num_runs_out = numRuns;    // e.g., [ ]
        // Determine temporary device storage requirements
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run encoding
        cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items);
        // d_unique_out      <-- [0, 2, 9, 5, 8]
        // d_counts_out      <-- [1, 2, 1, 3, 1]
        // d_num_runs_out    <-- [5]
      } 

      uint32_t hNumRuns;
      cudaMemcpy(&hNumRuns, numRuns, sizeof(uint32_t), cudaMemcpyDeviceToHost);

      // alternatively...
      numClusters = hNumRuns;
      if (prevNumClusters == numClusters) break; // can't merge any more
      prevNumClusters = numClusters;

      // unmark cluster flags where too many elements exist in a cluster
      {
        mergeClusters<<<div_up(numElements, numThreads), numThreads>>>(
          numElements,
          hNumRuns,
          flags,
          codesSorted,
          sortedIndexToCluster,
          uniqueSortedIndexToClusterCount,
          numElements // here, we don't care how many elements go in a cluster. just for adaptive sampling...
        ); 
        cudaDeviceSynchronize();
      }
      
      // cudaMemcpy(&numClusters,(sortedIndexToCluster + (numElements - 1)),sizeof(uint32_t),cudaMemcpyDeviceToHost);
      // numClusters += 1; // account for subtract by one.

      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error(cudaGetErrorString(error));
      }

      if (numClusters < maxNumClusters) break;
      
      std::cout<<"generated " << numClusters << " clusters... merging..." << std::endl;
    }
    std::cout<<"done..." << std::endl;

    cudaFree(uniqueSortedIndexToCluster);
    cudaFree(uniqueSortedIndexToClusterCount);
    
    clustersBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(box4f), numClusters, nullptr);
    box4f* clusters = (box4f*)owlBufferGetPointer(clustersBuffer, 0);

    // fill in clusters
    {
      clearClusters<<<div_up(numClusters, numThreads), numThreads>>>(clusters, numClusters);

      makeClusters<<<div_up(numElements, numThreads), numThreads>>>(
        clusters,
        sortedIndexToCluster,
        numClusters,
        elementIdsSorted,
        d_vertices, d_scalars, d_tetrahedra, d_pyramids, d_wedges, d_hexahedra, 
        umeshPtrs[0]->tets.size(), umeshPtrs[0]->pyrs.size(), umeshPtrs[0]->wedges.size(), umeshPtrs[0]->hexes.size()
      ); 
    }

    cudaDeviceSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      throw std::runtime_error(cudaGetErrorString(error));
    }

    if (!returnSortedIndexToCluster) cudaFree(sortedIndexToCluster);
  }

  void Renderer::buildClusters(
    uint32_t coarseClusterPrecision, uint32_t mediumClusterPrecision, uint32_t fineClusterPrecision, 
    uint32_t &numCoarseClusters, OWLBuffer &coarseClusters, 
    uint32_t &numMediumClusters, OWLBuffer &mediumClusters, 
    uint32_t &numFineClusters, OWLBuffer &fineClusters)
  {

    uint64_t* codesSorted;
    uint32_t* elementIdsSorted;
    sortElements(codesSorted, elementIdsSorted);
    buildClusters(codesSorted, elementIdsSorted, coarseClusterPrecision, numCoarseClusters, coarseClusters);
    buildClusters(codesSorted, elementIdsSorted, mediumClusterPrecision, numMediumClusters, mediumClusters);
    buildClusters(codesSorted, elementIdsSorted, fineClusterPrecision, numFineClusters, fineClusters);
    cudaFree(codesSorted);
    cudaFree(elementIdsSorted);
  }

  void Renderer::buildClusters(
    uint32_t maxNumClusters, uint32_t &numClusters, OWLBuffer &clusters)
  {
    uint64_t* codesSorted;
    uint32_t* elementIdsSorted;
    sortElements(codesSorted, elementIdsSorted);
    buildClusters(codesSorted, elementIdsSorted, maxNumClusters, numClusters, clusters);
    cudaFree(codesSorted);
    cudaFree(elementIdsSorted);
  }


  __global__ void countElements( 
                                uint32_t* sortedIndexToCluster,
                                uint32_t numClusters,
                                uint32_t *tetrahedraInCluster,
                                uint32_t *wedgesInCluster,
                                uint32_t *pyramidsInCluster,
                                uint32_t *hexahedraInCluster,
                                const uint32_t *elementIds,
                                const uint64_t numTetrahedra,
                                const uint64_t numPyramids,
                                const uint64_t numWedges,
                                const uint64_t numHexahedra
                                )
  {
    uint32_t pyrOffset = numTetrahedra;
    uint32_t wedOffset = numTetrahedra + numPyramids;
    uint32_t hexOffset = numTetrahedra + numPyramids + numWedges;
    uint32_t totalElements = numTetrahedra + numPyramids + numWedges + numHexahedra;

    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalElements) return;
    uint32_t primIdx = elementIds[index];    
    uint32_t clusterIndex = sortedIndexToCluster[index];    

    if (primIdx < pyrOffset) {
      atomicAdd(&tetrahedraInCluster[clusterIndex], 1);
    } else if (primIdx < wedOffset) {
      atomicAdd(&pyramidsInCluster[clusterIndex], 1);
    } else if (primIdx < hexOffset) {
      atomicAdd(&wedgesInCluster[clusterIndex], 1);
    } else {
      atomicAdd(&hexahedraInCluster[clusterIndex], 1);
    }
  }

  __global__ void markElementTypes( 
                                uint32_t elementsInCluster,
                                const uint32_t *elementIds,
                                const uint64_t numTetrahedra,
                                const uint64_t numPyramids,
                                const uint64_t numWedges,
                                const uint64_t numHexahedra,
                                uint8_t *elementTypes
                                )
  {
    uint32_t pyrOffset = numTetrahedra;
    uint32_t wedOffset = numTetrahedra + numPyramids;
    uint32_t hexOffset = numTetrahedra + numPyramids + numWedges;
    uint32_t totalElements = numTetrahedra + numPyramids + numWedges + numHexahedra;

    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elementsInCluster) return;
    uint32_t primIdx = elementIds[index];

    if (primIdx < pyrOffset) {
      elementTypes[index] = 0;
    } else if (primIdx < wedOffset) {
      elementTypes[index] = 1;
    } else if (primIdx < hexOffset) {
      elementTypes[index] = 2;
    } else {
      elementTypes[index] = 3;
    }
  }

  __global__
  void markUsed(
    uint8_t* isUsed,
    uint32_t* clusterElementIds,
    uint32_t totalElementsInCluster,
    const int *tetrahedra,
    const int *pyramids,
    const int *wedges,
    const int *hexahedra,
    const uint64_t numTetrahedra,
    const uint64_t numPyramids,
    const uint64_t numWedges,
    const uint64_t numHexahedra,
    const uint64_t numVertices)
  {
    uint32_t tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= totalElementsInCluster) return;

    uint32_t pyrOffset = numTetrahedra;
    uint32_t wedOffset = numTetrahedra + numPyramids;
    uint32_t hexOffset = numTetrahedra + numPyramids + numWedges;
    uint32_t totalElements = numTetrahedra + numPyramids + numWedges + numHexahedra;

    uint32_t primIdx = clusterElementIds[tid];
    if (primIdx < pyrOffset) {
      isUsed[tetrahedra[primIdx * 4ull + 0ull]] = true;
      isUsed[tetrahedra[primIdx * 4ull + 1ull]] = true;
      isUsed[tetrahedra[primIdx * 4ull + 2ull]] = true;
      isUsed[tetrahedra[primIdx * 4ull + 3ull]] = true;
    } 
    else if (primIdx < wedOffset) {
      primIdx -= pyrOffset;
      isUsed[pyramids[primIdx * 5ull + 0ull]] = true;
      isUsed[pyramids[primIdx * 5ull + 1ull]] = true;
      isUsed[pyramids[primIdx * 5ull + 2ull]] = true;
      isUsed[pyramids[primIdx * 5ull + 3ull]] = true;
      isUsed[pyramids[primIdx * 5ull + 4ull]] = true;
    }
    else if (primIdx < hexOffset) {
      primIdx -= wedOffset;
      isUsed[wedges[primIdx * 6ull + 0ull]] = true;
      isUsed[wedges[primIdx * 6ull + 1ull]] = true;
      isUsed[wedges[primIdx * 6ull + 2ull]] = true;
      isUsed[wedges[primIdx * 6ull + 3ull]] = true;
      isUsed[wedges[primIdx * 6ull + 4ull]] = true;
      isUsed[wedges[primIdx * 6ull + 5ull]] = true;
    }
    else {
      primIdx -= hexOffset;
      isUsed[hexahedra[primIdx * 8ull + 0ull]] = true;
      isUsed[hexahedra[primIdx * 8ull + 1ull]] = true;
      isUsed[hexahedra[primIdx * 8ull + 2ull]] = true;
      isUsed[hexahedra[primIdx * 8ull + 3ull]] = true;
      isUsed[hexahedra[primIdx * 8ull + 4ull]] = true;
      isUsed[hexahedra[primIdx * 8ull + 5ull]] = true;
      isUsed[hexahedra[primIdx * 8ull + 6ull]] = true;
      isUsed[hexahedra[primIdx * 8ull + 7ull]] = true;
    }
  }

  __global__
  void replaceUnused(
    uint8_t* isUsed,
    uint32_t numVertices,
    VertexValue *vertices,
    uint32_t *elementIDs,
    uint64_t *vertexCodes,
    const int *tetrahedra,
    const int *pyramids,
    const int *wedges,
    const int *hexahedra,
    const uint64_t numTetrahedra,
    const uint64_t numPyramids,
    const uint64_t numWedges,
    const uint64_t numHexahedra)
  {
    uint32_t tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numVertices) return;

    uint32_t vertAddr;
    if (isUsed[tid]) vertAddr = tid;
    else {
      uint32_t primID = elementIDs[0];
      if (numTetrahedra > 0) vertAddr = tetrahedra[primID * 4];
      else if (numPyramids) vertAddr = pyramids[primID * 5];
      else if (numWedges) vertAddr = wedges[primID * 6];
      else vertAddr = hexahedra[primID * 8];
    } 

    vertices[tid] = vertices[vertAddr];
    vertexCodes[tid] = vertexCodes[vertAddr];

  }

  __global__
  void setNoDup(
    uint32_t *noDup, 
    uint64_t *vtxCode, 
    uint32_t num
  ) {
    uint32_t tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid < num)
      noDup[tid]
        = (tid == 0)
        ? 1
        : (vtxCode[tid] != vtxCode[tid-1]);
  }

  __global__
  void scatterVertices(
    uint32_t *noDup, 
    uint32_t *newIdx, 
    VertexValue *newVertices,
    VertexValue *oldVertices,
    uint32_t numVertices
  ) {
    uint32_t tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numVertices) return;
    if (!noDup[tid]) return;
    newVertices[newIdx[tid]] = oldVertices[tid];
  }

  __global__
  void setPerm(uint32_t *perm,
              VertexValue *newVertices,
              uint32_t numVertices)
  {
    uint32_t tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numVertices) return;
    perm[newVertices[tid].address] = tid;
  }

  __global__
  void translateVertices(
    const int *tetrahedra,
    const int *pyramids,
    const int *wedges,
    const int *hexahedra,
    uint32_t numTetrahedra,
    uint32_t numPyramids,
    uint32_t numWedges,
    uint32_t numHexahedra,
    uint32_t numVertices,
    void *clusterTetrahedra,
    void *clusterPyramids,
    void *clusterWedges,
    void *clusterHexahedra,
    uint32_t newNumTetrahedra,
    uint32_t newNumPyramids,
    uint32_t newNumWedges,
    uint32_t newNumHexahedra,
    uint32_t newNumVertices,
    uint32_t *clusterElementIds,
    uint32_t *perm,
    uint32_t totalElementsInCluster,
    uint8_t bytesPerIndex
  )
  {
    uint32_t tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= totalElementsInCluster) return;

    uint64_t oldPyrOffset = numTetrahedra;
    uint64_t oldWedOffset = numTetrahedra + numPyramids;
    uint64_t oldHexOffset = numTetrahedra + numPyramids + numWedges;
    
    uint64_t newPyrOffset = newNumTetrahedra;
    uint64_t newWedOffset = newNumTetrahedra + newNumPyramids;
    uint64_t newHexOffset = newNumTetrahedra + newNumPyramids + newNumWedges;
    
    // idea here, perm is a lookup table that, given an old vert index, returns the new index.
    uint64_t oldPrimIdx = clusterElementIds[tid];
    uint64_t newPrimIdx = tid;

    if (newPrimIdx < newPyrOffset) {
      if (bytesPerIndex == 1) {
        uint8_t* ctets = (uint8_t*)clusterTetrahedra;
        ctets[newPrimIdx * 4ull + 0ull] = perm[tetrahedra[oldPrimIdx * 4ull + 0ull]];
        ctets[newPrimIdx * 4ull + 1ull] = perm[tetrahedra[oldPrimIdx * 4ull + 1ull]];
        ctets[newPrimIdx * 4ull + 2ull] = perm[tetrahedra[oldPrimIdx * 4ull + 2ull]];
        ctets[newPrimIdx * 4ull + 3ull] = perm[tetrahedra[oldPrimIdx * 4ull + 3ull]];
      } else if (bytesPerIndex == 2) {
        uint16_t* ctets = (uint16_t*)clusterTetrahedra;
        ctets[newPrimIdx * 4ull + 0ull] = perm[tetrahedra[oldPrimIdx * 4ull + 0ull]];
        ctets[newPrimIdx * 4ull + 1ull] = perm[tetrahedra[oldPrimIdx * 4ull + 1ull]];
        ctets[newPrimIdx * 4ull + 2ull] = perm[tetrahedra[oldPrimIdx * 4ull + 2ull]];
        ctets[newPrimIdx * 4ull + 3ull] = perm[tetrahedra[oldPrimIdx * 4ull + 3ull]];
      } else if (bytesPerIndex == 4) {
        uint32_t* ctets = (uint32_t*)clusterTetrahedra;
        ctets[newPrimIdx * 4ull + 0ull] = perm[tetrahedra[oldPrimIdx * 4ull + 0ull]];
        ctets[newPrimIdx * 4ull + 1ull] = perm[tetrahedra[oldPrimIdx * 4ull + 1ull]];
        ctets[newPrimIdx * 4ull + 2ull] = perm[tetrahedra[oldPrimIdx * 4ull + 2ull]];
        ctets[newPrimIdx * 4ull + 3ull] = perm[tetrahedra[oldPrimIdx * 4ull + 3ull]];
      }
    } 
    else if (newPrimIdx < newWedOffset) {
      oldPrimIdx -= oldPyrOffset;
      newPrimIdx -= newPyrOffset;
      if (bytesPerIndex == 1) {
        uint8_t* cpyrs = (uint8_t*)clusterPyramids;
        cpyrs[newPrimIdx * 5ull + 0ull] = perm[pyramids[oldPrimIdx * 5ull + 0ull]];
        cpyrs[newPrimIdx * 5ull + 1ull] = perm[pyramids[oldPrimIdx * 5ull + 1ull]];
        cpyrs[newPrimIdx * 5ull + 2ull] = perm[pyramids[oldPrimIdx * 5ull + 2ull]];
        cpyrs[newPrimIdx * 5ull + 3ull] = perm[pyramids[oldPrimIdx * 5ull + 3ull]];
        cpyrs[newPrimIdx * 5ull + 4ull] = perm[pyramids[oldPrimIdx * 5ull + 4ull]];
      } else if (bytesPerIndex == 2) {
        uint16_t* cpyrs = (uint16_t*)clusterPyramids;
        cpyrs[newPrimIdx * 5ull + 0ull] = perm[pyramids[oldPrimIdx * 5ull + 0ull]];
        cpyrs[newPrimIdx * 5ull + 1ull] = perm[pyramids[oldPrimIdx * 5ull + 1ull]];
        cpyrs[newPrimIdx * 5ull + 2ull] = perm[pyramids[oldPrimIdx * 5ull + 2ull]];
        cpyrs[newPrimIdx * 5ull + 3ull] = perm[pyramids[oldPrimIdx * 5ull + 3ull]];
        cpyrs[newPrimIdx * 5ull + 4ull] = perm[pyramids[oldPrimIdx * 5ull + 4ull]];
      } else if (bytesPerIndex == 4) {
        uint32_t* cpyrs = (uint32_t*)clusterPyramids;
        cpyrs[newPrimIdx * 5ull + 0ull] = perm[pyramids[oldPrimIdx * 5ull + 0ull]];
        cpyrs[newPrimIdx * 5ull + 1ull] = perm[pyramids[oldPrimIdx * 5ull + 1ull]];
        cpyrs[newPrimIdx * 5ull + 2ull] = perm[pyramids[oldPrimIdx * 5ull + 2ull]];
        cpyrs[newPrimIdx * 5ull + 3ull] = perm[pyramids[oldPrimIdx * 5ull + 3ull]];
        cpyrs[newPrimIdx * 5ull + 4ull] = perm[pyramids[oldPrimIdx * 5ull + 4ull]];
      }
    }
    else if (newPrimIdx < newHexOffset) {
      oldPrimIdx -= oldWedOffset;
      newPrimIdx -= newWedOffset;
      if (bytesPerIndex == 1) {
        uint8_t* cwedges = (uint8_t*)clusterWedges;
        cwedges[newPrimIdx * 6ull + 0ull] = perm[wedges[oldPrimIdx * 6ull + 0ull]];
        cwedges[newPrimIdx * 6ull + 1ull] = perm[wedges[oldPrimIdx * 6ull + 1ull]];
        cwedges[newPrimIdx * 6ull + 2ull] = perm[wedges[oldPrimIdx * 6ull + 2ull]];
        cwedges[newPrimIdx * 6ull + 3ull] = perm[wedges[oldPrimIdx * 6ull + 3ull]];
        cwedges[newPrimIdx * 6ull + 4ull] = perm[wedges[oldPrimIdx * 6ull + 4ull]];
        cwedges[newPrimIdx * 6ull + 5ull] = perm[wedges[oldPrimIdx * 6ull + 5ull]];
      } else if (bytesPerIndex == 2) {
        uint16_t* cwedges = (uint16_t*)clusterWedges;
        cwedges[newPrimIdx * 6ull + 0ull] = perm[wedges[oldPrimIdx * 6ull + 0ull]];
        cwedges[newPrimIdx * 6ull + 1ull] = perm[wedges[oldPrimIdx * 6ull + 1ull]];
        cwedges[newPrimIdx * 6ull + 2ull] = perm[wedges[oldPrimIdx * 6ull + 2ull]];
        cwedges[newPrimIdx * 6ull + 3ull] = perm[wedges[oldPrimIdx * 6ull + 3ull]];
        cwedges[newPrimIdx * 6ull + 4ull] = perm[wedges[oldPrimIdx * 6ull + 4ull]];
        cwedges[newPrimIdx * 6ull + 5ull] = perm[wedges[oldPrimIdx * 6ull + 5ull]];
      } else if (bytesPerIndex == 4) {
        uint32_t* cwedges = (uint32_t*)clusterWedges;
        cwedges[newPrimIdx * 6ull + 0ull] = perm[wedges[oldPrimIdx * 6ull + 0ull]];
        cwedges[newPrimIdx * 6ull + 1ull] = perm[wedges[oldPrimIdx * 6ull + 1ull]];
        cwedges[newPrimIdx * 6ull + 2ull] = perm[wedges[oldPrimIdx * 6ull + 2ull]];
        cwedges[newPrimIdx * 6ull + 3ull] = perm[wedges[oldPrimIdx * 6ull + 3ull]];
        cwedges[newPrimIdx * 6ull + 4ull] = perm[wedges[oldPrimIdx * 6ull + 4ull]];
        cwedges[newPrimIdx * 6ull + 5ull] = perm[wedges[oldPrimIdx * 6ull + 5ull]];
      }    
    }
    else {
      oldPrimIdx -= oldHexOffset;
      newPrimIdx -= newHexOffset;  
      if (bytesPerIndex == 1) {
        uint8_t* chexes = (uint8_t*)clusterHexahedra;
        chexes[newPrimIdx * 8ull + 0ull] = perm[hexahedra[oldPrimIdx * 8ull + 0ull]];
        chexes[newPrimIdx * 8ull + 1ull] = perm[hexahedra[oldPrimIdx * 8ull + 1ull]];
        chexes[newPrimIdx * 8ull + 2ull] = perm[hexahedra[oldPrimIdx * 8ull + 2ull]];
        chexes[newPrimIdx * 8ull + 3ull] = perm[hexahedra[oldPrimIdx * 8ull + 3ull]];
        chexes[newPrimIdx * 8ull + 4ull] = perm[hexahedra[oldPrimIdx * 8ull + 4ull]];
        chexes[newPrimIdx * 8ull + 5ull] = perm[hexahedra[oldPrimIdx * 8ull + 5ull]];
        chexes[newPrimIdx * 8ull + 6ull] = perm[hexahedra[oldPrimIdx * 8ull + 6ull]];
        chexes[newPrimIdx * 8ull + 7ull] = perm[hexahedra[oldPrimIdx * 8ull + 7ull]];
      } else if (bytesPerIndex == 2) {
        uint16_t* chexes = (uint16_t*)clusterHexahedra;
        chexes[newPrimIdx * 8ull + 0ull] = perm[hexahedra[oldPrimIdx * 8ull + 0ull]];
        chexes[newPrimIdx * 8ull + 1ull] = perm[hexahedra[oldPrimIdx * 8ull + 1ull]];
        chexes[newPrimIdx * 8ull + 2ull] = perm[hexahedra[oldPrimIdx * 8ull + 2ull]];
        chexes[newPrimIdx * 8ull + 3ull] = perm[hexahedra[oldPrimIdx * 8ull + 3ull]];
        chexes[newPrimIdx * 8ull + 4ull] = perm[hexahedra[oldPrimIdx * 8ull + 4ull]];
        chexes[newPrimIdx * 8ull + 5ull] = perm[hexahedra[oldPrimIdx * 8ull + 5ull]];
        chexes[newPrimIdx * 8ull + 6ull] = perm[hexahedra[oldPrimIdx * 8ull + 6ull]];
        chexes[newPrimIdx * 8ull + 7ull] = perm[hexahedra[oldPrimIdx * 8ull + 7ull]];
      } else if (bytesPerIndex == 4) {
        uint32_t* chexes = (uint32_t*)clusterHexahedra;
        chexes[newPrimIdx * 8ull + 0ull] = perm[hexahedra[oldPrimIdx * 8ull + 0ull]];
        chexes[newPrimIdx * 8ull + 1ull] = perm[hexahedra[oldPrimIdx * 8ull + 1ull]];
        chexes[newPrimIdx * 8ull + 2ull] = perm[hexahedra[oldPrimIdx * 8ull + 2ull]];
        chexes[newPrimIdx * 8ull + 3ull] = perm[hexahedra[oldPrimIdx * 8ull + 3ull]];
        chexes[newPrimIdx * 8ull + 4ull] = perm[hexahedra[oldPrimIdx * 8ull + 4ull]];
        chexes[newPrimIdx * 8ull + 5ull] = perm[hexahedra[oldPrimIdx * 8ull + 5ull]];
        chexes[newPrimIdx * 8ull + 6ull] = perm[hexahedra[oldPrimIdx * 8ull + 6ull]];
        chexes[newPrimIdx * 8ull + 7ull] = perm[hexahedra[oldPrimIdx * 8ull + 7ull]];
      }    
    }
  }

  __global__
  void generateSequence(
    uint32_t* numbers,
    uint32_t totalNumbers
  )
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= totalNumbers) return;
    numbers[tid] = tid;
  }

  static const char *humanSize(uint64_t bytes)
  {
    char *suffix[] = {"B", "KB", "MB", "GB", "TB"};
    char length = sizeof(suffix) / sizeof(suffix[0]);

    int i = 0;
    double dblBytes = bytes;

    if (bytes > 1024) {
      for (i = 0; (bytes / 1024) > 0 && i<length-1; i++, bytes /= 1024)
        dblBytes = bytes / 1024.0;
    }

    static char output[200];
    sprintf(output, "%.02lf %s", dblBytes, suffix[i]);
    return output;
  }
};


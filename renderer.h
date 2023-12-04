#pragma once

#include <iostream>

#include "camera.h"

#include "owl/owl.h"
#include "owl/owl_host.h"
#include "umesh/io/UMesh.h"

#include "deviceCode.h"

#include "rawFile.h"

namespace dtracker
{
enum MeshType
{
  UNDEFINED,
  UMESH = 1,
  RAW = 2
};
class Renderer
{
private:
public:
    Renderer();
    ~Renderer();

    camera::Camera camera;
    std::shared_ptr<umesh::UMesh> umeshPtr;
    std::vector<std::shared_ptr<raw::RawR>> rawPtrs;

    /* raygen */
    OWLRayGen rayGen{0};
    OWLParams lp{0};

    /* owl */
    OWLContext context{0};
    OWLModule module;

    /* geom */
    std::vector<OWLGeom> elementGeom;
    std::vector<OWLGroup> elementBLAS;
    OWLGroup elementTLAS;
    OWLGroup rootMacrocellBLAS;
    OWLGroup rootMacrocellTLAS;

    OWLGeom trianglesGeom;
    OWLGroup trianglesGroup;
    OWLGroup triangleTLAS   { 0 };

    OWLGeomType macrocellType;
    OWLGeomType tetrahedraType;
    OWLGeomType pyramidType;
    OWLGeomType wedgeType;
    OWLGeomType hexahedraType;

    OWLGeomType triangleType;

    OWLBuffer tetrahedraData;
    OWLBuffer pyramidsData;
    OWLBuffer hexahedraData;
    OWLBuffer wedgesData;
    OWLBuffer verticesData;
    OWLBuffer scalarData;
    OWLBuffer gridBuffer;
    OWLBuffer majorantBuffer;
    OWLBuffer vertexBuffer;
    OWLBuffer indexBuffer;

    OWLBuffer macrocellsBuffer;
    OWLBuffer rootBBoxBuffer;
    OWLBuffer rootMaximaBuffer;
    OWLBuffer gridMaximaBuffer;
    OWLBuffer clusterMaximaBuffer;

    OWLBuffer clusterBBoxBuffer;

    /* frame */
    OWLBuffer accumBuffer{0};
    OWLBuffer frameBuffer{0};
    int accumID{0};
    int frameID{0};

    /* timing */
    float avgTime = 0.0f;
    float minTime = std::numeric_limits<float>::max();;
    float totalTime = 0.0f;

    /* scene */
    float dt = 0.05f;
    bool enableShadows = false;
    bool enableAccumulation = true;
    vec3f lightDir = vec3f(0.0001f, -1.f, 0.0001f);
    float lightIntensity = 1.0f;
    float ambient = 0.1f;
    vec2i fbSize = vec2i(1024, 1024);
    vec3f bgColor = vec3f(0.0f, 0.0f, 0.0f);
    MeshType meshType = MeshType::UNDEFINED;

    /* transfer function */
    OWLBuffer colorMapBuffer { 0 };
    cudaArray_t colorMapArray { 0 };
    cudaTextureObject_t colorMapTexture { 0 };

    interval<float> volDomain;
    interval<float> xfDomain;
    std::vector<vec4f> colorMap;
    float opacityScale = 1.0f;

    /* density majorants */
    uint32_t numClusters = 1;
    unsigned macrocellsPerSide = 32; // 4096 exceeds the size of a uint32_t when squared...

    /*! initializes renderer */
    void Init(bool autoSetCamera = true);
    /*! renders the scene, visualizes heatmaps if param is true*/
    void Render(bool heatMap = false);
    /*! updates the launch params on device */
    void Update();
    /*! terminates the renderer*/
    void Terminate();
    /*! resizes the frame buffer */
    void Resize(vec2i newSize);
    /*! updates camera at device*/
    void UpdateCamera();
    /*! sets the colormap texture on device*/
    void SetXFColormap(std::vector<vec4f> newCM);
    /*! sets opacity scale*/
    void SetXFOpacityScale(float opacityScale);
    /*! sets the transfer function domain*/
    void SetXFRange(vec2f xfDomain);
    /*! recalculates the majorants*/
    void RecalculateDensityRanges();
    /*! sets the dt to avg span /2 of bounding boxes of elements*/
    void ResetDt();
    /*! sets the dt to a fixed value*/
    void SetDt(float dt);
    /*! sets the light direction*/
    void SetLightDirection(vec3f lightDir);
    /*! sets the light intensity*/
    void SetLightIntensity(float lightIntensity);
    /*! sets the ambient intensity*/
    void SetAmbient(float ambient);
    /*reset accumilation*/
    void ResetAccumulation();

    // For umeshes, use this to generate some object oriented macrocells.
    // idea, rasterize elements into a grid, expand or shrink bounding boxes to contain elements,
    // avoid making macrocells in empty regions. Empty macrocells will have negative volume.
    OWLBuffer buildObjectOrientedMacrocells(const vec3i &dims, const box3f &bounds);

    // For umeshes, use this to generate some non-overlapping macrocells.
    // idea, rasterize elements into a grid, fixing the size of the macrocells to the grid,
    // potentially generating macrocells in empty regions
    OWLBuffer buildSpatialMacrocells(const vec3i &dims, const box3f &bounds);

    void computeRanges(OWLBuffer &ranges);
    void sortElements(uint64_t* &codesSorted, uint32_t* &elementIdsSorted);
    void buildClusters(const uint64_t* codesSorted, const uint32_t* elementIDsSorted,
      uint32_t clusterPrecision, uint32_t &numClusters, OWLBuffer &clusters, 
      bool returnSortedIndexToCluster = false, uint32_t* sortedIndexToCluster = nullptr); 
    void buildClusters(
      uint32_t maxNumClusters, uint32_t &numClusters, OWLBuffer &clusters);
    void buildClusters(
      uint32_t coarseClusterPrecision, uint32_t mediumClusterPrecision, uint32_t fineClusterPrecision, 
      uint32_t &numCoarseClusters, OWLBuffer &coarseClusters, 
      uint32_t &numMediumClusters, OWLBuffer &mediumClusters, 
      uint32_t &numFineClusters, OWLBuffer &fineClusters);

};

} // namespace dtracker
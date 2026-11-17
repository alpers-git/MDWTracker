#pragma once

#include <array>
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

enum Mode
{
  BASELINE = 0,
  MULTI = 1,
  MULTI_ALT = 2,
  CUMMULATIVE = 3,
  MAX = 4,
  MIX = 5,
  MARCHER_MULTI = 6,
  MARCHER_MAX = 7,
  MARCHER_MIX = 8,
  MARCHER_COMP = 9,
};

/* transfer function */
struct TFData
{
  OWLBuffer colorMapBuffer{0};
  cudaArray_t colorMapArray{0};
  cudaTextureObject_t colorMapTexture{0};

  interval<float> volDomain;
  interval<float> xfDomain = interval<float>({0.f, 1.f});
  std::vector<vec4f> colorMap;
  int numTexels = 0;
  float opacityScale = 1.0f;
};

/* N-dimensional transfer function (--tf-dim N > 1), host side of TransferFunctionND */
struct TFDataND
{
  cudaArray_t colorMapArray{0};
  cudaTextureObject_t colorMapTexture{0};

  // numTexels[0] * numTexels[1] * ... RGBA texels, axis 0 (value) varies fastest
  std::vector<vec4f> colorMap;
  int numTexels[MAX_TF_DIM] = {};
  // data-space range of each axis: 0 = value, 1 = gradient magnitude
  interval<float> domain[MAX_TF_DIM];
  float opacityScale = 1.0f;

  TransferFunctionND toDevice() const
  {
    TransferFunctionND d{};
    d.xf = colorMapTexture;
    for (int a = 0; a < MAX_TF_DIM; ++a)
    {
      d.domain[a] = make_float2(domain[a].lower, domain[a].upper);
      d.numTexels[a] = numTexels[a];
    }
    d.opacityScale = opacityScale;
    return d;
  }
};

class Renderer
{
private:
public:
    Renderer();
    ~Renderer();

    camera::Camera camera;
    std::vector<std::shared_ptr<umesh::UMesh>> umeshPtrs;
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

    OWLBuffer tetrahedraData; //Unstructured mesh elements
    OWLBuffer pyramidsData;   //Unstructured mesh elements
    OWLBuffer hexahedraData;  //Unstructured mesh elements
    OWLBuffer wedgesData;     //Unstructured mesh elements
    OWLBuffer verticesData;   //Unstructured mesh vertices
    OWLBuffer scalarData[MAX_CHANNELS];
    OWLBuffer gridBuffer;     
    OWLBuffer majorantBuffer; //this will account for all macrocell grids
    OWLBuffer vertexBuffer;   //triangle vertices
    OWLBuffer indexBuffer;    //triangle indices

    OWLBuffer macrocellsBuffer; //TODO N macrocell buffers for N meshes
    OWLBuffer rootBBoxBuffer;   // used for RootMacrocellTLAS
    OWLBuffer rootMaximaBuffer; // used for RootMacrocellTLAS
    OWLBuffer gridMaximaBuffer; 

    /* frame */
    OWLBuffer accumBuffer{0};
    OWLBuffer frameBuffer{0};
    int accumID{0};
    int frameID{0};
    int spp = 1;

    /* timing */
    float avgTime = 0.0f;
    float minTime = std::numeric_limits<float>::max();
    float totalTime = 0.0f;
    float heatMapScale = 1.0f;

    /* scene */
    float dt = 0.05f;
    bool enableShadows = false;
    bool enableGradientShading = false;
    bool enableAccumulation = true;
    vec3f lightDir = vec3f(0.0001f, -1.f, 0.0001f);
    float lightIntensity = 1.0f;
    float ambient = 0.1f;
    vec2i fbSize = vec2i(1024, 1024);
    vec3f bgColor = vec3f(0.0f, 0.0f, 0.0f);
    MeshType meshType = MeshType::UNDEFINED;

    std::vector<TFData> tfdatas;

    int tfDim = 1; // --tf-dim, set before Init
    std::vector<TFDataND> tfdatasND; // one per channel, only filled when tfDim > 1

    /* material properties per channel */
    struct MaterialData
    {
        float diffuse = 0.8f;     // Diffuse coefficient [0,1]
        float specular = 0.3f;    // Specular coefficient [0,1] 
        float shininess = 8.0f;   // Specular exponent [1,128]
        float gradientStep = 0.01f; // Gradient calculation step multiplier [0.001,0.1]
        vec3f specularColor = vec3f(1.0f, 1.0f, 1.0f); // RGB color for specular highlights
    };
    std::vector<MaterialData> materials;

    /* density majorants */
    uint32_t numClusters = 1;
    vec3ui macrocellDims = {0,0,0}; // 4096 exceeds the size of a uint32_t when squared...
    Mode mode = Mode::CUMMULATIVE;

    /*! initializes renderer */
    void Init(const unsigned int mode = 0, bool autoSetCamera = true);
  /*! renders the scene, visualizes heatmaps if param is true*/
  void Render(short heatMapMode= 0);

  // Framebuffer mirroring flag
  bool mirrorFramebuffer = false;
    /*! updates the launch params on device */
    void Update();
    /*! terminates the renderer*/
    void Terminate();
    /*! pushes a umesh*/
    bool PushMesh(std::shared_ptr<umesh::UMesh> umeshPtr);
    /*! pushes a raw file*/
    bool PushMesh(std::shared_ptr<raw::RawR> rawPtr);
    /*! resizes the frame buffer */
    void Resize(vec2i newSize);
    /*! updates camera at device*/
    void UpdateCamera();
    /*! sets the colormap texture on device*/
    void SetXFColormap(std::vector<vec4f> newCM, size_t tfID = 0);
    /*! sets opacity scale*/
    void SetXFOpacityScale(float opacityScale, size_t tfID = 0);
    /*! sets the transfer function domain*/
    void SetXFRange(vec2f xfDomain, size_t tfID = 0);
    /*! N-D transfer function (tfDim > 1): uploads the RGBA lookup table, axis 0 fastest */
    void SetXFColormapND(const std::vector<vec4f> &texels,
                         const std::array<int, MAX_TF_DIM> &numTexels, size_t tfID = 0);
    /*! N-D transfer function: sets the data-space range of one axis (0 = value, 1 = gradient magnitude) */
    void SetXFDomainND(int axis, vec2f domain, size_t tfID = 0);
    /*! N-D transfer function: sets the opacity scale */
    void SetXFOpacityScaleND(float opacityScale, size_t tfID = 0);
    /*! sets material properties for a channel*/
    void SetMaterialDiffuse(float diffuse, size_t channelID = 0);
    void SetMaterialSpecular(float specular, size_t channelID = 0);
    void SetMaterialShininess(float shininess, size_t channelID = 0);
    void SetMaterialGradientStep(float gradientStep, size_t channelID = 0);
    void SetMaterialSpecularColor(vec3f specularColor, size_t channelID = 0);
    /*! applies material preset to a channel*/
    void ApplyMaterialPreset(const std::string& presetName, size_t channelID = 0);
    /*! recalculates the majorants*/
    void RecalculateDensityRanges();
    /*! sets the globalOpacity/dt to avg span /2 of bounding boxes of elements*/
    void Resetdt();
    /*! sets the globalOpacity/dt to a fixed value*/
    void Setdt(float dt);
    /*! sets the light direction*/
    void SetLightDirection(vec3f lightDir);
    /*! sets the light intensity*/
    void SetLightIntensity(float lightIntensity);
    /*! sets the ambient intensity*/
    void SetAmbient(float ambient);
    /*reset accumilation*/
    void ResetAccumulation();
    /*!estimate best mcgrid sizes given the mesh*/
    vec3ui CalculateMCGridDims(int estimatedElementPerMc);

    // For structured (RAW) volumes, generates non-overlapping macrocells.
    // idea, rasterize voxels into a grid, fixing the size of the macrocells to the grid,
    // storing a per-channel scalar range in each cell.
    OWLBuffer buildSpatialMacrocells(const vec3i &dims, const box3f &bounds);

    // N-D transfer function (tfDim > 1) versions, dispatching on tfDim to the kernels in renderer.cu
    OWLBuffer buildSpatialMacrocellsND(const vec3i &dims, const box3f &bounds);
    void RecalculateDensityRangesND();
    /*! gradient magnitude of every stride-th voxel of raw channel channelID, as the 2D TF sees it */
    std::vector<float> VoxelGradientMagnitudes(size_t channelID, size_t stride = 1) const;

private:
    /*! name of the OptiX program to use for the current tfDim: <name> or <name>_tf<tfDim> */
    std::string TFProgramName(const std::string &name) const;
    /*! pushes tfdatasND to the transferFunctionND launch params */
    void UploadTFParamsND();
};

} // namespace dtracker
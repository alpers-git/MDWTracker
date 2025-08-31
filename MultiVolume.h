// CompressedMultiChannelVolume: manages multi-channel volume data with diff compression
#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include "rawFile.h"

struct ChannelInfo {
        raw::DataFormat type;
        owl::box4f bounds;
        raw::Vec3l dims;
        float diffMin;  // minimum diff value for decompression offset
        float diffMax;  // maximum diff value for decompression scaling
    };

class MultiVolume {
public:
    // Construct from a vector of RawR pointers (channels)
    MultiVolume(const std::vector<std::shared_ptr<raw::RawR>>& channels);

    // Get decompressed data for channel N (returns float vector)
    std::vector<float> getDecompressedChannel(size_t channelIdx) const;

    // Template function to get compressed data based on format and type
    template<typename T>
    std::vector<T> getCompressedChannelData(size_t channelIdx, raw::DataFormat expectedFormat) const;

    // Get compressed type for channel N
    raw::DataFormat getCompressedType(size_t channelIdx) const;

    // Get number of channels
    size_t numChannels() const { return channelInfo.size(); }

    // Get dimensions (assume all channels have same dims)
    raw::Vec3l getDims(size_t channelIdx = 0) const;

    // Get compressed volume bounds
    owl::box4f getBounds(size_t channelIdx = 0) const;

    // Get channel information
    ChannelInfo getChannelInfo(size_t channelIdx) const;

    // Get base channel data (for GPU reconstruction)
    std::vector<float> getBaseChannelData() const;

    // Get global bounds
    owl::box4f getGlobalBounds() const;

    // Is compression enabled?
    bool isCompressed() const { return compressed; }
    // Compresses channels and deletes original data, using targetBits for mantissa
    void compressChannels(int targetBits = 32);
    void decompressChannels();
private:
    std::vector<std::shared_ptr<raw::RawR>> channels_; // original channels
    std::vector<float> baseChannelData_; // base channel data (stored after compression)
    // std::vector<CompressedType> compressedTypes_;      // type per channel
    std::vector<std::vector<uint8_t>> compressedDiffs8_;   // for uint8 diffs
    std::vector<std::vector<uint16_t>> compressedDiffs16_;  // for uint16 diffs
    std::vector<std::vector<uint32_t>> compressedDiffs32_;  // for uint32 diffs
    std::vector<std::vector<float>> compressedDiffsFloat_; // for float diffs

    bool compressed = false;
    owl::box4f globalBounds;
    std::vector<ChannelInfo> channelInfo;
    
};

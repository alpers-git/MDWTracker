// CompressedMultiChannelVolume: manages multi-channel volume data with diff compression
#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include "rawFile.h"

// Enum for supported compressed types
enum class CompressedType {
    Int8,
    Int16,
    Int32,
    Float32
};

class MultiVolume {
public:
    // Construct from a vector of RawR pointers (channels)
    MultiVolume(const std::vector<std::shared_ptr<raw::RawR>>& channels);

    // Get decompressed data for channel N (returns float vector)
    std::vector<float> getDecompressedChannel(size_t channelIdx) const;

    // Get compressed type for channel N
    CompressedType getCompressedType(size_t channelIdx) const;

    // Get number of channels
    size_t numChannels() const { return channelInfo.size(); }

    // Get dimensions (assume all channels have same dims)
    raw::Vec3l getDims(size_t channelIdx = 0) const;

    // Get compressed volume bounds
    owl::box4f getBounds4f(size_t channelIdx = 0) const;

    // Get global bounds
    owl::box4f getGlobalBounds4f() const;

    // Is compression enabled?
    bool isCompressed() const { return compressed; }

    // Compresses channels and deletes original data
    void compressChannels();

    // Decompresses all channels and restores original data
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

    struct ChannelInfo {
        CompressedType type;
        owl::box4f bounds;
        raw::Vec3l dims;
        float diffMin;  // minimum diff value for decompression offset
        float diffMax;  // maximum diff value for decompression scaling
    };
    std::vector<ChannelInfo> channelInfo;
    
};

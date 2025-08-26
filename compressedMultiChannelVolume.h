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

class CompressedMultiChannelVolume {
public:
    // Construct from a vector of RawR pointers (channels), with optional compression
    CompressedMultiChannelVolume(const std::vector<std::shared_ptr<raw::RawR>>& channels, bool compress = true);

    // Get decompressed data for channel N (returns float vector)
    std::vector<float> getChannel(size_t channelIdx) const;

    // Get compressed type for channel N
    CompressedType getCompressedType(size_t channelIdx) const;

    // Get number of channels
    size_t numChannels() const { return channels_.size(); }

    // Get dimensions (assume all channels have same dims)
    raw::Vec3l getDims() const;

    // Get compressed volume bounds
    owl::box4f getBounds4f() const;

    // Is compression enabled?
    bool isCompressed() const { return compress_; }

private:
    std::vector<std::shared_ptr<raw::RawR>> channels_; // original channels
    std::vector<CompressedType> compressedTypes_;      // type per channel
    std::vector<std::vector<uint8_t>> compressedDiffs8_;   // for int8 diffs
    std::vector<std::vector<int16_t>> compressedDiffs16_;  // for int16 diffs
    std::vector<std::vector<int32_t>> compressedDiffs32_;  // for int32 diffs
    std::vector<std::vector<float>> compressedDiffsFloat_; // for float diffs

    bool compress_ = true;
    owl::box4f bounds_;

    // Helper: compress all channels
    void compressChannels();
};

// Next: implement the constructor and compression logic in a .cpp file.

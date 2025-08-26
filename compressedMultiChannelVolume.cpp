#include "compressedMultiChannelVolume.h"
#include <algorithm>
#include <limits>
#include <cassert>

CompressedMultiChannelVolume::CompressedMultiChannelVolume(const std::vector<std::shared_ptr<raw::RawR>>& channels, bool compress)
    : channels_(channels), compress_(compress)
{
    assert(!channels.empty());
    bounds_ = channels[0]->getBounds4f();
    for (size_t i = 1; i < channels.size(); i++) {
        bounds_.extend(channels[i]->getBounds4f());
    }
    if (compress_)
        compressChannels();
}

raw::Vec3l CompressedMultiChannelVolume::getDims() const {
    return channels_[0]->getDims();
}

CompressedType CompressedMultiChannelVolume::getCompressedType(size_t channelIdx) const {
    return compressedTypes_[channelIdx];
}

owl::box4f CompressedMultiChannelVolume::getBounds4f() const {
    return bounds_;
}

std::vector<float> CompressedMultiChannelVolume::getChannel(size_t channelIdx) const {
    if (!compress_ || channelIdx == 0) {
        return channels_[channelIdx]->getDataVector();
    }
    // decompress: channelN = channel0 - diffN
    const auto& base = channels_[0]->getDataVector();
    std::vector<float> result(base.size());
    switch (compressedTypes_[channelIdx]) {
        case CompressedType::Int8:
            for (size_t i = 0; i < base.size(); ++i)
                result[i] = base[i] - static_cast<float>(compressedDiffs8_[channelIdx-1][i]);
            break;
        case CompressedType::Int16:
            for (size_t i = 0; i < base.size(); ++i)
                result[i] = base[i] - static_cast<float>(compressedDiffs16_[channelIdx-1][i]);
            break;
        case CompressedType::Int32:
            for (size_t i = 0; i < base.size(); ++i)
                result[i] = base[i] - static_cast<float>(compressedDiffs32_[channelIdx-1][i]);
            break;
        case CompressedType::Float32:
            for (size_t i = 0; i < base.size(); ++i)
                result[i] = base[i] - compressedDiffsFloat_[channelIdx-1][i];
            break;
    }
    return result;
}

void CompressedMultiChannelVolume::compressChannels() {
    printf("Compressing multi-channel volume with %zu channels\n", channels_.size());
    const auto& base = channels_[0]->getDataVector();
    size_t numVoxels = base.size();
    size_t numChannels = channels_.size();
    compressedTypes_.resize(numChannels);
    compressedTypes_[0] = CompressedType::Float32; // base channel is uncompressed
    compressedDiffs8_.resize(numChannels-1);
    compressedDiffs16_.resize(numChannels-1);
    compressedDiffs32_.resize(numChannels-1);
    compressedDiffsFloat_.resize(numChannels-1);
    for (size_t c = 1; c < numChannels; ++c) {
        const auto& data = channels_[c]->getDataVector();
        std::vector<float> diff(numVoxels);
        float minDiff = std::numeric_limits<float>::max();
        float maxDiff = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < numVoxels; ++i) {
            diff[i] = base[i] - data[i];
            minDiff = std::min(minDiff, diff[i]);
            maxDiff = std::max(maxDiff, diff[i]);
        }
        // Choose smallest type
        if (minDiff >= std::numeric_limits<int8_t>::min() && maxDiff <= std::numeric_limits<int8_t>::max()) {
            compressedTypes_[c] = CompressedType::Int8;
            compressedDiffs8_[c-1].resize(numVoxels);
            for (size_t i = 0; i < numVoxels; ++i)
                compressedDiffs8_[c-1][i] = static_cast<int8_t>(diff[i]);
            printf("Compressed channel %zu as int8_t (range: %.2f to %.2f)\n", c, minDiff, maxDiff);
        } else if (minDiff >= std::numeric_limits<int16_t>::min() && maxDiff <= std::numeric_limits<int16_t>::max()) {
            compressedTypes_[c] = CompressedType::Int16;
            compressedDiffs16_[c-1].resize(numVoxels);
            for (size_t i = 0; i < numVoxels; ++i)
                compressedDiffs16_[c-1][i] = static_cast<int16_t>(diff[i]);
            printf("Compressed channel %zu as int16_t (range: %.2f to %.2f)\n", c, minDiff, maxDiff);
        } else if (minDiff >= std::numeric_limits<int32_t>::min() && maxDiff <= std::numeric_limits<int32_t>::max()) {
            compressedTypes_[c] = CompressedType::Int32;
            compressedDiffs32_[c-1].resize(numVoxels);
            for (size_t i = 0; i < numVoxels; ++i)
                compressedDiffs32_[c-1][i] = static_cast<int32_t>(diff[i]);
            printf("Compressed channel %zu as int32_t (range: %.2f to %.2f)\n", c, minDiff, maxDiff);
        } else {
            compressedTypes_[c] = CompressedType::Float32;
            compressedDiffsFloat_[c-1] = std::move(diff);
            printf("Compressed channel %zu as float (range: %.2f to %.2f)\n", c, minDiff, maxDiff);
        }
    }
}

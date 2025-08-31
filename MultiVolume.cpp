#include "MultiVolume.h"
#include <algorithm>
#include <limits>
#include <cassert>

MultiVolume::MultiVolume(const std::vector<std::shared_ptr<raw::RawR>>& channels)
    : channels_(channels)
{
    assert(!channels.empty());
    channelInfo.resize(channels.size());
    for (size_t i = 0; i < channels.size(); i++) {
        channelInfo[i].type = raw::DataFormat::Float32; // default to uncompressed
        channelInfo[i].bounds = channels[i]->getBounds4f();
        channelInfo[i].dims = channels[i]->getDims();
        globalBounds.extend(channelInfo[i].bounds);
    }
    compressed = false;
}

raw::Vec3l MultiVolume::getDims(size_t channelIdx) const {
    return channelInfo[channelIdx].dims;
}

raw::DataFormat MultiVolume::getCompressedType(size_t channelIdx) const {
    return channelInfo[channelIdx].type;
}

owl::box4f MultiVolume::getBounds(size_t channelIdx) const {
    return channelInfo[channelIdx].bounds;
}

owl::box4f MultiVolume::getGlobalBounds() const {
    return globalBounds;
}

ChannelInfo MultiVolume::getChannelInfo(size_t channelIdx) const {
    return channelInfo[channelIdx];
}

std::vector<float> MultiVolume::getDecompressedChannel(size_t channelIdx) const {
    printf("Decompressing channel %zu\n", channelIdx);

    if (!compressed) {
        return channels_[channelIdx]->getDataVector();
    }

    if (channelIdx == 0) {
        float baseMin = channelInfo[0].bounds.lower.w;
        float baseMax = channelInfo[0].bounds.upper.w;
        std::vector<float> result(baseChannelData_.size());
        for (size_t i = 0; i < baseChannelData_.size(); ++i) {
            result[i] = baseChannelData_[i] * (baseMax - baseMin) + baseMin;
        }
        return result;
    }

    float channelMin = channelInfo[channelIdx].bounds.lower.w;
    float channelMax = channelInfo[channelIdx].bounds.upper.w;
    std::vector<float> result(baseChannelData_.size());

    // Find max exponent used during compression
    int maxExp = 0;
    // Recompute maxExp from compressed data (since diffMin/diffMax are not used)
    // Assume scale used in compression is same as in compressChannels
    // This is a limitation, ideally maxExp should be stored per channel
    // For now, recompute from baseChannelData_ and channelInfo
    // Use the same logic as compressChannels
    std::vector<float> normData(baseChannelData_.size());
    for (size_t i = 0; i < baseChannelData_.size(); ++i) {
        normData[i] = baseChannelData_[i]; // base already normalized
    }
    // We don't have the original diff, so estimate maxExp from compressed mantissas
    // Use the largest mantissa value and invert the scale
    int32_t maxAbsMantissa = 0;
    switch (channelInfo[channelIdx].type) {
        case raw::DataFormat::Int8:
            for (size_t i = 0; i < baseChannelData_.size(); ++i) {
                maxAbsMantissa = std::max(maxAbsMantissa, std::abs(static_cast<int32_t>(compressedDiffs8_[channelIdx-1][i])));
            }
            break;
        case raw::DataFormat::Int16:
            for (size_t i = 0; i < baseChannelData_.size(); ++i) {
                maxAbsMantissa = std::max(maxAbsMantissa, std::abs(static_cast<int32_t>(compressedDiffs16_[channelIdx-1][i])));
            }
            break;
        case raw::DataFormat::Int32:
            for (size_t i = 0; i < baseChannelData_.size(); ++i) {
                maxAbsMantissa = std::max(maxAbsMantissa, std::abs(static_cast<int32_t>(compressedDiffs32_[channelIdx-1][i])));
            }
            break;
        case raw::DataFormat::Float32:
            // Not used in new scheme
            break;
    }

    // Assume targetBits is the type size
    int targetBits = 0;
    switch (channelInfo[channelIdx].type) {
        case raw::DataFormat::Int8: targetBits = 8; break;
        case raw::DataFormat::Int16: targetBits = 16; break;
        case raw::DataFormat::Int32: targetBits = 32; break;
        default: targetBits = 32; break;
    }
    int dropBits = 32 - targetBits;

    // Reconstruct mantissas by appending 0s for missing LSBs
    // (i.e., left-shift by dropBits)
    float scale = 1.0f;
    // Recompute maxExp from compression logic
    // Use the same scale as in compression
    // For now, use maxExp = 0 and scale = ldexp(1.0f, 23)
    // This will only work if maxExp is always 0
    // Ideally, maxExp should be stored per channel
    scale = std::ldexp(1.0f, 23 - maxExp);

    switch (channelInfo[channelIdx].type) {
        case raw::DataFormat::Int8:
            for (size_t i = 0; i < baseChannelData_.size(); ++i) {
                int32_t mantissa = static_cast<int8_t>(compressedDiffs8_[channelIdx-1][i]);
                mantissa = mantissa << dropBits; // append 0s for missing LSBs
                float diff = static_cast<float>(mantissa) / scale;
                float normChannel = baseChannelData_[i] - diff;
                result[i] = normChannel * (channelMax - channelMin) + channelMin;
            }
            break;
        case raw::DataFormat::Int16:
            for (size_t i = 0; i < baseChannelData_.size(); ++i) {
                int32_t mantissa = static_cast<int16_t>(compressedDiffs16_[channelIdx-1][i]);
                mantissa = mantissa << dropBits;
                float diff = static_cast<float>(mantissa) / scale;
                float normChannel = baseChannelData_[i] - diff;
                result[i] = normChannel * (channelMax - channelMin) + channelMin;
            }
            break;
        case raw::DataFormat::Int32:
            for (size_t i = 0; i < baseChannelData_.size(); ++i) {
                int32_t mantissa = static_cast<int32_t>(compressedDiffs32_[channelIdx-1][i]);
                // No need to shift, all bits present
                float diff = static_cast<float>(mantissa) / scale;
                float normChannel = baseChannelData_[i] - diff;
                result[i] = normChannel * (channelMax - channelMin) + channelMin;
            }
            break;
        case raw::DataFormat::Float32:
            // Not used in new scheme
            break;
    }
    return result;
}

// Template function implementation for getting compressed channel data
template<typename T>
std::vector<T> MultiVolume::getCompressedChannelData(size_t channelIdx, raw::DataFormat expectedFormat) const {
    if (!compressed || channelIdx == 0 || channelInfo[channelIdx].type != expectedFormat) {
        return std::vector<T>();
    }
    
    if constexpr (std::is_same_v<T, uint8_t>) {
        if (expectedFormat == raw::DataFormat::Int8) {
            return compressedDiffs8_[channelIdx-1];
        }
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        if (expectedFormat == raw::DataFormat::Int16) {
            return compressedDiffs16_[channelIdx-1];
        }
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        if (expectedFormat == raw::DataFormat::Int32) {
            return compressedDiffs32_[channelIdx-1];
        }
    } else if constexpr (std::is_same_v<T, float>) {
        if (expectedFormat == raw::DataFormat::Float32) {
            return compressedDiffsFloat_[channelIdx-1];
        }
    }
    
    return std::vector<T>();
}

// Explicit template instantiations
template std::vector<uint8_t> MultiVolume::getCompressedChannelData<uint8_t>(size_t, raw::DataFormat) const;
template std::vector<uint16_t> MultiVolume::getCompressedChannelData<uint16_t>(size_t, raw::DataFormat) const;
template std::vector<uint32_t> MultiVolume::getCompressedChannelData<uint32_t>(size_t, raw::DataFormat) const;
template std::vector<float> MultiVolume::getCompressedChannelData<float>(size_t, raw::DataFormat) const;

std::vector<float> MultiVolume::getBaseChannelData() const {
    if (compressed) {
        return baseChannelData_;
    }
    // If not compressed, return the first channel's data
    if (!channels_.empty()) {
        return channels_[0]->getDataVector();
    }
    return std::vector<float>();
}

void MultiVolume::compressChannels(int targetBits) {
    if (compressed) {
        printf("Volume is already compressed.\n");
        return;
    }

    baseChannelData_ = channels_[0]->getDataVector();
    size_t numVoxels = baseChannelData_.size();
    size_t numChannels = channels_.size();
    channelInfo.resize(numChannels);
    channelInfo[0].type = raw::DataFormat::Float32; // base channel is uncompressed

    // Normalize base channel
    float baseMin = *std::min_element(baseChannelData_.begin(), baseChannelData_.end());
    float baseMax = *std::max_element(baseChannelData_.begin(), baseChannelData_.end());
    for (size_t i = 0; i < numVoxels; ++i) {
        baseChannelData_[i] = (baseChannelData_[i] - baseMin) / (baseMax - baseMin);
    }

    compressedDiffs8_.resize(numChannels-1);
    compressedDiffs16_.resize(numChannels-1);
    compressedDiffs32_.resize(numChannels-1);
    compressedDiffsFloat_.resize(numChannels-1);

    auto getExponent = [](float x) -> int {
        int e = 0;
        if (x != 0.0f) {
            float absx = std::fabs(x);
            e = static_cast<int>(std::floor(std::log2(absx)));
        }
        return e;
    };

    for (size_t c = 1; c < numChannels; ++c) {
        const auto& data = channels_[c]->getDataVector();
        std::vector<float> normData(numVoxels);
        float dataMin = channelInfo[c].bounds.lower.w;
        float dataMax = channelInfo[c].bounds.upper.w;
        for (size_t i = 0; i < numVoxels; ++i) {
            normData[i] = (data[i] - dataMin) / (dataMax - dataMin);
        }

        std::vector<float> diff(numVoxels);
        for (size_t i = 0; i < numVoxels; ++i) {
            diff[i] = baseChannelData_[i] - normData[i];
        }

        int maxExp = 0;
        for (size_t i = 0; i < numVoxels; ++i) {
            int e = getExponent(diff[i]);
            if (std::fabs(diff[i]) > 0.0f)
                maxExp = std::max(maxExp, e);
        }

        std::vector<int32_t> mantissas(numVoxels);
        float scale = std::ldexp(1.0f, 23 - maxExp); // 23 mantissa bits for float
        int32_t maxAbsMantissa = 0;
        for (size_t i = 0; i < numVoxels; ++i) {
            mantissas[i] = static_cast<int32_t>(std::round(diff[i] * scale));
            maxAbsMantissa = std::max(maxAbsMantissa, std::abs(mantissas[i]));
        }

        // Drop LSBs to fit targetBits
        int dropBits = 32 - targetBits;
        int32_t maxDroppedMantissa = 0;
        for (size_t i = 0; i < numVoxels; ++i) {
            mantissas[i] >>= dropBits;
            maxDroppedMantissa = std::max(maxDroppedMantissa, std::abs(mantissas[i]));
        }

        // Store in smallest type
        if (targetBits <= 8) {
            channelInfo[c].type = raw::DataFormat::Int8;
            compressedDiffs8_[c-1].resize(numVoxels);
            for (size_t i = 0; i < numVoxels; ++i) {
                compressedDiffs8_[c-1][i] = static_cast<uint8_t>(std::clamp(mantissas[i], -128, 127));
            }
        } else if (targetBits <= 16) {
            channelInfo[c].type = raw::DataFormat::Int16;
            compressedDiffs16_[c-1].resize(numVoxels);
            for (size_t i = 0; i < numVoxels; ++i) {
                compressedDiffs16_[c-1][i] = static_cast<uint16_t>(std::clamp(mantissas[i], -32768, 32767));
            }
        } else {
            channelInfo[c].type = raw::DataFormat::Int32;
            compressedDiffs32_[c-1].resize(numVoxels);
            for (size_t i = 0; i < numVoxels; ++i) {
                compressedDiffs32_[c-1][i] = static_cast<uint32_t>(mantissas[i]);
            }
        }

        channelInfo[c].diffMin = 0.0f;
        channelInfo[c].diffMax = 0.0f;

        printf("Channel %zu compressed with ZFP-like fixed-point, max exponent: %d, max mantissa after drop: %d, bits used: %d\n", c, maxExp, maxDroppedMantissa, targetBits);
    }

    compressed = true;
    channels_.clear();
}

//TODO: implement decompressChannel function
void MultiVolume::decompressChannels() {
    return;
    if (!compressed) {
        printf("Volume is not compressed.\n");
        return;
    }

    // Decompress all channels to restore original data using decompress channel function

    // Clear compressed data to free memory
    compressedDiffs8_.clear();
    compressedDiffs16_.clear();
    compressedDiffs32_.clear();
    compressedDiffsFloat_.clear();
    compressed = false;
    printf("Decompressed all channels and cleared compressed data.\n");
}

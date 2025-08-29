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
    
    if (!compressed || channelIdx == 0) {
        if (compressed && channelIdx == 0) {
            // Return stored base channel after denormalization
            float baseMin = channelInfo[0].bounds.lower.w;
            float baseMax = channelInfo[0].bounds.upper.w;

            // Denormalize base channel data
            std::vector<float> result(baseChannelData_.size());
            for (size_t i = 0; i < baseChannelData_.size(); ++i) {
                result[i] = baseChannelData_[i] * (baseMax - baseMin) + baseMin;
            }
            return result;
        }
        return baseChannelData_;
    }
    
    // Get normalization parameters from bounds
    float channelMin = channelInfo[channelIdx].bounds.lower.w;
    float channelMax = channelInfo[channelIdx].bounds.upper.w;
    
    // decompress: channelN = channel0 - diffN, then denormalize
    std::vector<float> result(baseChannelData_.size());
    
    switch (channelInfo[channelIdx].type) {
        case raw::DataFormat::Int8:
            {
                float diffMin = channelInfo[channelIdx].diffMin;
                float diffMax = channelInfo[channelIdx].diffMax;
                float scale = (diffMax - diffMin) / 255.0f;
                for (size_t i = 0; i < baseChannelData_.size(); ++i) {
                    float dequantizedDiff = static_cast<float>(compressedDiffs8_[channelIdx-1][i]) * scale + diffMin;
                    float normChannel = baseChannelData_[i] - dequantizedDiff;
                    result[i] = normChannel * (channelMax - channelMin) + channelMin;
                }
            }
            break;
        case raw::DataFormat::Int16:
            {
                float diffMin = channelInfo[channelIdx].diffMin;
                float diffMax = channelInfo[channelIdx].diffMax;
                float scale = (diffMax - diffMin) / 65535.0f;
                for (size_t i = 0; i < baseChannelData_.size(); ++i) {
                    float dequantizedDiff = static_cast<float>(compressedDiffs16_[channelIdx-1][i]) * scale + diffMin;
                    float normChannel = baseChannelData_[i] - dequantizedDiff;
                    result[i] = normChannel * (channelMax - channelMin) + channelMin;
                }
            }
            break;
        case raw::DataFormat::Int32:
            {
                float diffMin = channelInfo[channelIdx].diffMin;
                float diffMax = channelInfo[channelIdx].diffMax;
                float scale = (diffMax - diffMin) / static_cast<float>(std::numeric_limits<uint32_t>::max());
                for (size_t i = 0; i < baseChannelData_.size(); ++i) {
                    float dequantizedDiff = static_cast<float>(compressedDiffs32_[channelIdx-1][i]) * scale + diffMin;
                    float normChannel = baseChannelData_[i] - dequantizedDiff;
                    result[i] = normChannel * (channelMax - channelMin) + channelMin;
                }
            }
            break;
        case raw::DataFormat::Float32:
            for (size_t i = 0; i < baseChannelData_.size(); ++i) {
                float normChannel = baseChannelData_[i] - compressedDiffsFloat_[channelIdx-1][i];
                result[i] = normChannel * (channelMax - channelMin) + channelMin;
            }
            break;
    }
    return result;
}

void MultiVolume::compressChannels() {
    if (compressed) {
        printf("Volume is already compressed.\n");
        return;
    }

    printf("Compressing multi-channel volume with %zu channels\n", channels_.size());
    baseChannelData_ = channels_[0]->getDataVector();
    
    //normalize base
    float baseMin = *std::min_element(baseChannelData_.begin(), baseChannelData_.end());
    float baseMax = *std::max_element(baseChannelData_.begin(), baseChannelData_.end());
    printf("Base channel range: [%.6f, %.6f]\n", baseMin, baseMax);
    
    for (size_t i = 0; i < baseChannelData_.size(); ++i) {
        baseChannelData_[i] = (baseChannelData_[i] - baseMin) / (baseMax - baseMin);
    }
    size_t numVoxels = baseChannelData_.size();
    size_t numChannels = channels_.size();
    channelInfo.resize(numChannels);
    channelInfo[0].type = raw::DataFormat::Float32; // base channel is uncompressed
    compressedDiffs8_.resize(numChannels-1);
    compressedDiffs16_.resize(numChannels-1);
    compressedDiffs32_.resize(numChannels-1);
    compressedDiffsFloat_.resize(numChannels-1);
    for (size_t c = 1; c < numChannels; ++c) {
        const auto& data = channels_[c]->getDataVector();
        
        // normalize
        std::vector<float> normData = data;
        float dataMin = channelInfo[c].bounds.lower.w;
        float dataMax = channelInfo[c].bounds.upper.w;
        printf("Channel %zu range: [%.6f, %.6f]\n", c, dataMin, dataMax);
        
        for (size_t i = 0; i < normData.size(); ++i) {
            normData[i] = (data[i] - dataMin) / (dataMax - dataMin);
        }
        
        std::vector<float> diff(numVoxels);
        float minDiff = std::numeric_limits<float>::max();
        float maxDiff = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < numVoxels; ++i) {
            diff[i] = baseChannelData_[i] - normData[i];
            minDiff = std::min(minDiff, diff[i]);
            maxDiff = std::max(maxDiff, diff[i]);
        }
        
        printf("Channel %zu diff range: [%.6f, %.6f]\n", c, minDiff, maxDiff);
        
        // Store diff range for decompression
        channelInfo[c].diffMin = minDiff;
        channelInfo[c].diffMax = maxDiff;
        
        // Choose compression type based on diff range
        float diffRange = maxDiff - minDiff;
        
        // Check if fits in int8 (0 to 255 range)
        if (diffRange <= 255.0f) {
            channelInfo[c].type = raw::DataFormat::Int8;
            compressedDiffs8_[c-1].resize(numVoxels);
            float scale = 255.0f / diffRange;
            for (size_t i = 0; i < numVoxels; ++i) {
                float normalized = (diff[i] - minDiff) * scale;
                compressedDiffs8_[c-1][i] = static_cast<uint8_t>(std::round(std::clamp(normalized, 0.0f, 255.0f)));
            }
            printf("Compressed channel %zu as int8 (range: %.6f to %.6f, scale: %.6f)\n", c, minDiff, maxDiff, scale);
        }
        // Check if fits in int16 (0 to 65535 range)
        else if (diffRange <= 65535.0f) {
            channelInfo[c].type = raw::DataFormat::Int16;
            compressedDiffs16_[c-1].resize(numVoxels);
            float scale = 65535.0f / diffRange;
            for (size_t i = 0; i < numVoxels; ++i) {
                float normalized = (diff[i] - minDiff) * scale;
                compressedDiffs16_[c-1][i] = static_cast<uint16_t>(std::round(std::clamp(normalized, 0.0f, 65535.0f)));
            }
            printf("Compressed channel %zu as int16 (range: %.6f to %.6f, scale: %.6f)\n", c, minDiff, maxDiff, scale);
        }
        // Check if fits in int32 
        else if (diffRange <= static_cast<float>(std::numeric_limits<uint32_t>::max())) {
            channelInfo[c].type = raw::DataFormat::Int32;
            compressedDiffs32_[c-1].resize(numVoxels);
            float scale = static_cast<float>(std::numeric_limits<uint32_t>::max()) / diffRange;
            for (size_t i = 0; i < numVoxels; ++i) {
                float normalized = (diff[i] - minDiff) * scale;
                compressedDiffs32_[c-1][i] = static_cast<uint32_t>(std::round(std::clamp(normalized, 0.0f, static_cast<float>(std::numeric_limits<uint32_t>::max()))));
            }
            printf("Compressed channel %zu as int32 (range: %.6f to %.6f, scale: %.6f)\n", c, minDiff, maxDiff, scale);
        }
        // Fall back to float32 for everything else
        else {
            channelInfo[c].type = raw::DataFormat::Float32;
            compressedDiffsFloat_[c-1] = std::move(diff);
            printf("Compressed channel %zu as float32 (range: %.6f to %.6f)\n", c, minDiff, maxDiff);
        }
    }
    
    // deallocate the channels
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

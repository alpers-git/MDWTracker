#include <cassert>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include <rawFile.h>

//#include "RawFile_impl.hpp"

static std::vector<std::string> string_split(std::string s, char delim)
{
    std::vector<std::string> result;

    std::istringstream stream(s);

    for (std::string token; std::getline(stream, token, delim); )
    {
        result.push_back(token);
    }

    return result;
}

//-------------------------------------------------------------------------------------------------
// C++ API
//

namespace raw
{
    RawR::RawR(char const* fileName, char const* mode)
        : fileName_(fileName)
        , mode_(mode)
    {
        file_ = fopen(fileName_, mode_);

        // Try to parse dimensions and bpv from file name
        std::vector<std::string> strings;

        strings = string_split(fileName_, '_');

        for (auto str : strings)
        {
            int32_t dimx = 0;
            int32_t dimy = 0;
            int32_t dimz = 0;
            size_t bpv = 0;
            int res = 0;

            // Remove the dot and everything to the right of it
            size_t dotPosition = str.find('.');
             if (dotPosition != std::string::npos)
                str= str.substr(0, dotPosition);

            // Dimensions
            res = sscanf(str.c_str(), "%dx%dx%d", &dimx, &dimy, &dimz);
            if (res == 3)
            {
                dims_ = { dimx, dimy, dimz };
                continue;
            }

            if (str == "int8") {
                dataFormat_ = DataFormat::Int8; continue;
            } else if (str == "int16") {
                dataFormat_ = DataFormat::Int16; continue;
            } else if (str == "int32") {
                dataFormat_ = DataFormat::Int32; continue;
            } else if (str == "uint8") {
                dataFormat_ = DataFormat::UInt8; continue;
            } else if (str == "uint16") {
                dataFormat_ = DataFormat::UInt16; continue;
            } else if (str == "uint32") {
                dataFormat_ = DataFormat::UInt32; continue;
            } else if (str == "float32") {
                dataFormat_ = DataFormat::Float32; continue;
            } else if (str == "double64") {
                dataFormat_ = DataFormat::Double64; continue;
            } else {
                dataFormat_ = DataFormat::Unspecified; continue;
            }
        }

        //allocate enough memory for data
        auto buf = new char[dims_.x * dims_.y * dims_.z * getBytesPerVoxel()];
        //read the data into the allocated memory
        this->read(buf, getBytesPerVoxel() * dims_.x * dims_.y * dims_.z);
        //go over the data and convert it to float and store it in data_
        data_ = std::vector<float>(dims_.x * dims_.y * dims_.z);
        for (int i = 0; i < dims_.x * dims_.y * dims_.z; i++)
        {
            switch (dataFormat_)
            {
            case DataFormat::Int8:
                data_[i] = ((float)((int8_t*)buf)[i] * 255 - 128);
                break;
            case DataFormat::Int16:
                data_[i] = ((float)((int16_t*)buf)[i] * 65535 - 32768);
                break;
            case DataFormat::Int32:
                data_[i] = ((float)((int32_t*)buf)[i] * 4294967295 - 2147483648);
                break;
            case DataFormat::UInt8:
                data_[i] = ((float)((uint8_t*)buf)[i] * 255);
                break;
            case DataFormat::UInt16:
                data_[i] = ((float)((uint16_t*)buf)[i] * 65535);
                break;
            case DataFormat::UInt32:
                data_[i] = ((float)((uint32_t*)buf)[i] * 4294967295);
                break;
            case DataFormat::Float32:
                data_[i] = ((float*)buf)[i];
                break;
            case DataFormat::Double64:
                data_[i] = (float)((double*)buf)[i];
                break;
            default:
                break;
            }
            //update bounds w
            if (data_[i] > bounds_.upper.w)
                bounds_.upper.w = data_[i];
            if (data_[i] < bounds_.lower.w)
                bounds_.lower.w = data_[i];
        }
    }

    RawR::RawR(FILE* file)
        : file_(file)
    {
        file_ = fopen(fileName_, mode_);
    }

    RawR::~RawR()
    {
        fclose(file_);
    }

    std::size_t RawR::read(char* buf, std::size_t len)
    {
        return fread(buf, len, 1, file_);
    }

    std::size_t RawR::write(char const* buf, std::size_t len)
    {
        return fwrite(buf, len, 1, file_);
    }

    bool RawR::seek(std::size_t pos)
    {
        if (!good())
            return false;

        int res = fseek(file_, (long)pos, SEEK_SET);

        return res == 0;
    }

    bool RawR::flush()
    {
        if (!good())
            return false;

        int res = fflush(file_);

        return res == 0;
    }

    bool RawR::good() const
    {
        return file_ != nullptr;
    }

    void RawR::setDims(Vec3i dims)
    {
        dims_ = dims;
    }

    Vec3i RawR::getDims() const
    {
        return dims_;
    }

    void RawR::setDataFormat(DataFormat dataFormat)
    {
        dataFormat_ = dataFormat;
    }

    DataFormat RawR::getDataFormat() const
    {
        return dataFormat_;
    }

    size_t RawR::getBytesPerVoxel() const
    {
        return raw::getSizeInBytes(dataFormat_);
    }

    const std::vector<float> RawR::getDataVector() const
    {
        return data_;
    }

    owl::box4f RawR::getBounds4f() const
    {
        return bounds_;
    }

    owl::box3f RawR::getBounds() const
    {
        return owl::box3f({bounds_.lower.x, bounds_.lower.y, bounds_.lower.z},
            {bounds_.upper.x, bounds_.upper.y, bounds_.upper.z});
    }

    void RawR::reshapeBounds()
    {
        //rescale the bounds to match the dimension resolution
        bounds_.lower.x = bounds_.lower.x * dims_.x;
        bounds_.lower.y = bounds_.lower.y * dims_.y;
        bounds_.lower.z = bounds_.lower.z * dims_.z;
        bounds_.upper.x = bounds_.upper.x * dims_.x;
        bounds_.upper.y = bounds_.upper.y * dims_.y;
        bounds_.upper.z = bounds_.upper.z * dims_.z;
    }

     void RawR::reshapeBounds(const owl::vec3f remap_dims)
    {
        //rescale the bounds to match the dimension resolution
        bounds_.lower.x = bounds_.lower.x * remap_dims.x;
        bounds_.lower.y = bounds_.lower.y * remap_dims.y;
        bounds_.lower.z = bounds_.lower.z * remap_dims.z;
        bounds_.upper.x = bounds_.upper.x * remap_dims.x;
        bounds_.upper.y = bounds_.upper.y * remap_dims.y;
        bounds_.upper.z = bounds_.upper.z * remap_dims.z;
    }

} // raw

//
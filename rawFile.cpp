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

namespace vkt
{
    RawFile::RawFile(char const* fileName, char const* mode)
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
            uint16_t bpv = 0;
            int res = 0;

            // Dimensions
            res = sscanf(str.c_str(), "%dx%dx%d", &dimx, &dimy, &dimz);
            if (res == 3)
                dims_ = { dimx, dimy, dimz };

            res = sscanf(str.c_str(), "int%hu", &bpv);
            if (res == 1)
            {
                switch (bpv)
                {
                case 8:
                    dataFormat_ = DataFormat::Int8;
                    break;

                case 16:
                    dataFormat_ = DataFormat::Int16;
                    break;

                case 32:
                    dataFormat_ = DataFormat::Int32;
                    break;

                default:
                    dataFormat_ = DataFormat::Unspecified;
                    break;
                }
            }

            res = sscanf(str.c_str(), "uint%hu", &bpv);
            if (res == 1)
            {
                switch (bpv)
                {
                case 8:
                    dataFormat_ = DataFormat::UInt8;
                    break;

                case 16:
                    dataFormat_ = DataFormat::UInt16;
                    break;

                case 32:
                    dataFormat_ = DataFormat::UInt32;
                    break;

                default:
                    dataFormat_ = DataFormat::Unspecified;
                    break;
                }
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
                data_[i] = (float)((int8_t*)buf)[i];
                break;
            case DataFormat::Int16:
                data_[i] = (float)((int16_t*)buf)[i];
                break;
            case DataFormat::Int32:
                data_[i] = (float)((int32_t*)buf)[i];
                break;
            case DataFormat::UInt8:
                data_[i] = (float)((uint8_t*)buf)[i];
                break;
            case DataFormat::UInt16:
                data_[i] = (float)((uint16_t*)buf)[i];
                break;
            case DataFormat::UInt32:
                data_[i] = (float)((uint32_t*)buf)[i];
                break;
            case DataFormat::Float32:
                data_[i] = ((float*)buf)[i];
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

    RawFile::RawFile(FILE* file)
        : file_(file)
    {
        file_ = fopen(fileName_, mode_);
    }

    RawFile::~RawFile()
    {
        fclose(file_);
    }

    std::size_t RawFile::read(char* buf, std::size_t len)
    {
        return fread(buf, len, 1, file_);
    }

    std::size_t RawFile::write(char const* buf, std::size_t len)
    {
        return fwrite(buf, len, 1, file_);
    }

    bool RawFile::seek(std::size_t pos)
    {
        if (!good())
            return false;

        int res = fseek(file_, (long)pos, SEEK_SET);

        return res == 0;
    }

    bool RawFile::flush()
    {
        if (!good())
            return false;

        int res = fflush(file_);

        return res == 0;
    }

    bool RawFile::good() const
    {
        return file_ != nullptr;
    }

    void RawFile::setDims(Vec3i dims)
    {
        dims_ = dims;
    }

    Vec3i RawFile::getDims() const
    {
        return dims_;
    }

    void RawFile::setDataFormat(DataFormat dataFormat)
    {
        dataFormat_ = dataFormat;
    }

    DataFormat RawFile::getDataFormat() const
    {
        return dataFormat_;
    }

    size_t RawFile::getBytesPerVoxel() const
    {
        return vkt::getSizeInBytes(dataFormat_);
    }

    const float* RawFile::getData() const
    {
        return data_.data();
    }

    owl::box4f RawFile::getBounds4f() const
    {
        return bounds_;
    }

} // vkt

//-------------------------------------------------------------------------------------------------
// C API
// //

// void vktRawFileCreateS(vktRawFile* file, char const* fileName, char const* mode)
// {
//     assert(file != nullptr);

//     *file = new vktRawFile_impl(fileName, mode);
// }

// void vktRawFileCreateFD(vktRawFile* file, FILE* fd)
// {
//     assert(file != nullptr);

//     *file = new vktRawFile_impl(fd);
// }

// vktDataSource vktRawFileGetBase(vktRawFile file)
// {
//     return file->base;
// }

// void vktRawFileDestroy(vktRawFile file)
// {
//     delete file;
// }

// size_t vktRawFileRead(vktRawFile file, char* buf, size_t len)
// {
//     return file->base->source->read(buf, len);
// }

// bool vktRawFileGood(vktRawFile file)
// {
//     return file->base->source->good() ? true : false;
// }

// Vec3i vktRawFileGetDims3iv(vktRawFile file)
// {
//     vkt::RawFile* rf = dynamic_cast<vkt::RawFile*>(file->base->source);
//     assert(rf);

//     Vec3i dims = rf->getDims();

//     return { dims.x, dims.y, dims.z };
// }

// vktDataFormat vktRawFileGetDataFormat(vktRawFile file)
// {
//     vkt::RawFile* rf = dynamic_cast<vkt::RawFile*>(file->base->source);
//     assert(rf);

//     return (vktDataFormat)rf->getDataFormat();

// }
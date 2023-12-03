// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <limits>

// #include "common.hpp"
// #include "linalg.hpp"
#include <owl/common/math/vec.h>
#include <owl/common/math/box.h>


namespace raw
{
    using Vec3i = owl::vec3i;
    enum class DataFormat
    {
        Unspecified,

        Int8,
        Int16,
        Int32,
        UInt8,
        UInt16,
        UInt32,
        Float32,

        // Keep last!
        Count,
    };

    struct DataFormatInfo
    {
        DataFormat dataFormat;
        uint8_t sizeInBytes;
    };

    static DataFormatInfo DataFormatInfoTable[(int)DataFormat::Count] = {
            { DataFormat::Unspecified,  0 },
            { DataFormat::Int8,         1 },
            { DataFormat::Int16,        2 },
            { DataFormat::Int32,        4 },
            { DataFormat::UInt8,        1 },
            { DataFormat::UInt16,       2 },
            { DataFormat::UInt32,       4 },
            { DataFormat::Float32,      4 },

    };

    // Equivalent to table, but can be used in CUDA device code
    constexpr inline uint8_t getSizeInBytes(DataFormat dataFormat)
    {
       if (dataFormat == DataFormat::Int8 || dataFormat == DataFormat::UInt8)
           return 1;

       if (dataFormat == DataFormat::Int16 || dataFormat == DataFormat::UInt16)
           return 2;

       if (dataFormat == DataFormat::Int32 || dataFormat == DataFormat::UInt32)
           return 4;

       if (dataFormat == DataFormat::Float32)
           return 4;

       return 255;
    }

     /*!
     * @brief  Data source base class for file I/O
     */
    class DataSource
    {
    public:
        virtual ~DataSource() {}
        virtual std::size_t read(char* buf, std::size_t len) = 0;
        virtual std::size_t write(char const* buf, std::size_t len) = 0;
        virtual bool seek(std::size_t pos) = 0;
        virtual bool flush() = 0;
        virtual bool good() const = 0;

    };

    class RawR : public DataSource
    {
    public:
        RawR(char const* fileName, char const* mode);
        RawR(FILE* file);
       ~RawR();

        virtual std::size_t read(char* buf, std::size_t len);
        virtual std::size_t write(char const* buf, std::size_t len);
        virtual bool seek(std::size_t pos);
        virtual bool flush();
        virtual bool good() const;

        

        /*!
         * @brief  Set structured volume dimensions
         */
        void setDims(Vec3i dims); 

        /*!
         * @brief  Structured volume dimensions parsed from file name,
         *         0 if not successful
         */
        Vec3i getDims() const;

        /*!
         * @brief  Set structured volume data format
         */
        void setDataFormat(DataFormat dataFormat);

        /*!
         * @brief  Get structured volume data format
         */
        DataFormat getDataFormat() const;


        /*!
        * @brief Get bytes per voxel
        */
        size_t getBytesPerVoxel() const;

        /*!
        * @brief Get data
        */
        const std::vector<float> getDataVector() const;

        /*!
        * @brief Get domain of the values in the volume
        */
        owl::box4f getBounds4f() const;

        /*!
        * @brief Get domain of the values in the volume
        */
        owl::box3f getBounds() const;

    private:
        char const* fileName_ = 0;
        char const* mode_ = 0;
        FILE* file_ = 0;
        std::vector<float> data_;// I will assume they are all floats for now

        Vec3i dims_ = { 0, 0, 0 };
        owl::box4f bounds_ = {{0,0,0,0},{1,1,1,1}};
        DataFormat dataFormat_ = DataFormat::UInt8;

    };

} // raw
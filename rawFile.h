// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cstdio>
#include <cstddef>
#include <cstdint>

// #include "common.hpp"
// #include "linalg.hpp"
#include <owl/common/math/vec.h>


namespace vkt
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

    class RawFile : public DataSource
    {
    public:
        RawFile(char const* fileName, char const* mode);
        RawFile(FILE* file);
       ~RawFile();

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

    private:
        char const* fileName_ = 0;
        char const* mode_ = 0;
        FILE* file_ = 0;

        Vec3i dims_ = { 0, 0, 0 };
        DataFormat dataFormat_ = DataFormat::UInt8;

    };

} // vkt
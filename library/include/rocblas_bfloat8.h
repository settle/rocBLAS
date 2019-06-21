/**
 * MIT License
 *
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*!\file
 * \brief rocblas_bfloat8.h provides struct for rocblas_bfloat8 typedef
 */

#pragma once
#ifndef _ROCBLAS_BFLOAT8_H_
#define _ROCBLAS_BFLOAT8_H_

#ifndef __cplusplus

#include <inttypes.h>

/** @struct rocblas_bfloat8
 * struct used for bfloat8
 * @var rocblas_bfloat8::data
 * struct member variable used to store bfloat8 data
 */
typedef struct
{
    uint8_t data;
} rocblas_bfloat8;

#else // __cplusplus

#include <hip/hip_runtime.h>

#include <cinttypes>
#include <cmath>
#include <iostream>
#include <type_traits>

struct rocblas_bfloat8
{
    uint8_t data;

    // Skip initializing `data` on purpose so that `bfloat8` could be used
    // with `__share__`, which forbids any initializer, including the implicit
    // one.
    __host__ __device__ rocblas_bfloat8() {}

    // round upper 8 bits of IEEE float to convert to bfloat8
    explicit __host__ __device__ constexpr rocblas_bfloat8(float f)
        : data(float_to_bfloat8(f))
    {
    }

    // zero extend lower 8 bits of bfloat8 to convert to IEEE float
    explicit __host__ __device__ constexpr operator float() const
    {
        union
        {
            uint16_t int16;
            float    fp16;
        } u = {uint16_t(data) << 8};
        return u.fp16;
    }

private:
    static __host__ __device__ constexpr uint8_t float_to_bfloat8(float f)
    {
        union
        {
            float    fp16;
            uint16_t int16;
        } u = {f};
        if(~u.int16 & 0x7c00)
        {
            // When the exponent bits are not all 1s, then the value is zero, normal,
            // or subnormal. We round the bfloat8 mantissa up by adding 0x7F, plus
            // 1 if the least significant bit of the bfloat8 mantissa is 1 (odd).
            // This causes the bfloat8's mantissa to be incremented by 1 if the 8
            // least significant bits of the half mantissa are greater than 0x80,
            // or if they are equal to 0x80 and the least significant bit of the
            // bfloat8 mantissa is 1 (odd). This causes it to be rounded to even when
            // the lower 8 bits are exactly 0x80. If the bfloat8 mantissa already
            // has the value 0x3, then incrementing it causes it to become 0x0 and
            // the exponent is incremented by one, which is the next higher FP value
            // to the unrounded bfloat8 value. When the bfloat8 value is subnormal
            // with an exponent of 0x0 and a mantissa of 0x3, it may be rounded up
            // to a normal value with an exponent of 0x1 and a mantissa of 0x0.
            // When the bfloat8 value has an exponent of 0x1E and a mantissa of 0x3,
            // incrementing it causes it to become an exponent of 0x1F and a mantissa
            // of 0x0, which is Inf, the next higher value to the unrounded value.
            u.int16 += 0x7f + ((u.int16 >> 8) & 1); // Round to nearest, round to even
        }
        else if(u.int16 & 0xff)
        {
            // When all of the exponent bits are 1, the value is Inf or NaN.
            // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
            // mantissa bit. Quiet NaN is indicated by the most significant mantissa
            // bit being 1. Signaling NaN is indicated by the most significant
            // mantissa bit being 0 but some other bit(s) being 1. If any of the
            // lower 8 bits of the mantissa are 1, we set the least significant bit
            // of the bfloat8 mantissa, in order to preserve signaling NaN in case
            // the bloat8's mantissa bits are all 0.
            u.int16 |= 0x100; // Preserve signaling NaN
        }
        return uint8_t(u.int16 >> 8);
    }
};

static_assert(std::is_standard_layout<rocblas_bfloat8>{},
              "rocblas_bfloat8 is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivially_copyable<rocblas_bfloat8>{},
              "rocblas_bfloat8 is not trivially copyable, and thus is "
              "incompatible with C.");

inline std::ostream& operator<<(std::ostream& os, const rocblas_bfloat8& bf8)
{
    return os << float(bf8);
}
inline __host__ __device__ rocblas_bfloat8 operator+(rocblas_bfloat8 a)
{
    return a;
}
inline __host__ __device__ rocblas_bfloat8 operator-(rocblas_bfloat8 a)
{
    a.data ^= 0x80;
    return a;
}
inline __host__ __device__ rocblas_bfloat8 operator+(rocblas_bfloat8 a, rocblas_bfloat8 b)
{
    return rocblas_bfloat8(float(a) + float(b));
}
inline __host__ __device__ rocblas_bfloat8 operator-(rocblas_bfloat8 a, rocblas_bfloat8 b)
{
    return rocblas_bfloat8(float(a) - float(b));
}
inline __host__ __device__ rocblas_bfloat8 operator*(rocblas_bfloat8 a, rocblas_bfloat8 b)
{
    return rocblas_bfloat8(float(a) * float(b));
}
inline __host__ __device__ rocblas_bfloat8 operator/(rocblas_bfloat8 a, rocblas_bfloat8 b)
{
    return rocblas_bfloat8(float(a) / float(b));
}
inline __host__ __device__ bool operator<(rocblas_bfloat8 a, rocblas_bfloat8 b)
{
    return float(a) < float(b);
}
inline __host__ __device__ bool operator==(rocblas_bfloat8 a, rocblas_bfloat8 b)
{
    return float(a) == float(b);
}
inline __host__ __device__ bool operator>(rocblas_bfloat8 a, rocblas_bfloat8 b)
{
    return b < a;
}
inline __host__ __device__ bool operator<=(rocblas_bfloat8 a, rocblas_bfloat8 b)
{
    return !(a > b);
}
inline __host__ __device__ bool operator!=(rocblas_bfloat8 a, rocblas_bfloat8 b)
{
    return !(a == b);
}
inline __host__ __device__ bool operator>=(rocblas_bfloat8 a, rocblas_bfloat8 b)
{
    return !(a < b);
}
inline __host__ __device__ rocblas_bfloat8& operator+=(rocblas_bfloat8& a, rocblas_bfloat8 b)
{
    return a = a + b;
}
inline __host__ __device__ rocblas_bfloat8& operator-=(rocblas_bfloat8& a, rocblas_bfloat8 b)
{
    return a = a - b;
}
inline __host__ __device__ rocblas_bfloat8& operator*=(rocblas_bfloat8& a, rocblas_bfloat8 b)
{
    return a = a * b;
}
inline __host__ __device__ rocblas_bfloat8& operator/=(rocblas_bfloat8& a, rocblas_bfloat8 b)
{
    return a = a / b;
}
inline __host__ __device__ rocblas_bfloat8& operator++(rocblas_bfloat8& a)
{
    return a += rocblas_bfloat8(1.0f);
}
inline __host__ __device__ rocblas_bfloat8& operator--(rocblas_bfloat8& a)
{
    return a -= rocblas_bfloat8(1.0f);
}
inline __host__ __device__ rocblas_bfloat8 operator++(rocblas_bfloat8& a, int)
{
    rocblas_bfloat8 orig = a;
    ++a;
    return orig;
}
inline __host__ __device__ rocblas_bfloat8 operator--(rocblas_bfloat8& a, int)
{
    rocblas_bfloat8 orig = a;
    --a;
    return orig;
}
inline __host__ __device__ bool isinf(rocblas_bfloat8 a)
{
    return !(~a.data & 0x7c) && !(a.data & 0x3);
}
inline __host__ __device__ bool isnan(rocblas_bfloat8 a)
{
    return !(~a.data & 0x7c) && +(a.data & 0x3);
}
inline __host__ __device__ bool iszero(rocblas_bfloat8 a)
{
    return !(a.data & 0x7f);
}
inline __host__ __device__ rocblas_bfloat8 abs(rocblas_bfloat8 a)
{
    a.data &= 0x7f;
    return a;
}
inline rocblas_bfloat8 sin(rocblas_bfloat8 a)
{
    return rocblas_bfloat8(sinf(float(a)));
}
inline rocblas_bfloat8 cos(rocblas_bfloat8 a)
{
    return rocblas_bfloat8(cosf(float(a)));
}

#endif // __cplusplus

#endif // _ROCBLAS_BFLOAT8_H_

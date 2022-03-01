#pragma once

#include <assert.h>
#include <stdint.h>
#include "FreeImage.h"

template <typename T>
static inline T clamp(T value, T min, T max)
{
    if (value > min) {
        if (value < max) {
            return value;
        }
        return max;
    }
    return min;
}

template <typename T>
T *ImageUtils_RGB16ToArray(FIBITMAP *image)
{
    unsigned int width = FreeImage_GetWidth(image);
    unsigned int height = FreeImage_GetHeight(image);
    T *result = new T[width * height];

    unsigned int pitch = FreeImage_GetPitch(image);
    BYTE *in_data = (BYTE *) FreeImage_GetBits(image);

    for(size_t y = 0; y < height; y++) {
        FIRGB16 *pixel_in = (FIRGB16 *) in_data;
        for(size_t x = 0; x < width; x++) {
            T pixel = 0;
            pixel += 0.2126 * pixel_in[x].red;
            pixel += 0.7152 * pixel_in[x].green;
            pixel += 0.0722 * pixel_in[x].blue;
            pixel /= UINT16_MAX;

            result[y * width + x] = pixel;
        }
        // next line
        in_data += pitch;
    }

    return result;
}

template <typename T>
FIBITMAP *ImageUtils_ArrayToRGB16(const T *image, int width, int height)
{
    T min = 0.0;
    T max = 1.0;

    for(int n = 0; n < width * height; n++) {
        if (image[n] > max)
            max = image[n];
        if (image[n] < min)
            min = image[n];
    }
    const T normalizer = 1.0 / (max - min);

    FIBITMAP *result = FreeImage_AllocateT(FIT_RGB16, width, height, 8 * sizeof(FIRGB16));
    unsigned int out_pitch = FreeImage_GetPitch(result);
    BYTE *out_data = (BYTE *) FreeImage_GetBits(result);

    for(int y = 0; y < height; y++) {
        FIRGB16 *pixel_out = (FIRGB16 *) out_data;
        for(int x = 0; x < width; x++) {
            uint16_t value = UINT16_MAX * ((image[y * width + x] - min) * normalizer);
            pixel_out[x].red = value;
            pixel_out[x].green = value;
            pixel_out[x].blue = value;
        }
        // next line
        out_data += out_pitch;
    }

    return result;
}

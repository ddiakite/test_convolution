#pragma once

#include <chrono>
#include <stddef.h>
#include <tuple>
#include <vector>

typedef std::chrono::nanoseconds::rep elapsed_time_t;

#define TICK() std::chrono::high_resolution_clock::time_point _tick{std::chrono::high_resolution_clock::now()}
#define TOCK() std::chrono::high_resolution_clock::time_point _tock{std::chrono::high_resolution_clock::now()}
#define ELAPSED_TIME (std::chrono::duration_cast<std::chrono::nanoseconds>(_tock - _tick).count())

#define INSTANTIATE(name, T) template T * name <T> (const T *__restrict__ input, size_t input_width, size_t input_height,\
            const T *__restrict__ mask, size_t radius, std::vector<elapsed_time_t> &results, size_t loop_count)



enum float_precision {
    single_float = 0,
    double_float = 1,
};

typedef float *(*conv_function_s)(const float *, size_t, size_t, const float *, size_t, std::vector<elapsed_time_t>&, size_t);
typedef double *(*conv_function_d)(const double *, size_t, size_t, const double *, size_t, std::vector<elapsed_time_t>&, size_t);

typedef std::tuple<float *, double *> mixed_precision_tuple;

//typedef std::tuple<conv_function_h, conv_function_s, conv_function_d> conv_function_tuple;

#define DECLARE_IMPLEMENTATION(name) template <typename T> T * name(const T *__restrict__ input, size_t input_width, size_t input_height,\
            const T *__restrict__ mask, size_t mask_width, std::vector<elapsed_time_t> &results, size_t loop_count)

namespace conv {

DECLARE_IMPLEMENTATION(cpu_convolution_naive);
DECLARE_IMPLEMENTATION(oneapi_convolution);

}

#include "utils.h"

#include <math.h>
#include <map>
#include <stdio.h>
#include <limits>

size_t upper_power_of_two(size_t x)
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;

    return x;
}

template <typename T>
double compare_images(T *first, T *second, size_t width, size_t height,
    std::ostream* errorwriter, double local_threshold, double global_threshold)
{
    double global_error {0.0f};
    for(size_t y = 0; y < height; ++y) {
        for(size_t x = 0; x < height; ++x) {
	    bool notZero=fabs(first[y * width + x]) > std::numeric_limits<float>::epsilon()*second[y * width + x];
	    double error= fabs(first[y * width + x] - second[y * width + x]);
	    if(notZero){
		 error /= first[y * width + x];
	    }
            if (error > local_threshold) {
                fprintf(stderr, "Difference at coordinate %lu, %lu is %f, higher than threshold %f\n",
                        x, y, error, local_threshold);
                return INFINITY;
            }
            global_error += error;
	        if(errorwriter){
		        errorwriter->write((char*)&error,sizeof(double));
            }
        }
    }
    global_error /= width * height;

    if (global_error > global_threshold) {
        fprintf(stderr, "Global difference (normalized) is %f, higher than threshold %f\n",
                global_error, global_threshold);
        return INFINITY;
    }
    return global_error;
}

template double compare_images<float>(float *first, float *second, size_t width, size_t height,
        std::ostream* errorwriter=0, double local_threshold=INFINITY, double global_threshold=INFINITY);
template double compare_images<double>(double *first, double *second, size_t width, size_t height,
        std::ostream* errorwriter=0, double local_threshold=INFINITY, double global_threshold=INFINITY);

void openofstream(std::ofstream& s,const char* fileName){
	s.open(fileName,std::ios::binary);
	s.exceptions(s.failbit|s.badbit);
}

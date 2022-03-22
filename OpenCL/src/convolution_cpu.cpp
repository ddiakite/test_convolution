#include "convolution.h"

#include <stddef.h>
#include <vector>

//#include "utils.h"

namespace conv {

template <typename T>
T *cpu_convolution_naive(const T *__restrict__ input, size_t input_width, size_t input_height,
        const T *__restrict__ mask, size_t mask_width, std::vector<elapsed_time_t> &results, size_t loop_count)
{
    const size_t radius {mask_width /2};
    T *output = new T[input_width * input_height];

    for (size_t i = 0; i <= loop_count; i++) {
        TICK();
        for(size_t y = 0; y < input_height; ++y) {
            for(size_t x = 0; x < input_width; ++x) {
                T result = 0.;

                for(size_t my = 0; my < mask_width; ++my) {
                    for(size_t mx = 0; mx < mask_width; ++mx) {
                        size_t mask_index = my * mask_width + mx;
                        T image_value = 0.0f;
                        if (x + mx >= radius
                                && x + mx - radius < input_width
                                && y + my >= radius
                                && y + my - radius < input_height)
                            image_value = input[(y + my - radius) * input_width + (x + mx - radius)];
                        result += mask[mask_index] * image_value;
                    }
                }
                output[y * input_width + x] = result;
            }
        }
        TOCK();
        if (i > 0)
            results.push_back(ELAPSED_TIME);
    }
    return output;
}

INSTANTIATE(cpu_convolution_naive, float);
INSTANTIATE(cpu_convolution_naive, double);

}


namespace conv {

template <typename T>
T *cpu_convolution_naive_p(const T *__restrict__ input, size_t input_width, size_t input_height,
        const T *__restrict__ mask, size_t mask_width, std::vector<elapsed_time_t> &results, size_t loop_count)
{
    const size_t radius {mask_width /2};
    T *output = new T[input_width * input_height];
    
    float input_local[input_width * mask_width];
    float output_local[input_width];
    float mask_local[mask_width * mask_width];
    
    for(size_t l=0; l < mask_width; ++l){
		for(size_t m=0; m < mask_width; ++m){
			mask_local[l * mask_width + m] = 1.0 * (l * mask_width + m);
		}
	}

    for (size_t i = 0; i <= loop_count; i++) {
        TICK();
        for(size_t y = 0; y < input_height; ++y) {
        
        size_t y_prime;
        y_prime = y - radius;
        
        if(y < radius)
        	y_prime = 0;
        	 
        
        for(unsigned int p = 0; p < input_width * mask_width; p++){
		input_local[p] = input[(y_prime * input_width + p)];
	}
	
	/*for(size_t l=0; l < mask_width; ++l){
		for(size_t m=0; m < mask_width; ++m){
			printf("%f\t", mask_local[l * mask_width + m]);
		}
		printf("\n");
	}*/
	//circular shift register applied to the mask. 
	float temp[mask_width];
	for(size_t k=0; k < mask_width; ++k)
		temp[k] = mask_local[k];
		
	for(size_t l=0; l < mask_width - 1; ++l){
		for(size_t m=0; m < mask_width; ++m){
			size_t id = l * mask_width + m;
			mask_local[id] = mask_local[id + mask_width];
		}
		
	}
	for(size_t k=0; k < mask_width; ++k)
		mask_local[mask_width*(mask_width-1) + k] = temp[k];
	
	for(unsigned int p = 0; p < input_width; p++){
		output_local[p] = 0.0f;
	}
            for(size_t x = 0; x < input_width; ++x) {
                T result = 0.; int i = 0;

                for(size_t my = 0; my < mask_width; ++my) {
                    for(size_t mx = 0; mx < mask_width; ++mx) {
                        size_t mask_index = my * mask_width + mx;
                        T image_value = 0.0f;
                        if (x + mx >= radius
                                && x + mx - radius < input_width
                                && y + my >= radius
                                && y + my - radius < input_height)
                            //image_value = input[(y + my - radius) * input_width + (x + mx - radius)];
                            image_value = input_local[my * input_width + (x + mx - radius)];
                            
                            
                        result += mask[mask_index] * image_value;
                    }
                }
                //output[y * input_width + x] = result;
                output_local [x] = result;
            }
            
            for(unsigned int p = 0; p < input_width; p++){
		output[y * input_width + p] = output_local[p];
	}
        }
        TOCK();
        if (i > 0)
            results.push_back(ELAPSED_TIME);
    }
    return output;
}

INSTANTIATE(cpu_convolution_naive_p, float);
INSTANTIATE(cpu_convolution_naive_p, double);

}

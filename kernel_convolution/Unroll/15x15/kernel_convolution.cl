#define input_height 512
#define input_width 512
#define mask_width 15

__kernel void convolution_2D(__global const float * restrict input,
				 __global float * restrict output,
				 __constant float * restrict mask
				)
{


const size_t radius = mask_width /2;

for(unsigned int y = 0; y < input_height; ++y) {
	//#pragma unroll 2
	for(unsigned int x = 0; x < input_width; ++x) {
		float result = 0.;
		
		#pragma unroll
		for(unsigned int my = 0; my < mask_width; ++my) {
		    #pragma unroll
		    for(unsigned int mx = 0; mx < mask_width; ++mx) {
			unsigned int mask_index = my * mask_width + mx;
			float image_value = 0.0f;
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

}

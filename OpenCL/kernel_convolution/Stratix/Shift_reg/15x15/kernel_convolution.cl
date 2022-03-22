#define input_height 512
#define input_width 512
#define mask_width 15

__kernel void convolution_2D(__global const float * restrict input,
				 __global float * restrict output,
				 __constant float * restrict mask
				)
{


const size_t radius = mask_width /2;


float //__attribute__((numbanks(1)))
 	     //__attribute__((doublepump))
    	      //__attribute__((numwriteports(2)))
 	      //__attribute__((numreadports(226))) 
 	      input_local[input_width * mask_width];
 	      


float mask_local[mask_width * mask_width];
 	      
 	      
float __attribute__((numbanks(2)))
	__attribute__((numwriteports(2)))
       __attribute__((numreadports(1))) output_local[input_width];

	size_t cpt = mask_width - radius;
	size_t n = 0;
	

for(size_t l=0; l < mask_width; ++l){
	#pragma unroll 
	for(size_t m=0; m < mask_width; ++m){
		mask_local[l * mask_width + m] = mask[l * mask_width + m];
	}
}


for(unsigned int q = 0; q < (mask_width - radius); q++){
	#pragma unroll 16
	for(unsigned int p = 0; p < input_width; p++){
		input_local[q*input_width + p] = input[q*input_width + p];
	}
}

	
//#pragma max_concurrency 8
for(unsigned int y = 0; y < input_height; ++y) {

	size_t y_prime;
	
	if(y < radius || y >= (input_height  - radius))
        	y_prime = y;
	else
        	y_prime = y + radius;
        
        
	
	#pragma unroll 16
	for(unsigned int p = 0; p < input_width; p++){
		
		input_local[(cpt * input_width) + p] = input[y_prime * input_width + p];
	}
	cpt++;
	if(cpt == mask_width){
		cpt = 0;
	}
	
	//#pragma unroll 64
	for(unsigned int p = 0; p < input_width; p++){
		output_local[p] = 0.0f;
	}
	
	//#pragma max_concurrency input_width
	//#pragma unroll 4
	for(unsigned int x = 0; x < input_width; ++x) {
		float result = 0.0f;

		#pragma unroll
		for(unsigned int my = 0; my < mask_width; ++my) {
		    #pragma unroll
		    for(unsigned int mx = 0; mx < mask_width; ++mx) {
			unsigned int mask_index = my * mask_width + mx;
			float image_value = 0.0f;
			if (x + mx >= radius
				&& x + mx - radius < input_width
				&& y + my > radius
				&& y + my - radius < input_height)
			    //image_value = input[(y + my - radius) * input_width + (x + mx - radius)];
			    image_value = input_local[my * input_width + (x + mx - radius)];
		
			result += mask_local[mask_index] * image_value;
		    }
		}
		    
		    //output[y * input_width + x] = result;
		    output_local [x] = result;
		    
	}
	
	
	#pragma unroll 64
	for(unsigned int p = 0; p < input_width; p++){
		output[y * input_width + p] = output_local[p];
	}
	
	/*
	#pragma unroll 8
	for(unsigned int p = 0; p < input_width * (mask_width - 1); p++){
		input_local[p] = input_local[p + input_width];
	}
	*/
}
        
}

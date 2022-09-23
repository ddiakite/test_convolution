#define input_height 512
#define input_width 512
#define mask_width 5

__kernel void convolution_2D(__global const float * restrict input,
				 __global float * restrict output,
				 __constant float * restrict mask
				)
{


const size_t radius = mask_width /2;

float input_local[input_width * (mask_width + radius)];
		
		
float output_local[input_width];

size_t cpt = mask_width - 1;

	size_t n = 0;
	//#pragma unroll 16
	for(unsigned int p = 0; p < input_width * (mask_width - 1); p++){
		if(p < radius * input_width){
			input_local[p] = 0.0f;
		}else{
			input_local[p] = input[n];
			++n;
		}
	}
#pragma max_concurrency 8
for(unsigned int y = 0; y < input_height; ++y) {

	size_t y_prime;
        y_prime = y + radius;
        
        //if(y < radius)
        //	y_prime = 0;
	
	if(cpt == (mask_width + radius)){
		cpt = radius;
	}
	
	#pragma unroll 16
	for(unsigned int p = 0; p < input_width; p++){
		input_local[(cpt * input_width) + p] = input[y_prime * input_width + p];
	}
	
	cpt++;
	
	
	
	#pragma unroll 64
	for(unsigned int p = 0; p < input_width; p++){
		output_local[p] = 0.0f;
	}
	
	#pragma max_concurrency input_width
	#pragma unroll 4
	for(unsigned int x = 0; x < input_width; ++x) {
		float result = 0.0f;
		size_t adr = 0;
		
		if(y<radius)
			adr = y;
		else
			adr = radius;
		
		//if((cpt > mask_width) && (y <= radius))
		//	adr = cpt - mask_width;
		

		#pragma unroll
		for(unsigned int my = 0; my < mask_width; ++my) {
		    #pragma unroll
		    for(unsigned int mx = 0; mx < mask_width; ++mx) {
			unsigned int mask_index = my * mask_width + mx;
			float image_value = 0.0f;
			if ((x + mx >= radius && x + mx - radius < input_width ))
			//if ((x + mx >= radius && x + mx - radius < input_width && y + my >= radius && y + my - radius < input_height))
			    //image_value = input[(y + my - radius) * input_width + (x + mx - radius)];
			    image_value = input_local[adr * input_width + (x + mx - radius)];
		
			result += mask[mask_index] * image_value;
		    }
		    adr++;
		}
		    
		    //output[y * input_width + x] = result;
		    output_local [x] = result;
		    
	}
	
	#pragma unroll 64
	for(unsigned int p = 0; p < input_width; p++){
		output[y * input_width + p] = output_local[p];
	}
	
	#pragma unroll 8
	for(unsigned int p = 0; p < input_width * (mask_width - 1); p++){
		//input_local[p] = input_local[p + input_width];
	}

}
        
        
}

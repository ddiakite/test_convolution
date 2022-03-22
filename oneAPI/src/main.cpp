#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <fstream>
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <cstdio>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

#include "FreeImage.h"
#include "ImageUtils.h"
#include "docopt.h"

#include "convolution.h"
#include "convolution_oneapi.hpp"
#include "convolution_cpu.hpp"
#include "utils.h"
#include "utils.hpp"

using namespace cl::sycl;
using namespace conv;



int main(int argc, char *argv[])
{
	printf("Starting main cpp\n");
	 
    	//if(argc < 2) {
	//	printf("Erreur format. Syntaxe correcte :\n Input image missing\n");
	//	return -1;
	//}
	
	using std::chrono::high_resolution_clock;
	using std::chrono::milliseconds;
	high_resolution_clock::time_point start_time, end_time;
	
	std::string image_path = "../images/cameraman.png";
	
	std::vector<elapsed_time_t> useless;
	
	size_t loop_count = 0;
	 
	size_t mask_width, mask_size, size;
	
	mixed_precision_tuple image_float;
	
		mask_width = 3;
		printf("Mask width: %zu\n", mask_width);
    	mask_size = mask_width * mask_width;
    	//mask_width = sqrt(mask_size);
    	const size_t radius {mask_width /2};
    	
    	mixed_precision_tuple mask;
		
    	
    	
    	std::get<single_float>(mask) = new float[mask_size];
    	std::fill_n(std::get<single_float>(mask), mask_size, 1.0f / mask_size);
    	

    	
    	FIBITMAP *image_source = FreeImage_Load(FIF_PNG, image_path.c_str(), PNG_DEFAULT);
        assert(image_source != nullptr);

        FIBITMAP *image_rgb = FreeImage_ConvertToRGB16(image_source);
        FreeImage_Unload(image_source);
        
        size_t width {FreeImage_GetWidth(image_rgb)};
    	size_t height {FreeImage_GetHeight(image_rgb)};
    	
    	printf("image width: %zu image height: %zu \n", width, height);
    	
    	std::get<single_float>(image_float) = ImageUtils_RGB16ToArray<float>(image_rgb);
    	
    	FreeImage_Unload(image_rgb);
    	
    	std::string base_path{image_path.substr(0, image_path.length() - 4)};
        std::string golden_path {base_path + "_golden_" + std::to_string(mask_width) + ".png"};
        std::string golden_path_raw {base_path + "_golden_" + std::to_string(mask_width) + ".raw"};
        
        start_time = high_resolution_clock::now();
    	
	float *image_ref = oneapi_convolution(std::get<single_float>(image_float), width, height, std::get<single_float>(mask), mask_width, useless, loop_count);
	
	end_time = high_resolution_clock::now();
	
	float *image_golden_float = cpu_convolution_naive(std::get<single_float>(image_float), width, height, std::get<single_float>(mask), mask_width, useless, loop_count);
	
	//printf("\n Convolution time: %0.3f ms\n", (end_time - start_time) * 1e3);
	milliseconds total_ms = std::chrono::duration_cast<milliseconds>(end_time - start_time);
	std::cout <<"CPU Convolution time:"  << total_ms.count() << "ms\n";
	
	
	export_to_raw(golden_path_raw.c_str(),image_golden_float,width*height);
	FIBITMAP *image_golden_rgb = ImageUtils_ArrayToRGB16(image_golden_float, width, height);
	FreeImage_Save(FIF_PNG, image_golden_rgb, golden_path.c_str(), PNG_DEFAULT);
	FreeImage_Unload(image_golden_rgb);
	

	 bool pass = true; 

        for(unsigned j = 0; j < (width * height) && pass; ++j) {
                if(fabsf(image_ref[j] - image_golden_float[j]) > 1.0e-6f) {
                        printf("Failed verification @ index %d\nOutput: %f\nReference: %f\n",
                        j, image_golden_float[j], image_ref[j]);
                       
                        pass = false;
                }

        }
        printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");

	
		

        delete[] image_golden_float;
        delete[] image_ref;
		delete[] std::get<single_float>(image_float); 
	
	return 0;
}

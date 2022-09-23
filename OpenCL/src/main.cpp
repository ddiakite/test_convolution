#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <chrono>
#include <iostream>
#include <map>
#include <cstring>
#include <tuple>
#include <fstream>

#include "FreeImage.h"
#include "ImageUtils.h"
#include "docopt.h"

#include "convolution.h"
#include "utils.h"
#include "clean.h"


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
	
	mask_width = 15;
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
    	
    	printf("mask width: %lu\n", mask_width);
    	printf("image width: %lu image height: %lu \n", width, height);
    	
    	std::get<single_float>(image_float) = ImageUtils_RGB16ToArray<float>(image_rgb);
    	
    	FreeImage_Unload(image_rgb);
    	
    	std::string base_path{image_path.substr(0, image_path.length() - 4)};
        std::string golden_path {base_path + "_golden_" + std::to_string(mask_width) + ".png"};
        std::string golden_path_raw {base_path + "_golden_" + std::to_string(mask_width) + ".raw"};
        
        start_time = high_resolution_clock::now();
    	
	float *image_ref = cpu_convolution_naive(std::get<single_float>(image_float), width, height, std::get<single_float>(mask), mask_width, useless, loop_count);
	
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

	//for(unsigned j = 0; j < (width * (height)) && pass; ++j) {
        for(unsigned j = (radius)*width; j < (width * (height-radius)) && pass; ++j) {
                if(fabsf(image_ref[j] - image_golden_float[j]) > 1.0e-9f) {
                        printf("Failed verification @ index %d\nOutput: %.9f\nReference: %.9f\n",
                        j, image_golden_float[j], image_ref[j]);
                       
                        pass = false;
                }

        }
        printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");

	
		
        delete[] image_golden_float;
        delete[] image_ref;
	
	delete[] std::get<single_float>(image_float); 



	// Running CLEAN Deconvolution

	float *dirty = new float[1280*1280];
	float *psf = new float[2560*2560];

	float *clean = new float[1280*1280];
	float *residuals = new float[1280*1280];

	float gamma = 0.1;
	float threshold = 0.204045191393;
	int niter = 1280;

	std::string dirty_path = "../images/dirty.dat";
	std::string psf_path = "../images/psf.dat";

	FILE* mydirtyfile;
	
	mydirtyfile = fopen(dirty_path.c_str(), "rb");
	
	if(mydirtyfile){
		std::cout << " dirty File opening ok \n ";
	}
	else{
		std::cout << "dirty File opening failed\n ";
	}
	
	fread((dirty), sizeof(float), 1280*1280, mydirtyfile);
	fclose (mydirtyfile);

	FILE* mypsffile;
	
	mypsffile = fopen(psf_path.c_str(), "rb");
	
	if(mypsffile){
		std::cout << " dirty File opening ok \n ";
	}
	else{
		std::cout << "dirty File opening failed\n ";
	}
	
	fread((psf), sizeof(float), 2560*2560, mypsffile);
	fclose (mypsffile);

	//std::cout << psf[1280+1280*2560] << "\n ";
	// for (size_t j = 0; j < 20; j++){

	// 	//printf("%f\t", dirty[j]);
	// 	std::cout << dirty[j] << "\n ";
	// }

	memcpy(residuals,dirty,1280*1280*sizeof(float));
	
	//hogbom_clean(dirty, psf, clean, residuals, gamma, threshold, niter);
	deconvolution_clean(dirty, psf, clean, residuals, gamma, threshold, niter);
	


	delete[] dirty;
	delete[] psf;
	delete[] clean;
	delete[] residuals;
	
	return 0;
}

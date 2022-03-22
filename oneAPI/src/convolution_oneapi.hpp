/*
** 2D convolution in OneAPI DPC++
* Author: Diakite Daouda
*/

#include "convolution.h"
//#include "convolution_oneapi.h"

#include <stddef.h>
#include <vector>

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <cstdio>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif


#define im_height 512
#define im_width 512
#define width_mask 3

using namespace cl::sycl;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};


class conv2D;

void conv_oneapi(queue &q, const float *input_array, float *output_array, const float *mask_array){

	range<1> num_items{im_width*im_height};
 	range<1> num_items_mask{width_mask*width_mask};
 	
 	buffer input_buf(input_array, num_items);
 	buffer output_buf(output_array, num_items);
 	buffer mask_buf(mask_array, num_items_mask);
 	
 	auto evt = q.submit([&](handler &h) {
 	
	 	accessor<float, 1, access::mode::read,
			 access::target::global_buffer> input(input_buf, h);
	    	accessor<float, 1, access::mode::read,
			 access::target::constant_buffer> mask(mask_buf, h);
			 
		accessor<float, 1, access::mode::write,
			 access::target::global_buffer> output(output_buf, h);
		
		
		h.single_task<class conv2D>([=]()[[intel::kernel_args_restrict]]{ 
			
			const size_t radius = width_mask /2;

			for(unsigned int y = 0; y < im_height; ++y) {
				  //#pragma unroll 4
				  for(unsigned int x = 0; x < im_width; ++x) {
					float result = 0.0f;

					#pragma unroll
					for(unsigned int my = 0; my < width_mask; ++my) {
						#pragma unroll
					    for(unsigned int mx = 0; mx < width_mask; ++mx) {
						unsigned int mask_index = my * width_mask + mx;
						float image_value = 0.0f;
						if (x + mx >= radius
						        && x + mx - radius < im_width
						        && y + my >= radius
						        && y + my - radius < im_height)
						    image_value = input[(y + my - radius) * im_width + (x + mx - radius)];
						result += mask[mask_index] * image_value;
					    }
					}
					output[y * im_width + x] = result;
				}
			}
		
		});
 	
 	});
 	
  auto start = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << "Kernel execution time: "
            << ((double)(end - start)) / 1000000.0 << " ms\n";
}


namespace conv {

template <typename T>
T *oneapi_convolution(const T *__restrict__ input, size_t input_width, size_t input_height, const T *__restrict__ mask, size_t mask_width, std::vector<elapsed_time_t> &results, size_t loop_count)
{
    // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

	float *output = new float[input_width*input_height];


	try {
	    sycl::property_list properties{sycl::property::queue::enable_profiling()};
	    queue q(d_selector, exception_handler, properties);
	    //sycl::queue q{sycl::property::queue::enable_profiling()};

	    // Print out the device information used for the kernel code.
	    std::cout << "Running on device: "
		      << q.get_device().get_info<info::device::name>() << "\n";
	    std::cout << "Input image  size: " << input_width*input_height << "\n";
	    std::cout << "Output image size: " << input_width*input_height << "\n";

	   
	    conv_oneapi(q, input, output, mask);

	  } catch (sycl::exception const &e) {
	    std::cout << "An exception is caught for vector add.\n";
	    std::terminate();
	  }


    return output;
}

INSTANTIATE(oneapi_convolution, float);
//INSTANTIATE(oneapi_convolution, double);

}

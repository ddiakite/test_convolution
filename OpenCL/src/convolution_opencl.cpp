/*
** 2D convolution in OpenCL
* Author: Diakite Daouda
*/

#include "convolution.h"

#include "CL/cl.h"
#include "AOCLUtils/aocl_utils.h"

#include <stddef.h>
#include <vector>

#define AOCL_ALIGNMENT 64

//#include "utils.h"


//Get platform and device information
static cl_platform_id *platform = NULL;
static cl_device_id device;
static cl_context context;
static cl_program program;
static cl_int ret;
static cl_command_queue queue;
static cl_kernel kernel;
cl_uint ret_num_platforms;
cl_uint num_devices_dilatation;
cl_event kernel_event;

using namespace aocl_utils;

#define CLEANUP()                                     \
    do {                                              \
        ret = clFlush(queue);             \
        ret = clFinish(queue);            \
        ret = clReleaseKernel(kernel);                \
        ret = clReleaseProgram(program);              \
        ret = clReleaseCommandQueue(queue);   \
        ret = clReleaseContext(context);              \
        free(source_str);                             \
    } while(0)
    

#define CHECK_RET(ret, str) \
do {                        \
    if(ret != CL_SUCCESS) {  \
        puts(str);            \
        printf("%d\n", ret);  \
    }                         \
} while(0)



namespace conv {

template <typename T>
T *opencl_convolution_naive(const T *__restrict__ input, size_t input_width, size_t input_height,
        const T *__restrict__ mask, size_t mask_width, std::vector<elapsed_time_t> &results, size_t loop_count)
{
    const size_t radius {mask_width /2};
    //T *output = new T[input_width * input_height];
    T * output = NULL;
    posix_memalign ((void **)&output, AOCL_ALIGNMENT, input_width * input_height * sizeof(T));
    
    T * input_aligned = NULL;
    posix_memalign ((void **)&input_aligned, AOCL_ALIGNMENT, input_width * input_height * sizeof(T));
    memcpy(input_aligned, input, input_width * input_height * sizeof(T));


    ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    platform = (cl_platform_id *) malloc(sizeof(cl_platform_id)*ret_num_platforms);

    ret = clGetPlatformIDs(ret_num_platforms, platform, NULL);
     if(platform == NULL) {
      printf("ERROR: Unable to find Altera OpenCL platform.\n");
      return output;
    }
    
    ret = clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_ALL, 1, &device, &num_devices_dilatation);
    
    printf("Found %d device(s)\n", num_devices_dilatation);

    /*platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    if(platform == NULL) {
      printf("ERROR: Unable to find Altera OpenCL platform.\n");
      return output;
    }
    

    // Query the available OpenCL device.
    cl_device_id *devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices_dilatation);
    printf("Platform: %s\n", getPlatformName(platform).c_str());
    printf("Found %d device(s)\n", num_devices_dilatation);
    
     // Just use the first device.
    device = devices[0];
    printf("Using %s\n", getDeviceName(device).c_str());
    delete[] devices;*/
    
    
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &ret);
    CHECK_RET(ret, "Could not create a valid OpenCL context");

    // Create command queue.
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
    CHECK_RET(ret, "Failed to create command queue");

    // Creation d'une variable d'evenement
    cl_event event = clCreateUserEvent(context, &ret);;
    
    // Create memory buffers on the device for the two images
    cl_mem input_img = clCreateBuffer(context, CL_MEM_READ_WRITE, input_width * input_height * sizeof(T),NULL,&ret);
    CHECK_RET(ret, "Unable to create the input_img buffer object");

    cl_mem output_img = clCreateBuffer(context, CL_MEM_READ_WRITE,input_width * input_height * sizeof(T),NULL,&ret);
    CHECK_RET(ret, "Unable to create the output_img buffer object");
    
    cl_mem mask_fpga = clCreateBuffer(context, CL_MEM_READ_WRITE,mask_width * mask_width * sizeof(T),NULL,&ret);
    CHECK_RET(ret, "Unable to create the mask_fpga buffer object");



    // Copy the image data to the memory buffer
    ret = clEnqueueWriteBuffer(queue, input_img, CL_TRUE, 0, input_width * input_height * sizeof(T), input_aligned, 0, NULL, NULL); 
    CHECK_RET(ret, "Error when copying the image data from the CPU to the FPGA OpenCL buffer");
    
    ret = clEnqueueWriteBuffer(queue, mask_fpga, CL_TRUE, 0, mask_width * mask_width * sizeof(T), mask, 0, NULL, NULL); 
    CHECK_RET(ret, "Error when copying the mask data from the CPU to the FPGA OpenCL buffer");

   // Create the program using binary already compiled offline using aoc (i.e. the .aocx file)
   
   //Lecture du .aocx
    unsigned char *source_str;
    size_t source_size;
    FILE *fp;
    char fileName[] = "kernel_convolution.aocx";
    fp = fopen(fileName, "rb");
    if (!fp) {
    	   fprintf(stderr, "Failed to load kernel.\n");
         exit(1);
    }
    fseek(fp, 0, SEEK_END);
    source_size = ftell(fp);
    rewind(fp);
    source_str = (unsigned char*) malloc(source_size * sizeof(unsigned char));
    if (fread(source_str, 1, source_size, fp) == 0) {
        puts("Could not read source file");
        exit(-1);
    }
    printf("Taille du binaire : %lu bytes\n", source_size);
    fclose(fp);
   
   cl_int kernel_status;
   // Create the program.
   //program = clCreateProgramWithSource(context, 1,(const char **)&source_str, NULL, &ret);
   program = clCreateProgramWithBinary(context, 1, &device, (const size_t *)&source_size, (const unsigned char **)&source_str, &kernel_status, &ret);
   CHECK_RET(ret, "source failed\n");

   if (ret != CL_SUCCESS) {
        puts("Could not create from binary");
        CLEANUP();
		exit(0);
   }
   

    /*std::string binary_file = getBoardBinaryFile("kernel_convolution", device);
    printf("Using AOCX: %s\n", binary_file.c_str());

    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	// build the program
    ret = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    CHECK_RET(ret, "Failed to build program");*/

	// Create the OpenCL kernel. This is basically one function of the program declared with the __kernel qualifier
    kernel = clCreateKernel(program, "convolution_2D", &ret);
    CHECK_RET(ret, "Failed to create the OpenCL Kernel from the built program");

	// Set the arguments of the kernel
	
    int argk = 0;
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), (void *)&input_img); 
    CHECK_RET(ret, "Could not set the kernel's \"input_img\" argument\n");

    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), (void *)&output_img);
    CHECK_RET(ret, "Could not set the kernel's \"output_img\" argument\n");
    
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), (void *)&mask_fpga);
    CHECK_RET(ret, "Could not set the kernel's \"mask_fpga\" argument\n");

   //enqueue the kernel into the OpenCL device for execution
   size_t globalWorkItemSize = input_width * input_height;//the total size of 1 dimension of the work items. Basically the whole image buffer size
   size_t workGroupSize = 64; //The size of one work group
   
   double average_time =0.0;
   for(size_t n = 0; n <= loop_count; n++){
   
	   //ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkItemSize, &workGroupSize, 1, &event, &kernel_event);

	   ret = clEnqueueTask(queue, kernel, 1, &event, &kernel_event);

	   clSetUserEventStatus(event, CL_COMPLETE);
	   clWaitForEvents(1, &kernel_event);
	   
	   double elapsed = 0;
	   cl_ulong time_start, time_end;
	   
	   clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
	   clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
	   elapsed = (time_end - time_start);
   	   average_time += elapsed;
   }
   
   average_time = average_time / (loop_count + 1);
   
   ///Read the memory buffer of the new image on the device to the new Data local variable
   ret = clEnqueueReadBuffer(queue, output_img, CL_TRUE, 0, input_width * input_height * sizeof(T), output, 0, NULL, NULL);
   
    
    printf("FPGA Convolution time: %0.3f ms \n", average_time * 1e-6);
    printf("Throughput = %0.3f FPS\n", (double)(1 / (average_time * 1e-9)));
   
    ///Clean up everything
    
    
    clReleaseMemObject(input_img);
    clReleaseMemObject(output_img);
    clReleaseMemObject(mask_fpga);
    CLEANUP();  
    
    return output;
}

INSTANTIATE(opencl_convolution_naive, float);
INSTANTIATE(opencl_convolution_naive, double);

}

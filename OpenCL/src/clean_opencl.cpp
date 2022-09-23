#include "CL/cl.h"
#include "AOCLUtils/aocl_utils.h"

#include <stddef.h>
#include <vector>

#define AOCL_ALIGNMENT 64


#include "clean.h"
#include <iostream>
#include <cmath>
#include <tuple>
#include <cstring>
#include <algorithm>
#include <stdio.h>
//#include <dlib/optimization.h>

#define Width 1280
#define Height 1280

#define psfWidth 2560
#define psfHeight 2560



using namespace std;
//using namespace dlib;

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



void deconvolution_clean(float *dirty, float *psf, float *clean, float *residuals, float gamma, float threshold, int niter){

    printf("Clean Deconvolution on OpenCL FPGA\n");

    ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    platform = (cl_platform_id *) malloc(sizeof(cl_platform_id)*ret_num_platforms);

    ret = clGetPlatformIDs(ret_num_platforms, platform, NULL);
     if(platform == NULL) {
      printf("ERROR: Unable to find Altera OpenCL platform.\n");
      //return output;
    }
    
    ret = clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_ALL, 1, &device, &num_devices_dilatation);
    
    printf("Found %d device(s)\n", num_devices_dilatation);


    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &ret);
    CHECK_RET(ret, "Could not create a valid OpenCL context");

    // Create command queue.
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
    CHECK_RET(ret, "Failed to create command queue");

    // Creation d'une variable d'evenement
    cl_event event = clCreateUserEvent(context, &ret);;
    
    // Create memory buffers on the device for the two images
    cl_mem dirty_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Height * Width * sizeof(float),NULL,&ret);
    CHECK_RET(ret, "Unable to create the input_img buffer object");

    cl_mem psf_d = clCreateBuffer(context, CL_MEM_READ_WRITE, psfHeight * psfWidth * sizeof(float),NULL,&ret);
    CHECK_RET(ret, "Unable to create the input_img buffer object");

    cl_mem clean_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Height * Width * sizeof(float),NULL,&ret);
    CHECK_RET(ret, "Unable to create the input_img buffer object");

    cl_mem residuals_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Height * Width * sizeof(float),NULL,&ret);
    CHECK_RET(ret, "Unable to create the input_img buffer object");


    // Copy the image data to the memory buffer
    ret = clEnqueueWriteBuffer(queue, dirty_d, CL_TRUE, 0, Height * Width * sizeof(float), dirty, 0, NULL, NULL); 
    CHECK_RET(ret, "Error when copying the image data from the CPU to the FPGA OpenCL buffer");
    
    ret = clEnqueueWriteBuffer(queue, psf_d, CL_TRUE, 0, psfHeight * psfWidth * sizeof(float), psf, 0, NULL, NULL); 
    CHECK_RET(ret, "Error when copying the mask data from the CPU to the FPGA OpenCL buffer");

    ret = clEnqueueWriteBuffer(queue, residuals_d, CL_TRUE, 0, Height * Width * sizeof(float), residuals, 0, NULL, NULL); 
    CHECK_RET(ret, "Error when copying the mask data from the CPU to the FPGA OpenCL buffer");


    //Lecture du .aocx
    unsigned char *source_str;
    size_t source_size;
    FILE *fp;
    char fileName[] = "kernel_clean.aocx";
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
   

    kernel = clCreateKernel(program, "hogbom_clean", &ret);
    CHECK_RET(ret, "Failed to create the OpenCL Kernel from the built program");

    // Set the arguments of the kernel
    int num = 0;
    int argk = 0;
    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), (void *)&dirty_d); 
    CHECK_RET(ret, "Could not set the kernel's \"input_img\" argument\n");

    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), (void *)&psf_d);
    CHECK_RET(ret, "Could not set the kernel's \"output_img\" argument\n");

    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), (void *)&clean_d);
    CHECK_RET(ret, "Could not set the kernel's \"mask_fpga\" argument\n");

    ret = clSetKernelArg(kernel, argk++, sizeof(cl_mem), (void *)&residuals_d);
    CHECK_RET(ret, "Could not set the kernel's \"mask_fpga\" argument\n");

    ret = clSetKernelArg(kernel, argk++, sizeof(float), &gamma);
    CHECK_RET(ret, "Could not set the kernel's \"mask_fpga\" argument\n");

    ret = clSetKernelArg(kernel, argk++, sizeof(float), &threshold);
    CHECK_RET(ret, "Could not set the kernel's \"mask_fpga\" argument\n");

    ret = clSetKernelArg(kernel, argk++, sizeof(int), &niter);
    CHECK_RET(ret, "Could not set the kernel's \"mask_fpga\" argument\n");

    //enqueue the kernel into the OpenCL device for execution
    size_t globalWorkItemSize[] = {1280, 1280};//the total size of 1 dimension of the work items. Basically the whole image buffer size
    size_t workGroupSize[] = {1280,1}; //The size of one work group

    double average_time =0.0;
    size_t loop_count = 1;
    for(size_t n = 0; n < loop_count; n++){

        ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkItemSize, workGroupSize, 1, &event, &kernel_event);

        //ret = clEnqueueTask(queue, kernel, 1, &event, &kernel_event);

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
    //ret = clEnqueueReadBuffer(queue, clean_d, CL_TRUE, 0, Height * Width * sizeof(float), clean, 0, NULL, NULL);
    //ret = clEnqueueReadBuffer(queue, residuals_d, CL_TRUE, 0, Height * Width * sizeof(float), residuals, 0, NULL, NULL);

    printf("FPGA CLEAN Deconvolution time: %0.3f ms \n", average_time * 1e-6);

    //printf("Cleaning done after %d iterations.\n", num);

    clReleaseMemObject(dirty_d);
    clReleaseMemObject(psf_d);
    clReleaseMemObject(clean_d);
    clReleaseMemObject(residuals_d);

    CLEANUP();

}
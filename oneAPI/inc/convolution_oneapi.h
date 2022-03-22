#pragma once

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
//#include <fstream>
#include <cstdio>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

void BPVI(queue &q, float *input, float *output, float *mask);

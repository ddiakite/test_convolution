#pragma once

#include <math.h>
#include <stddef.h>
#include <fstream>

#ifndef MIN
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#endif

size_t upper_power_of_two(size_t x);

template <typename T>
T ceili(T a, T b)
{
    return (a + b - 1) / b;
}

// template <typename T>
// double compare_images(T *first, T *second, size_t width, size_t height,
// std::ostream* errorwriter=0, double local_threshold=INFINITY, double global_threshold=INFINITY);

void openofstream(std::ofstream& s,const char* fileName);

template<class T> void export_to_raw(const char* fileName,T* array,size_t length){
	std::ofstream file;
	openofstream(file,fileName);
	for(int i=0;i<length;i++){
		file.write((char*)(array+i),sizeof(T));
	}
}

template<class OutputType,class InputType=OutputType> void import_from_raw(const char* fileName,OutputType* array,size_t length){
	std::ifstream file(fileName,std::ios::binary);
	file.exceptions(file.failbit|file.badbit);
	for(int i=0;i<length;i++){
		InputType tmp;
		file.read((char*)&tmp,sizeof(InputType));
		array[i]=(OutputType)tmp;
	}
}


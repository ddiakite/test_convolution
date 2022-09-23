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



using namespace std;
//using namespace dlib;


struct float64
{
    float x;
    float y;
};



void twod_gaussian(int *x, int *y, float *g, float amplitude, int x0, int y0, float sigma_x, float sigma_y, int theta, float offset, int size){


    float xo = float(x0);
    float yo = float(y0);

    float a = (cos(theta)*cos(theta))/(2*sigma_x*sigma_x) + (sin(theta)*sin(theta))/(2*sigma_y*sigma_y);
    float b = -(sin(2*theta))/(4*sigma_x*sigma_x) + (sin(2*theta))/(4*sigma_y*sigma_y);
    float c = (sin(theta)*sin(theta))/(2*sigma_x*sigma_x) + (cos(theta)*cos(theta))/(2*sigma_y*sigma_y);

    for(int j = 0; j < size; j++){
        for(int i = 0;i < size; i++){

            g[i + j * size] = (offset + amplitude * exp(- (a*((x[i]-xo)*(x[i]-xo)) + 2*b*(x[i]-xo)*(y[j]-yo) + c*((y[j]-yo)*(y[j]-yo)))));
        
        }
    }

    

}


float fit_2d_gaussian(float *psf, float *psf_fit, float *data_fitted, int size){

    float *x = new float[size];
    float *y = new float[size];

    int lk = sqrt(size);
    int mk = lk; 

    float value = 0.0;
    for(int i = 0; i < size; i++){

        x[i] = value;
        y[i] = value;

        value = value + 1 / (size-1);
    }

    float initial_guess[7] = {0.5, lk/2, mk/2, 1.75, 1.4, -4.0, 0};

    //isotonic_regression.fit_with_linear_output_interpolation()

    //twod_gaussian(x, y, data_fitted);

}



std::tuple<int, int, int, int, float> find_peak(float *residuals){


    // struct retVals {        // Declare a local structure 
    //     int minx, miny, maxx, maxy, peak_intensity;
    // };


    float abs_residuals[Height*Width];
    memcpy(abs_residuals,residuals,Height*Width*sizeof(float));

    //float minmax;
    std::pair<float*, float*> minmax = std::minmax_element(std::begin(abs_residuals), std::end(abs_residuals));
    float min_peak = *minmax.first; // min abs_residuals
    float max_peak = *minmax.second; //max abs_residuals

    std::cout <<"min_peak "  << min_peak << " max_peak "  << max_peak  << " peak "  << "\n";


    //printf("min_peak = %f, max_peak = %f value = %f valuer = %f\n", min_peak, max_peak, residuals[20000], residuals[20000]);

    float intensity, peak_intensity;
    int minx, miny, maxx, maxy;
    int nx, ny;
    minx = miny = maxx = maxy = -1;
    peak_intensity = -1.0;

    nx = Height;
    ny = Width;

    // for(int j = 0; j < 1; j++){
    //     for (int i = 0; i < 20; i++)
    //     {
    //         printf("%f\t", abs_residuals[i+j*Width]);
    //     }
        
    // }
    //printf("Before loop \n");
    for(int y = 0; y < ny; y++){
        for(int x = 0; x < nx; x++){

            intensity = abs_residuals[x + y * nx];
            //printf("%f\t", intensity);

            if(intensity == min_peak){
                minx = x;
                miny = y;
                //printf("Minimum peak found\t");
            }
            
            if(intensity == max_peak){
                maxx = x;
                maxy = y;
                peak_intensity = intensity;
                //printf("Maximum peak found: %f\t", peak_intensity);
            }

        // if(minx == -1 || miny == -1){
        //     printf("Minimum peak not found\t");
        //     //minx = 0; miny = 0;
        // }

        // if(maxx == -1 || maxy == -1){
        //     printf("Maximum peak not found\t");
        //     //maxx = 0; maxy = 0;
        // }

        }
    }
    
    //delete[] abs_residuals; 
    //printf("After loop \n");
    return std::make_tuple(maxx, maxy, minx, miny, peak_intensity);
}


void build_cleanmap(float *clean, float intensity, float gamma, int p, int q){
    clean[p + q*Height] += intensity*gamma;
}


void update_residual(float *residuals, float intensity, float gamma, int p, int q, int npix, float *psf){
    npix = Width; //residual Width
    int m = 0;
    for(size_t j = npix -1 - q; j < 2*npix-1-q; j++){
        int n = 0;
        for(size_t i = npix - 1 - p; i < 2*npix-1-p; i++){
            
            residuals[n + m*Height] -= gamma*intensity*psf[i + j*2*Height];
            //std::cout << residuals[p + q*Height] << "\n ";
            n+=1;
        }
       m+=1; 
    }
    
    
}





void hogbom_clean(float *dirty, float *psf, float *clean, float *residuals, float gamma, float threshold, int niter){

    // Performs Hogbom Clean on the  ``dirty`` image given the ``psf``.
    int p = 0, q  = 0, pmin = 0, qmin = 0, npix = 0, none1=0, none2=0;
    float intensity = 0.0;
    //int npix = 0;

    //memcpy(residuals,dirty,Height*Width*sizeof(float));
    //printf("Before Calling find_peak \n");

    std::tie(p, q, pmin, qmin, intensity) = find_peak(residuals);

    int i = 0;
    //std::cout <<"x "  << p << " y "  << q  << " peak "  << intensity  << " threshold "  << threshold  << " iter "  << i << "\n";
    //printf("Before while loop \n");
    while(abs(intensity) > threshold && i <= niter){

        build_cleanmap(clean, intensity, gamma, p, q);

        update_residual(residuals, intensity, gamma, p, q, npix, psf);

        std::tie(p, q, none1, none2, intensity) = find_peak(residuals);

        if (i%100 == 0)
        {
            std::cout <<"x "  << p << " y "  << q  << " peak "  << intensity  << " threshold "  << threshold  << " iter "  << i << "\n";
        }
        //printf("x %d y %d peak %f threshold %f iter %i\n", p, q, intensity, threshold, i);
        //std::cout <<"x "  << p << " y "  << q  << " peak "  << intensity  << " threshold "  << threshold  << " iter "  << i << "\n";
        
        i+=1;
        

        if (i > niter)
        {
            printf("Number of iterations exceeded\n");
        }

       
        
        
    }
    printf("Cleaning done after %d iterations.\n", i);
}
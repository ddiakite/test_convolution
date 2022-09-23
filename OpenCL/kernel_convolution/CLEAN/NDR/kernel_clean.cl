#define Width 1280
#define Height 1280

#define psfWidth 2560
#define psfHeight 2560

float2 min_max(float * input){
    float2 value;
    float min, max;

    min = input[0];
    max = input[1];  

    if(input[0] > input[1]){
        min = input[1];
        max = input[0];
    }

    //for(int j = 0; j < Height; j++){
    for(int i = 2; i < Width*Height; i++){

        if(input[i] > max){
            max = input[i];
        }

        if(input[i] < min){
            min = input[i];
        }

    }

    //}

    value.s0 = min;
    value.s1 = max;

    return value;
}

//__attribute__((num_simd_work_items(4)))
//__attribute__((num_compute_units(8)))
__attribute__((reqd_work_group_size(1280,1,1)))
__kernel void hogbom_clean(__global float * restrict dirty,
                            __global float * restrict psf,
                            __global float * restrict clean,
                            __global float * restrict residuals,
                            float gamma,
                            float threshold,
                            int niter)
{

    int p = 0, q  = 0, pmin = 0, qmin = 0, npix = 0, none1=0, none2=0;
    float intensity = 0.0;

    int x, y;

    x = get_global_id(0);
    y = get_global_id(1);

    // Finding residuals peak value 
    //float abs_residuals[Height*Width];
    //float2 min_max_value;

    float local_max[Width];
    float local_min[Width];

    float min, max;
    //float minmax;
    //std::pair<float*, float*> minmax = std::minmax_element(std::begin(abs_residuals), std::end(abs_residuals));
    float min_peak; //= *minmax.first; // min abs_residuals
    float max_peak; //= *minmax.second; //max abs_residuals
    //min_max_value = min_max(residuals);
    //if(y == 0){
    
    min = residuals[0];
    max = residuals[1];  

    if(residuals[0] > residuals[1]){
        min = residuals[1];
        max = residuals[0];
    }
if(y==0){
    for(int j = 2; j < Height; j++){
    //for(int l = 2; l < Width*Height; l++){
        int id = j + x*Width;

        if(residuals[id] > max){
            max = residuals[id];
        }

        if(residuals[id] < min){
            min = residuals[id];
        }
    
    }
    
    local_max[x] = max;
    local_min[x] = min;
    //printf("%f\t", local_min[x]);
}
    //}

    barrier(CLK_LOCAL_MEM_FENCE);

    if(x == 0 && y == 0){ //single threaded

    max = local_max[0];
    min = local_min[0];
    

    for(int index = 1; index < Height; index++){
    //for(int l = 2; l < Width*Height; l++){
       

        if(local_max[index] > max){
            max = local_max[index];
        }

        if(local_min[index] < min){
            min = local_min[index];
        }
        //printf("%f\t", local_max[5]);
    
    }
    
    min_peak = min;
    max_peak = max;

    }

    

    float intensity_local, peak_intensity;
    int minx, miny, maxx, maxy;
    int nx, ny;
    minx = miny = maxx = maxy = -1;
    peak_intensity = -1.0;

    nx = Height;
    ny = Width;

    //find_peak(residuals);

    // for(int y = 0; y < ny; y++){
    //     for(int x = 0; x < nx; x++){

            intensity_local = residuals[x + y * nx];
            

            if(intensity_local == min_peak){
                minx = x;
                miny = y;
            }
            
            if(intensity_local == max_peak){
                maxx = x;
                maxy = y;
                peak_intensity = intensity_local;
            }

    //     }
    // } 

    p = maxx; q = maxy; pmin = minx; qmin = miny; intensity = peak_intensity;
    
    barrier(CLK_GLOBAL_MEM_FENCE);


    int i = 0;

    // while(fabs(intensity) > threshold && i <= niter){ //fabs(intensity) > threshold && 


    //     // Building Clean map
    //     //build_cleanmap(clean, intensity, gamma, p, q);
    //     clean[p + q*Height] += intensity*gamma;


    //     // Updating residuals
    //     //update_residual(residuals, intensity, gamma, p, q, npix, psf);
    //     int npix = Width; //residual Width
    //     // int m = 0;
    //     // for(size_t j = npix -1 - q; j < 2*npix-1-q; j++){
    //     //     int n = 0;
    //     //     for(size_t i = npix - 1 - p; i < 2*npix-1-p; i++){
    //     int k = npix -1 - p + x;
    //     int l = npix -1 - q + y;
                
    //             residuals[x + y*Height] -= gamma*intensity*psf[k + l*psfHeight];
                
    //     //         n+=1;
    //     //     }
    //     //     m+=1; 
    //     // }

    //     // Finding residuals peak value  
    //     //find_peak(residuals);
    //     min = residuals[0];
    //     max = residuals[1];  

    //     if(residuals[0] > residuals[1]){
    //         min = residuals[1];
    //         max = residuals[0];
    //     }

    //     //for(int j = 0; j < Height; j++){
    //     //for(int l = 2; l < Width*Height; l++){

    //         if(residuals[x + y*Width] > max){
    //             max = residuals[x + y*Width];
    //         }

    //         if(residuals[x + y*Width] < min){
    //             min = residuals[x + y*Width];
    //         }

    //     //}
        
    //     float min_peak = min;
    //     float max_peak = max; 

    //     // for(int y = 0; y < ny; y++){
    //     //     for(int x = 0; x < nx; x++){

    //             intensity_local = residuals[x + y * nx];
                

    //             if(intensity_local == min_peak){
    //                 minx = x;
    //                 miny = y;
    //             }
                
    //             if(intensity_local == max_peak){
    //                 maxx = x;
    //                 maxy = y;
    //                 peak_intensity = intensity_local;
    //             }

    //     //     }
    //     // } 

    //     p = maxx; q = maxy; intensity = peak_intensity;

    //     barrier(CLK_GLOBAL_MEM_FENCE);
        
    //     // if(i%100==0)
    //     //     printf("x %d y %d peak %f threshold %f iter %i\n", p, q, intensity, threshold, i);
           

    //     i+=1;
        
    // }
    if(x==0 && y==0){
        printf("Cleaning done after %f iterations.\n", min_peak);
    }

}

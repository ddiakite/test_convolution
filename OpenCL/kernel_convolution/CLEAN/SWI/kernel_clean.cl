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

    // Finding residuals peak value 
    //float abs_residuals[Height*Width];
    //float2 min_max_value;

    float min, max;
    //min_max_value = min_max(residuals);
    min = residuals[0];
    max = residuals[1];  

    if(residuals[0] > residuals[1]){
        min = residuals[1];
        max = residuals[0];
    }

    //for(int j = 0; j < Height; j++){
    for(int i = 2; i < Width*Height; i++){

        if(residuals[i] > max){
            max = residuals[i];
        }

        if(residuals[i] < min){
            min = residuals[i];
        }

    }
    //float minmax;
    //std::pair<float*, float*> minmax = std::minmax_element(std::begin(abs_residuals), std::end(abs_residuals));
    float min_peak = min; //= *minmax.first; // min abs_residuals
    float max_peak = max; //= *minmax.second; //max abs_residuals

    float intensity_local, peak_intensity;
    int minx, miny, maxx, maxy;
    int nx, ny;
    minx = miny = maxx = maxy = -1;
    peak_intensity = -1.0;

    nx = Height;
    ny = Width;

    //find_peak(residuals);

    for(int y = 0; y < ny; y++){
        for(int x = 0; x < nx; x++){

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

        }
    } 

    p = maxx; q = maxy; pmin = minx; qmin = miny; intensity = peak_intensity;
    
    int i = 0;

    while(fabs(intensity) > threshold && i <= niter){ //fabs(intensity) > threshold && 


        // Building Clean map
        //build_cleanmap(clean, intensity, gamma, p, q);
        clean[p + q*Height] += intensity*gamma;


        // Updating residuals
        //update_residual(residuals, intensity, gamma, p, q, npix, psf);
        int npix = Width; //residual Width
        int m = npix -1 - q;
        for(size_t j = 0; j < npix; j++){
            int n = npix - 1 - p;
            for(size_t i = 0; i < npix; i++){
                
                residuals[i + j*Height] -= gamma*intensity*psf[n + m*psfHeight];
                
                n+=1;
            }
            m+=1; 
        }


        // Finding residuals peak value  
        //find_peak(residuals);
        min = residuals[0];
        max = residuals[1];  

        if(residuals[0] > residuals[1]){
            min = residuals[1];
            max = residuals[0];
        }

        //for(int j = 0; j < Height; j++){
        for(int i = 2; i < Width*Height; i++){

            if(residuals[i] > max){
                max = residuals[i];
            }

            if(residuals[i] < min){
                min = residuals[i];
            }

        }
        
        float min_peak = min;
        float max_peak = max; 

        for(int y = 0; y < ny; y++){
            for(int x = 0; x < nx; x++){

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

            }
        } 

        p = maxx; q = maxy; intensity = peak_intensity;

        // if(i%100==0)
        //     printf("x %d y %d peak %f threshold %f iter %i\n", p, q, intensity, threshold, i);

        i+=1;
        
    }
    //printf("Cleaning done after %d iterations.\n", i);

}

#pragma once

#include <chrono>
#include <stddef.h>
#include <tuple>
#include <vector>



void twod_gaussian(int *x, int *y, float *g, float amplitude, int x0, int y0, float sigma_x, float sigma_y, int theta, float offset, int size);

float fit_2d_gaussian(float *psf, float *psf_fit, float *data_fitted, int size);

std::tuple<int, int, int, int, float> find_peak(float *residuals);

void build_cleanmap(float *clean, float intensity, float gamma, int p, int q);

void update_residual(float *residuals, float intensity, float gamma, int p, int q, int npix, float *psf);

void hogbom_clean(float *dirty, float *psf, float *clean, float *residuals, float gamma, float threshold, int niter);

void deconvolution_clean(float *dirty, float *psf, float *clean, float *residuals, float gamma, float threshold, int niter);
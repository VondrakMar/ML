#ifndef utils_H
#define utils_H
#include <iostream>
#include <limits> // for smallest numbers
double eval_loos(double ref,double pred);
size_t find_min(size_t N_data,double* data);
size_t find_max(size_t N_data,double* data);
double* normalized_data(size_t N_data,double* data);
double* denormalized_data(size_t N_data, double* norm_data, double original_min, double original_max);

#endif //utilsNN_H

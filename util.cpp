#include "util.hpp"

class MSE{
    MSE();
    double eval_loos(double ref,double pred);
};

MSE::MSE(){
    
}

double eval_loos(double ref,double pred){
    double diff = ref - pred;
    double diff2 = diff*diff;
    return diff2;
}

size_t find_min(size_t N_data,double* data){
    size_t min_pos = 0;
    double min = std::numeric_limits<double>::max();
    for (size_t i = 0 ; i < N_data ; i++){
        if (data[i] < min){
            min = data[i];
            min_pos = i;
        }
    }
    return min_pos;
}


size_t find_max(size_t N_data,double* data){
    size_t max_pos = 0;
    double max = std::numeric_limits<double>::lowest();
    for (size_t i = 0 ; i < N_data ; i++){
        if (data[i] > max){
            max = data[i];
            max_pos = i;
        }
    }
    return max_pos;
}

// def normalize_data(data,min_value,max_value):
    // normalized_data = (data - min_value) / (max_value - min_value)
    // return normalized_data
double* normalized_data(size_t N_data,double* data){
    double* norm_data = new double[N_data];
    int min_pos = find_min(N_data,data);
    int max_pos = find_max(N_data,data);
    double max_min = data[max_pos] - data[min_pos];
    for (size_t i = 0; i < N_data ; i++){
        norm_data[i] = (data[i]-data[min_pos])/(max_min);
    }
    return norm_data;
}

double* denormalized_data(size_t N_data, double* norm_data, double original_min, double original_max) {
    double max_min = original_max - original_min;
    double* denorm_data = new double[N_data];
    for (size_t i = 0; i < N_data; i++) {
        denorm_data[i] = norm_data[i] * max_min + original_min;
    }
    return denorm_data;
}

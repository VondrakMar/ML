#include <iostream>
#include <cmath>
#include <limits> // for smallest numbers
#include "NN.hpp"


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

int main(){
    size_t N_train = 10;
    double x1[N_train];
    double x2[N_train];
    double y[N_train];
    for (size_t d = 0; d < N_train ; d++){
        x1[d] = random_number(-10,10);
        x2[d] = random_number(-10,10);
        // y[d] = 3*x1[d] + 5*x2[d];
        y[d] = sin(x1[d]) + 5*x2[d];
    }
    auto norm_x1 = normalized_data(N_train,x1);
    auto norm_x2 = normalized_data(N_train,x2);
    auto norm_y = normalized_data(N_train,y);
    size_t number_of_layers = 3;
    size_t size_first_layer = 2;
    size_t NN_layer[number_of_layers] = {size_first_layer,5,1};
    auto my_NN = NN(number_of_layers,NN_layer);
    double inputs[size_first_layer] = {norm_x1[0],norm_x2[0]};
    double output_min = y[find_min(N_train,y)];
    double output_max = y[find_max(N_train,y)];
    double output_max_min = output_max-output_min;
    for (int i =0 ; i < 500 ; i++){
        my_NN.forward(inputs);
        my_NN.backward(norm_y);
        my_NN.update_weights(0.1);
        std::cout << "output " << my_NN.get_output()*output_max_min+output_min << " " << y[0] << std::endl;
        my_NN.zero_grads();
    }
    return 0;
}

/*
    size_t N_l = 2;
    size_t NN_layer[N_l] = {2,5,1};
    auto my_NN = NN(N_l,NN_layer);
    size_t First_layer = 2;
    // double inputs[First_layer] = {2.3};//,5.1};
    double inputs[First_layer] = {2,1};//,5.1};
    for (int i =0 ; i < 100 ; i++){
        my_NN.forward(inputs);
        my_NN.backward(0.6);
        my_NN.update_weights(0.1);
        std::cout << "output " << my_NN.get_output() << " " << my_NN.layers[0]->w[0][0] << std::endl;
        my_NN.zero_grads();
    }
    my_NN.print_NN();
    return 0;
}
*/


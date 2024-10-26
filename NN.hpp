#include <iostream>

#ifndef NN_H
#define NN_H

void print_vector(double* vector, size_t sample_size);
double sigmoid(double z);
double dsigmoid(double z);
double random_number(double r_down,double r_up);

class Layer{
 public:
    Layer(size_t N_size);
    // ~Layer();
    void print_layer();
    size_t N_size;
    double* a;
    double* delta; 
    double* b;
    double** w;
    double** dw;
};

class NN{
public:
    NN(size_t N_layers, size_t* layer_sizes);
    // ~NN();
    void print_NN();
    void forward(double* input);
    void backward(double output_ref);
    void update_weights(double lr);
    void print_weights();
    void zero_grads();
    double get_output();
    size_t N_layers;
    Layer** layers;
};



#endif // NN_H
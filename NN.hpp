#ifndef NNtest_H
#define NNtest_H
#include <iostream>

void print_vector(double* vector, size_t sample_size);
double sigmoid(double z);
double dsigmoid(double z);
double random_number(double r_down,double r_up);


class NN{
public:
    NN(size_t N_layers, size_t* layer_sizes);
    double* forward(double* input);
    void train(double* input, double* output,double lr,size_t epochs);
    void print_weights();
    void print_NN_scheme(double* input);
    // ~NN(); //descturctor
private:
    size_t num_of_layers;
    size_t* layers_sizes;
    double ***weights;
    double** weights_init(size_t layer0, size_t layer1);
    // void weights_dealoc();
};



#endif // NN_H

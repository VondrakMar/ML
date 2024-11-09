#include <cmath>

#include "NN.hpp"
void print_vector(double* vector, size_t sample_size){
    for (size_t i = 0; i < sample_size ; i++){
        printf("%f ",vector[i]);
    }
    printf("\n");
}


double sigmoid(double z){
    return 1/(1+std::exp(-z));
}

double dsigmoid(double z){
    double sig = sigmoid(z);
    return sig * (1 - sig);
}


double random_number(double r_down,double r_up){
    double r = (double)(rand())/RAND_MAX;
    r = r_down + r*(r_up-r_down);
    return r;
}


Layer::Layer(size_t n_size){
    N_size = n_size;
    a = new double[N_size];
    b = new double[N_size];
    delta = new double[N_size];
    w = new double*[N_size];
    dw = new double*[N_size];
    for (size_t n = 0; n < N_size; n++){
        // z[n] = random_number(-2,2);
        a[n] = 0;// random_number(-2,2);
        b[n] = random_number(-2,2);
        delta[n] = 0;
        w[n] = nullptr;
        dw[n] = nullptr;
    }
}

void Layer::print_layer(){
    for (size_t l = 0; l < N_size; l++){
        std::cout << "b=" << b[l] << ", z=" << a[l] << std::endl;
    }
}



NN::NN(size_t n_layers, size_t* layer_sizes){
    // n_layers is a number of how many layers therre will be
    // layer_sizes is number of nodes in each layer. This also includes input and output layer, where you can set up input size and output size
    // layer sizes {1,1} is just 1 input layer and 1 output layer with 1 weight
    N_layers = n_layers; // number of layers
    layers = new Layer*[N_layers]; // allocating memory for pointers to layers objects
    
    for (size_t l = 0 ; l < N_layers-1; l++){
        layers[l] = new Layer(layer_sizes[l]);
        size_t current_layer_size = layers[l]->N_size;
        for (size_t n = 0; n < current_layer_size ; n++){
            layers[l]->w[n] = new double[layer_sizes[l+1]];
            layers[l]->dw[n] = new double[layer_sizes[l+1]];
            for (size_t n2 = 0; n2 < layer_sizes[l+1] ; n2++){
                layers[l]->w[n][n2] = random_number(-2,2);
                layers[l]->dw[n][n2] = 0;
            }
        }
    }
    layers[N_layers-1] = new Layer(layer_sizes[N_layers-1]);
}

NN::NN(){
   
}

void NN::print_NN(){
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n" << "Printing NN with " << this->N_layers << " layers\n" << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n" ;
    for (size_t l = 0; l < this->N_layers-1; l++){
        std::cout << "###################Layer " << l << " ############\n";
        for (size_t n = 0; n < this->layers[l]->N_size ; n++){
            std::cout << "for forward: b=" << this->layers[l]->b[n] << ", a=" << this->layers[l]->a[n] << std::endl;
            std::cout << "for backward: delta=" << this->layers[l]->delta[n] << std::endl;
            // std::cout << "this->layers[l+1]->N_size" << this->layers[l+1]->N_size << std::endl;
            for (size_t n2 = 0; n2 < this->layers[l+1]->N_size ; n2++){
                std::cout << this->layers[l]->w[n][n2] << " ";
            }
            std::cout << std::endl << "dws: ";
            for (size_t n2 = 0; n2 < this->layers[l+1]->N_size ; n2++){
                std::cout << this->layers[l]->dw[n][n2] << " ";
            }
            std::cout << std::endl;

        }
    }
    std::cout << "###################Output layer " << N_layers-1 << " ############\n";
    size_t l = N_layers - 1;
    for (size_t n = 0; n < this->layers[l]->N_size ; n++){
        std::cout << "b=" << this->layers[l]->b[n] << ", a=" << this->layers[l]->a[n] << std::endl;
        std::cout << "for backward: delta=" << this->layers[l]->delta[n] << std::endl;
        // std::cout << "this->layers[l+1]->N_size" << this->layers[l+1]->N_size << std::endl;
    }
}

double* NN::forward(size_t N_input,double** input){
    // this assumes that input is a correct size
    // TODO: check correct size
    // TODO: at this moment without biases
    // cur_layer->b[current_n];
    // N_input is number of input entries, each input has to have a lenght of layers[0]->N_size 
    double* results = new double[N_input]; // for now output layer is 1
    size_t id_last_layer = this->N_layers-1;

    for (size_t input_entry = 0; input_entry < N_input ; input_entry++){
        this->zero_grads(); // for now, this function also restart all parameters
        for (size_t n = 0 ; n < this->layers[0]->N_size ; n++){
            // std::cout <<input[input_entry][n] << " ";
            this->layers[0]->a[n] = input[input_entry][n];
        }
        // std::cout << std::endl;
        for (size_t l = 1 ; l < this->N_layers ; l++){
            auto cur_layer = this->layers[l];
            auto prev_layer = this->layers[l-1];
            for (size_t current_n = 0 ; current_n < cur_layer->N_size ; current_n++){
                cur_layer->a[current_n] = 0;
                for (size_t prev_n = 0 ; prev_n < prev_layer->N_size ; prev_n++){
                    cur_layer->a[current_n] += (sigmoid(prev_layer->a[prev_n]))*prev_layer->w[prev_n][current_n];
                }
            }
        }
        auto cur_layer = this->layers[id_last_layer];
        results[input_entry] = sigmoid(cur_layer->a[0]); // TODO: This has to change if the last layer is not of size 1;
    }
    return results;
}

void NN::forward_single(double* input_single){
    // this assumes that input is a correct size
    // TODO: check correct size
    // TODO: at this moment without biases
    // cur_layer->b[current_n];
    // N_input is number of input entries, each input has to have a lenght of layers[0]->N_size 
    for (size_t n = 0 ; n < this->layers[0]->N_size ; n++){
        this->layers[0]->a[n] = input_single[n];
    }
    for (size_t l = 1 ; l < this->N_layers ; l++){
        auto cur_layer = this->layers[l];
        auto prev_layer = this->layers[l-1];
        for (size_t current_n = 0 ; current_n < cur_layer->N_size ; current_n++){
            cur_layer->a[current_n] = 0;
            for (size_t prev_n = 0 ; prev_n < prev_layer->N_size ; prev_n++){
                cur_layer->a[current_n] += (sigmoid(prev_layer->a[prev_n]))*prev_layer->w[prev_n][current_n];
            }
        }
    }
}


void NN::backward(size_t N_data, double** input, double* output_ref){
    // last layer has to be always of the size 1 for now, that is why I have only 1 asterix
    // N_data is for both output and input
    for (size_t input_entry = 0; input_entry < N_data ; input_entry++){
        this->zero_grads();
        this->forward_single(input[input_entry]);
        double output_pred = sigmoid(this->layers[this->N_layers-1]->a[0]);
        double doutput_pred = dsigmoid(this->layers[this->N_layers-1]->a[0]);
        double diff = output_pred - output_ref[input_entry];
        double loss_last = pow(diff,2);
        double dloss = doutput_pred*diff; // MSE for 1 observable
        this->layers[N_layers - 1]->delta[0] = dloss;
        for (size_t dl = 0 ; dl < this->N_layers-1 ; dl++){
            // std::cout << "Hello seaman" << std::endl;
            auto cur_layer = this->layers[(N_layers-1) - dl-1];
            auto prev_layer = this->layers[(N_layers-1) - dl]; // previous from the back
            for (size_t current_n = 0 ; current_n < cur_layer->N_size ; current_n++){
                for (size_t prev_n = 0 ; prev_n < prev_layer->N_size ; prev_n++){
                    // std::cout << "Prev error: " << prev_layer->delta[prev_n] << " " << current_n << " " << prev_n << std::endl; //
                    // std::cout << cur_layer->delta[current_n] << std::endl;// += (dsigmoid(cur_layer->a[prev_n]))*prev_layer->w[prev_n][current_n]*prev_layer->delta[prev_n];
                    cur_layer->delta[current_n] = (dsigmoid(cur_layer->a[current_n]))*cur_layer->w[current_n][prev_n]*prev_layer->delta[prev_n];
                    cur_layer->dw[current_n][prev_n] += prev_layer->delta[prev_n]*sigmoid(cur_layer->a[current_n]);
                }
            }
        }
    }
}

void NN::update_weights(double lr){
    for (size_t l = 0 ; l < this->N_layers-1; l++){
        size_t current_layer_size = layers[l]->N_size;
        size_t next_layer_size = layers[l+1]->N_size;
        for (size_t n = 0; n < current_layer_size ; n++){
            for (size_t n2 = 0; n2 < next_layer_size ; n2++){
                // std::cout << "dw " << layers[l]->dw[n][n2] << " " << -lr*layers[l]->dw[n][n2] <<std::endl;
                layers[l]->w[n][n2] += -lr*layers[l]->dw[n][n2];
            }
        }
    }
}

void NN::print_weights(){
    std::cout << "#Printing weights\n";
    for (size_t l = 0 ; l < this->N_layers-1; l++){
        size_t current_layer_size = layers[l]->N_size;
        size_t next_layer_size = layers[l+1]->N_size;
        for (size_t n = 0; n < current_layer_size ; n++){
            for (size_t n2 = 0; n2 < next_layer_size ; n2++){
                // std::cout << "dw " << layers[l]->dw[n][n2] << " " << -lr*layers[l]->dw[n][n2] <<std::endl;
                std::cout << layers[l]->w[n][n2] << " (" << layers[l]->dw[n][n2] << ") ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "@Weights printing done \n";
}


double NN::get_output(){
    return sigmoid(this->layers[N_layers-1]->a[0]);
    // return this->layers[N_layers-1]->a[0];
}

void NN::zero_grads(){
    for (size_t l = 0 ; l < this->N_layers-1; l++){
        size_t current_layer_size = layers[l]->N_size;
        size_t next_layer_size = layers[l+1]->N_size;
        for (size_t n = 0; n < current_layer_size ; n++){
            layers[l]->delta[n] = 0;
            for (size_t n2 = 0; n2 < next_layer_size ; n2++){
                layers[l]->dw[n][n2] = 0;
            }
        }
    }
}

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

NN::NN(size_t n_layers, size_t* sizes) : num_of_layers(n_layers), layers_sizes(new size_t[n_layers]) {
    for (size_t layer = 0; layer < num_of_layers ; layer++){
        layers_sizes[layer] = sizes[layer]; 
    }
    weights = new double**[num_of_layers-1];
    for (size_t layer = 0; layer < num_of_layers -1 ; layer++){
        weights[layer] = weights_init(layers_sizes[layer],layers_sizes[layer+1]);
    }
}

double** NN::forward(double** input,size_t num_of_inputs){
    size_t size_input_layer = this->layers_sizes[0];
    size_t size_output_layer = this->layers_sizes[this->num_of_layers-1];
    double** nodes_outputs = new double*[num_of_layers];
    double** output = new double*[num_of_inputs];
    for (size_t n_output = 0; n_output < num_of_inputs ; n_output++){
        output[n_output] = new double[size_output_layer];
    }
    for (size_t layer = 0; layer < this->num_of_layers ; layer++){
        nodes_outputs[layer] = new double[this->layers_sizes[layer]];
    }
    for (size_t one_input = 0; one_input < num_of_inputs ; one_input++){
        for (size_t node = 0; node < size_input_layer ; node++){
            nodes_outputs[0][node] = input[one_input][node];
        }
        
        for (size_t layer = 1; layer < this->num_of_layers ; layer++){
            size_t size_layer_from = this->layers_sizes[layer-1];
            size_t size_layer_to = this->layers_sizes[layer];
            for (size_t node1 = 0; node1 < size_layer_to ; node1++){
                nodes_outputs[layer][node1] = 0;
                for (size_t node0 = 0; node0 < size_layer_from ; node0++){
                    nodes_outputs[layer][node1] += sigmoid(nodes_outputs[layer-1][node0])*this->weights[layer-1][node0][node1];
                }
            }
        }
        for (size_t output_node = 0; output_node < size_output_layer ; output_node++){
            output[one_input][output_node] = sigmoid(nodes_outputs[this->num_of_layers-1][output_node]);
        }
    }
    for (size_t layer = 0; layer < num_of_layers ; layer++){
        delete[] nodes_outputs[layer];
    }
    delete[] nodes_outputs;
    return output;
}

void NN::train(double* input, double* output,double lr,size_t epochs){
    for (size_t epoch = 0; epoch < epochs ; epoch++){
        double** nodes_outputs = new double*[num_of_layers];
        double** errors = new double*[num_of_layers];
        /*
          copy from forward for now, here are changes that nodes_outputs are sigmoid, this has to be changed also in forward
        */
        size_t size_input_layer = this->layers_sizes[0];
        size_t size_output_layer = this->layers_sizes[this->num_of_layers-1];
        double* pred_output = new double[size_output_layer];
        for (size_t layer = 0; layer < this->num_of_layers ; layer++){
            nodes_outputs[layer] = new double[this->layers_sizes[layer]];
            errors[layer] = new double[this->layers_sizes[layer]];
        }
        for (size_t node = 0; node < size_input_layer ; node++){
            nodes_outputs[0][node] = input[node];
        }
        for (size_t layer = 1; layer < this->num_of_layers ; layer++){
            size_t size_layer_from = this->layers_sizes[layer-1];
            size_t size_layer_to = this->layers_sizes[layer];
            for (size_t node1 = 0; node1 < size_layer_to ; node1++){
                nodes_outputs[layer][node1] = 0;
                for (size_t node0 = 0; node0 < size_layer_from ; node0++){
                    nodes_outputs[layer][node1] += sigmoid(nodes_outputs[layer-1][node0])*this->weights[layer-1][node0][node1];
                }
            }
        }
        // double diff = 0;
        for (size_t output_node = 0; output_node < size_output_layer ; output_node++){
            pred_output[output_node] = sigmoid(nodes_outputs[this->num_of_layers-1][output_node]);
            double diff =  pred_output[output_node]-output[output_node];
            // std::cout << "diff " << diff << std::endl;
            errors[this->num_of_layers-1][output_node] = diff*dsigmoid(nodes_outputs[this->num_of_layers-1][output_node]);
        }
        /*
          ######################################################################
        */
        for (size_t layer = this->num_of_layers-1; layer >= 1 ; layer--){
            // here from and to is exchange wrt to forward
            size_t size_layer_from = this->layers_sizes[layer];
            size_t size_layer_to = this->layers_sizes[layer-1];
            for (size_t node_to = 0; node_to < size_layer_to ; node_to++){
                errors[layer-1][node_to] = 0;
                for (size_t node_from = 0; node_from < size_layer_from ; node_from++){
                    errors[layer-1][node_to] += dsigmoid(nodes_outputs[layer-1][node_to])*weights[layer-1][node_to][node_from]*errors[layer][node_from];
                    // std::cout << layer-1 << " " << node1 << " " << errors[layer-1][node1] << " "  << dsigmoid(nodes_outputs[layer-1][node_from]) << " " << weights[layer-1][node1][node_from] << " " <<errors[layer][node0] <<std::endl; 
                }
            }
        }
        for (size_t layer = this->num_of_layers-1; layer >= 1 ; layer--){
            size_t size_layer_from = this->layers_sizes[layer];
            size_t size_layer_to = this->layers_sizes[layer-1];
            for (size_t node_to = 0; node_to < size_layer_to ; node_to++){
                for (size_t node_from = 0; node_from < size_layer_from ; node_from++){
                    weights[layer-1][node_to][node_from] -= lr*sigmoid(nodes_outputs[layer-1][node_to])*errors[layer][node_from];
                }
            }
        }
    }
}
// NN::~NN(){

// }

void NN::print_weights(){
    for (size_t layer = 1; layer < this->num_of_layers ; layer++){
        std::cout << "Weights from layer" << layer -1 << std::endl;
        size_t size_layer_from = this->layers_sizes[layer-1];
        size_t size_layer_to = this->layers_sizes[layer];
        for (size_t node1 = 0; node1 < size_layer_to ; node1++){
            for (size_t node0 = 0; node0 < size_layer_from ; node0++){
                std::cout <<"w"<<layer-1<< "_" <<node0 << node1 << "=" << weights[layer-1][node0][node1] << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void NN::print_NN_scheme(double* input){
    size_t size_input_layer = this->layers_sizes[0];
    for (size_t node = 0; node < size_input_layer ; node++){
        std::cout << "node_out_" << 0 << "_" <<node << "=" << input[node] << std::endl;
    }
    size_t size_output_layer = this->layers_sizes[this->num_of_layers-1];
    for (size_t layer = 1; layer < this->num_of_layers ; layer++){
        size_t size_layer_from = this->layers_sizes[layer-1];
        size_t size_layer_to = this->layers_sizes[layer];
        for (size_t node1 = 0; node1 < size_layer_to ; node1++){
            std::cout << "node_out_" << layer << "_" <<node1 << "=0" << std::endl;
            for (size_t node0 = 0; node0 < size_layer_from ; node0++){
                std::cout << "node_out_" << layer << "_" <<node1 << "+=sigmoid(node_out_" << layer-1 << "_"<<node0<<")*w"<<layer-1<<"_"<<node0<<node1<<std::endl;
            }
        }
    }
}

double** NN::weights_init(size_t layer_from, size_t layer_to){
    //layer_from = rows of the weight matrix
    //layer_to = columns
    // weights shape (layers,node_from,node_to)
    double** weights = new double*[layer_from];
    for (size_t i = 0; i < layer_from; ++i) {
        weights[i] = new double[layer_to];
        for (size_t j = 0; j < layer_to; ++j)
            weights[i][j] = random_number(-1.0 ,1.0);
    }
    return weights;
}

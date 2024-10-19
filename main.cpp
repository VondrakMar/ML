#include <iostream>
#include <cmath>

void print_vector(double* vector, size_t sample_size){
    for (size_t i = 0; i < sample_size ; i++){
        printf("%f ",vector[i]);
    }
    printf("\n");
}

double sigmoind(double z){
    return 1/(1+std::exp(-z));
}

double random_number(double r_down,double r_up){
    double r = (double)(rand())/RAND_MAX;
    r = r_down + r*(r_up-r_down);
    return r;
}

class Layer{
 public:
    Layer(size_t N_size);
    // ~Layer();
    void print_layer();
    size_t N_size;
    double* z; 
    double* b;
    double** w;
};

Layer::Layer(size_t n_size){
    N_size = n_size;
    z = new double[N_size];
    b = new double[N_size];
    w = new double*[N_size];
    for (size_t n = 0; n < N_size; n++){
        // z[n] = random_number(-2,2);
        z[n] = 0;// random_number(-2,2);
        b[n] = random_number(-2,2);
        w[n] = nullptr;
    }
}

void Layer::print_layer(){
    for (size_t l = 0; l < N_size; l++){
        std::cout << "b=" << b[l] << ", z=" << z[l] << std::endl;
    }
}



// NN::~NN() {
//     for (size_t i = 0; i < N_layers; i++) {
//         delete layers[i]; 
//     }
//     delete[] layers;
// }


class NN{
public:
    NN(size_t N_layers, size_t* layer_sizes);
    // ~NN();
    void print_NN();
    void forward(double* input);
    size_t N_layers;
    Layer** layers;
};

NN::NN(size_t n_layers, size_t* layer_sizes){
    N_layers = n_layers;
    layers = new Layer*[N_layers];
    for (size_t l = 0 ; l < N_layers-1; l++){
        layers[l] = new Layer(layer_sizes[l]);
        size_t current_layer_size = layers[l]->N_size;
        for (size_t n = 0; n < current_layer_size ; n++){
            layers[l]->w[n] = new double[layer_sizes[l+1]];
            for (size_t n2 = 0; n2 < layer_sizes[l+1] ; n2++){
                layers[l]->w[n][n2] = random_number(-2,2);
            }
        }
    }
    layers[N_layers-1] = new Layer(layer_sizes[N_layers-1]);
}

void NN::print_NN(){
    std::cout << "Printing NN with " << this->N_layers << " layers\n";
    for (size_t l = 0; l < this->N_layers-1; l++){
        std::cout << "###################Layer " << l << " ############\n";
        for (size_t n = 0; n < this->layers[l]->N_size ; n++){
            std::cout << "b=" << this->layers[l]->b[n] << ", z=" << this->layers[l]->z[n] << std::endl;
            // std::cout << "this->layers[l+1]->N_size" << this->layers[l+1]->N_size << std::endl;
            for (size_t n2 = 0; n2 < this->layers[l+1]->N_size ; n2++){
                std::cout << this->layers[l]->w[n][n2] << " ";
            }
            std::cout << std::endl;
        }
    }
}

void NN::forward(double* input){
    // this assumes that input is a correct size
    // TODO: check correct size
    for (size_t n = 0 ; n < this->layers[0]->N_size ; n++){
        this->layers[0]->z[n] = input[n];
    }
    for (size_t l = 1 ; l < this->N_layers ; l++){
        auto cur_layer = this->layers[l];
        auto prev_layer = this->layers[l-1];
        for (size_t current_n = 0 ; current_n < cur_layer->N_size ; current_n++){
            for (size_t prev_n = 0 ; prev_n < prev_layer->N_size ; prev_n++){
                cur_layer->z[current_n] += prev_layer->z[prev_n]*prev_layer->w[prev_n][current_n]+ cur_layer->b[current_n];;
                // cur_layer->z[current_n] += prev_layer->z[prev_n]*prev_layer->w[current_n][prev_n] + cur_layer->b[current_n];
            }
        }
    }
}

int main(){
    size_t NN_layer[5] = {2, 3, 1, 3, 1};
    auto my_NN = NN(5,NN_layer);
    double inputs[3] = {2.3,5.1};
    my_NN.forward(inputs);
    my_NN.print_NN();
    // auto temp = Layer(10);
    // temp.print_layer();
    return 0;
}

#include <iostream>
#include <cmath>

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
    void backward(double output_ref);
    void update_weights(double lr);
    void print_weights();
    void zero_grads();
    double get_output();
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
            layers[l]->dw[n] = new double[layer_sizes[l+1]];
            for (size_t n2 = 0; n2 < layer_sizes[l+1] ; n2++){
                layers[l]->w[n][n2] = random_number(-2,2);
                layers[l]->dw[n][n2] = 0;
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

void NN::forward(double* input){
    // this assumes that input is a correct size
    // TODO: check correct size
    // TODO: at this moment without biases
    // cur_layer->b[current_n];
    for (size_t n = 0 ; n < this->layers[0]->N_size ; n++){
        this->layers[0]->a[n] = input[n];
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


void NN::backward(double output_ref){
    // this assumes only 1 reference value, and last layer has to be always of the size 1
    double output_pred = sigmoid(this->layers[this->N_layers-1]->a[0]);
    double doutput_pred = dsigmoid(this->layers[this->N_layers-1]->a[0]);
    double diff = output_pred - output_ref;
    double loss_last = pow(diff,2);
    // double dloss = -doutput_pred*diff; // MSE for 1 observable
    double dloss = doutput_pred*diff; // MSE for 1 observable
    // std::cout << "dsig " << doutput_pred  << " diff " << diff << " loss " << loss_last << " dloss " << dloss << std::endl;
    this->layers[N_layers - 1]->delta[0] = dloss;
    for (size_t dl = 0 ; dl < this->N_layers-1 ; dl++){
        // std::cout << "Hello seaman" << std::endl;
        auto cur_layer = this->layers[(N_layers-1) - dl-1];
        auto prev_layer = this->layers[(N_layers-1) - dl]; // previous from the back
        for (size_t current_n = 0 ; current_n < cur_layer->N_size ; current_n++){
            for (size_t prev_n = 0 ; prev_n < prev_layer->N_size ; prev_n++){
                // std::cout << "Prev error: " << prev_layer->delta[prev_n] << " " << current_n << " " << prev_n << std::endl; //
                // std::cout << cur_layer->delta[current_n] << std::endl;// += (dsigmoid(cur_layer->a[prev_n]))*prev_layer->w[prev_n][current_n]*prev_layer->delta[prev_n];
                cur_layer->delta[current_n] += (dsigmoid(cur_layer->a[current_n]))*cur_layer->w[current_n][prev_n]*prev_layer->delta[prev_n];
                cur_layer->dw[current_n][prev_n] = prev_layer->delta[prev_n]*sigmoid(cur_layer->a[current_n]);
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


int main(){
    size_t N_l = 2;
    size_t NN_layer[N_l] = {1, 1};
    auto my_NN = NN(N_l,NN_layer);
    size_t First_layer = 1;
    double inputs[First_layer] = {2.3};//,5.1};
    my_NN.forward(inputs);
    my_NN.backward(0.8);
    my_NN.print_NN();
    // my_NN.print_NN();
    my_NN.update_weights(0.1);
    // std::cout << "output " << my_NN.get_output() << std::endl;
    my_NN.zero_grads();
    // std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    my_NN.forward(inputs);
    my_NN.backward(0.8);
    // my_NN.print_NN();
    my_NN.update_weights(0.1);
    // my_NN.print_NN();
    my_NN.zero_grads();
    // std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    // my_NN.print_NN();
    for (int i =0 ; i < 5000 ; i++){
        my_NN.forward(inputs);
        my_NN.backward(0.8);
        // my_NN.print_NN();
        // std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
        // my_NN.print_weights();
        my_NN.update_weights(0.1);
        // my_NN.print_weights();
        std::cout << "output " << my_NN.get_output() << " " << my_NN.layers[0]->w[0][0] << std::endl;
        // std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
        my_NN.zero_grads();
        // double res = my_NN.get_output();
        // std::cout << res << std::endl;
        // my_NN.print_NN();
    }
    my_NN.print_NN();
    // // auto temp = Layer(10);
    // temp.print_layer();
    // std::cout << "sig " << dsigmoid(2.3);
    return 0;
}

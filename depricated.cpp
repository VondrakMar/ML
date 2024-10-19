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

class Edge{
public:
    Edge();
    double w;
};

class Neuron{
public:
    // Neuron(size_t N_neurons);
    Neuron();
    ~Neuron();
    size_t N_next_neurons;
    Neuron** next_layer;
    Edge** next_edges;
    double b;
    double z; 
    void forward_propagation();
};

// Neuron::Neuron(size_t N_neurons){
Neuron::Neuron(){
    z = 0;
    b = random_number(-2,2);
    // N_next_neurons = N_neurons;
    N_next_neurons = 0;
    next_layer = nullptr;//new Neuron*[N_next_neurons];
    next_edges = nullptr;//new Edge*[N_next_neurons];
}

Neuron::~Neuron() {
    if (next_layer) {
        delete[] next_layer; 
    }
    if (next_edges) {
        delete[] next_edges; 
    }
}


Edge::Edge(){
    w = random_number(-2,2);
}

class Layer{
 public:
    Layer(size_t N_size);
    ~Layer();
    Layer* next_layer;
    size_t N_neurons;
    Neuron** neurons;
};

Layer::Layer(size_t N_size){
    N_neurons = N_size;
    neurons = new Neuron*[N_size];
    for (size_t neuron = 0; neuron < N_neurons; neuron++){
        neurons[neuron] = new Neuron();
    }
    next_layer = nullptr;
}

Layer::~Layer() {
    for (size_t i = 0; i < N_neurons; i++) {
        delete neurons[i]; 
    }
    
    delete[] neurons; 
}


class NN{
public:
    NN(size_t input_size, size_t output_size);
    ~NN();
    void add_layer(Layer* added_layer);
    size_t N_layers;
    Layer** layers;
};

NN::NN(size_t input_size, size_t output_size){
    N_layers = 2;
    layers = new Layer*[N_layers];
    layers[0] = new Layer(input_size);
    layers[1] = new Layer(output_size);
}

NN::~NN() {
    for (size_t i = 0; i < N_layers; i++) {
        delete layers[i]; 
    }
    delete[] layers;
}

void NN::add_layer(Layer* added_layer){
    // this function add a layer, but will keep input and output layers unchange.
    this->N_layers += 1;
    Layer** new_layers = new Layer*[this->N_layers];
    // new_layers[0] =
    for (size_t layer = 0 ; layer < N_layers - 2 ; layer++){
        new_layers[layer] = this->layers[layer]; 
    }
    new_layers[this->N_layers-2] = added_layer;
    new_layers[this->N_layers-1] = this->layers[this->N_layers-2];
    delete[] this->layers; // This is not freeing pointers 1 level below, which is bad
    this->layers = new_layers;
}


// void connect_layers(Layer layer1, Layer layer2){
    
// }

class NeuralNetwork{
public:
    NeuralNetwork(size_t n_layers,size_t* n_neurons);
    size_t N_layers;
    size_t* N_neurons;
    double** weights;
    double** z;
    
};

NeuralNetwork::NeuralNetwork(size_t n_layers,size_t* n_neurons){
    weights = new double*[n_layers-1];
    // weights[0]
    for (size_t l = 1; l < n_layers ; l++){        
        size_t layer_neurons = n_neurons[l];
        weights[l] = new double[n_neurons[l-1]*n_neurons[l]];
        
    }
}

              

int main(){
    std::cout << "Hello world\n";
    NN tempNN = NN(10,4);
    std::cout << "###########0 layer######################\n";
    for (int i = 0; i < 10 ; i++){
        std::cout << tempNN.layers[0]->neurons[i]->b << std::endl;
    }
    std::cout << "###########1 layer######################\n";
    for (int i = 0; i < 4 ; i++){
        std::cout << tempNN.layers[1]->neurons[i]->b << std::endl;
    }
    Layer* temp_layer = new Layer(7);
    tempNN.add_layer(temp_layer);
    std::cout << "###########new 1 layer######################\n";
    for (int i = 0; i < 7 ; i++){
        std::cout << tempNN.layers[1]->neurons[i]->b << std::endl;
    }
    std::cout << "###########moved output layer######################\n";
    for (int i = 0; i < 4 ; i++){
        std::cout << tempNN.layers[2]->neurons[i]->b << std::endl;
    }
    
    // Neuron n1 = Neuron(3); // this is 1 neuron to 3
    // Neuron n2 = Neuron(1);
    // Edge e1 = Edge();
    // n1.next_layer[0] = &n2;
    // std::cout << n1.next_layer[0]->b << std::endl;
    // std::cout << n2.b << std::endl;
    // n1.next_edges[0] = &e1; 
    // n1.next_layer[0]->z = n1.z*n1.next_edges[0]->w + n1.next_layer[0]->b;
    return 0;
}

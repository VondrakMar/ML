#include <iostream>
#include <cmath>
#include "NN.hpp"
#include "util.hpp"

int main(){
    size_t n_lay = 4;
    size_t sizes[n_lay] = {3,100,100,2};
    auto my_NN = NN(n_lay,sizes);
    double input[3] = {2.3,4.1,12.45};
    double output[2] = {0.5,0.3};
    // my_NN.print_weights();
    // my_NN.print_NN_scheme(input);
    auto res1 =my_NN.forward(input);
    my_NN.train(input, output,0.05,10);
    auto res2 =my_NN.forward(input);
    print_vector(res1,2);
    print_vector(res2,2);
}

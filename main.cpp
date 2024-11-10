#include <iostream>
#include <cmath>
#include "NN.hpp"
#include "util.hpp"


double* function_to_fit(double* input){
    double* output = new double[2];
    auto x1 = input[0];
    auto x2 = input[1];
    auto x3 = input[2];
    output[0] = std::sin(x1) + std::cos(x2) - x3 * 0.5;
    output[1] = x1 * x2 * std::exp(-x3) + std::tan(x1 + x2);
    return output;
}

void print_matrix(double** matrix, size_t n_row,size_t n_col){
    for (size_t row=0 ; row < n_row ; row++){
        for (size_t col=0 ; col < n_col ; col++){
            std::cout << matrix[row][col] << " " ;
        }
        std::cout << std::endl;
    }
}

double find_min_matrix(double** matrix, size_t n_row,size_t n_col){
    // n_row = num_of_entries
    // n_col = entry_size
    double min_value = std::numeric_limits<double>::max();
    for (size_t row=0 ; row < n_row ; row++){
        for (size_t col=0 ; col < n_col ; col++){
            if (min_value > matrix[row][col]){
                min_value = matrix[row][col];
            }
        }
    }
    return min_value;
}


double find_max_matrix(double** matrix, size_t n_row,size_t n_col){
    double max_value = std::numeric_limits<double>::lowest();
    for (size_t row=0 ; row < n_row ; row++){
        for (size_t col=0 ; col < n_col ; col++){
            if (max_value < matrix[row][col]){
                max_value = matrix[row][col];
            }
        }
    }
    return max_value;
}


double** normalized_data_matrix(double** data,size_t num_of_entries,size_t entry_size){
    double** norm_data = new double*[num_of_entries];
    for (size_t entry = 0 ; entry < num_of_entries ; entry++){
        norm_data[entry] = new double[entry_size];
    }
    double min_value = find_min_matrix(data,num_of_entries,entry_size);
    double max_value = find_max_matrix(data,num_of_entries,entry_size);
    double max_min = max_value - min_value;
    std::cout << "max value: " << max_value << " min value: " << min_value << std::endl;
    for (size_t row=0 ; row < num_of_entries ; row++){
        for (size_t col=0 ; col < entry_size ; col++){
            norm_data[row][col] = (data[row][col]-min_value)/(max_min);
        }
    }
    return norm_data;
}


int main(){
    size_t n_lay = 4;
    size_t sizes[n_lay] = {3,64,64,2};
    auto my_NN = NN(n_lay,sizes);
    double** input = new double*[4];
    input[0] = new double[3]{2.3,4.1,-12.45};
    input[1] = new double[3]{1.2,5.2,1.2};
    input[2] = new double[3]{4.1,-2.644,12.1};
    input[3] = new double[3]{21.3,43.1,0.1};
    // print_matrix(input,2,3);
    // double* output = function_to_fit(input);
    double** norm_input=normalized_data_matrix(input,4,3);
    std::cout << "Original data:" << std::endl;
    print_matrix(input,4,3);
    std::cout << "Normed data:" << std::endl;
    print_matrix(norm_input,4,3);
    // double* norm_output=normalized_data(2,output);
    // my_NN.print_weights();
    // my_NN.print_NN_scheme(input);
    // auto res1 =my_NN.forward(norm_input);
    // my_NN.train(norm_input, norm_output,0.005,10000);
    auto res2 =my_NN.forward(input,4);
    std::cout << "Normed output:" << std::endl;
    print_matrix(res2,4,2);
    // print_vector(norm_output,2);
    // print_vector(res2,2);
    // delete[] output;
    // delete[] norm_input;
    // delete[] norm_output;
    // delete[] res2;
}

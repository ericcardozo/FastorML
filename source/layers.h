#ifndef LAYERS_H
#define LAYERS_H

#include "tensor_algebra.h"
#include "activation_functions.h"

template<size_type input_features, size_type output_features>
struct Parameters{
  Tensor<float, input_features, output_features> weight;
  Tensor<float, output_features> bias;
  
  void initialize(std::string initializer){
    std::random_device rd;
    std::mt19937 generator(rd());

    if(initializer == "he"){
      std::normal_distribution<float> distribution(0, std::sqrt(2.0 / input_features));
    }
    else if(initializer == "xavier"){
      std::normal_distribution<float> distribution(0, std::sqrt(2.0 / input_features + output_features));
    }
    else{
      std::cout << "Initializer not implemented" << std::endl;
    }

    bias = 0.0;
    for(auto i = 0; i < input_features; ++i){
      for(auto j = 0; j < output_features; ++j){
        weight(i, j) = distribution(generator);
      }
    }
  }
};

//Linear layer class

template<size_type input_features, size_type output_features>
class Linear{
  public:

    Linear(){
      parameters.initialize(std::string initializer = "he");
    }

    //forward method
    template<size_type batch_size>
    Tensor<float, batch_size, output_features> forward(const Tensor<float, batch_size, input_features> &input){
      Tensor<float,batch_size> ones(1);
      return matmul(input, parameters.weight) + outer(ones, parameters.bias); 
    }
  
    //backward method
    template<size_type batch_size>
    Tensor<float, batch_size, input_features> backward(
      const Tensor<float, batch_size, output_features> &gradient,
      const Tensor<float, batch_size, input_features>& input
    ){
      Tensor<float, input_features, output_features> weight_gradient = matmul(transpose(input), gradient);
      Tensor<float, output_features> bias_gradient(0.0); // Initialize bias_gradient with zeros

      for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < output_features; j++){
        bias_gradient(j) += gradient(i, j);
        }
      }

      Tensor<float, batch_size, input_features> input_gradient = matmul(gradient, parameters.weight);
      return input_gradient;
    }
  
  private:
    Parameters<input_features, output_features> parameters;
};

//ReLU layer class

template<size_type input_features, size_type output_features>
class ReLU{
  public:
    //forward method
    template<size_type batch_size>
    Tensor<float, batch_size, output_features> forward(const Tensor<float, batch_size, input_features> &input){
      return relu(input);
    }
  
    //backward method
    template<size_type batch_size>
    Tensor<float, batch_size, input_features> backward(
      const Tensor<float, batch_size, output_features> &gradient,
      const Tensor<float, batch_size, input_features>& input
    ){
      return gradient * relu_gradient(input);
    }
};

template<size_type input_features, size_type output_features>
class LogSoftMax{
  public:
    //forward method
    template<size_type batch_size>
    Tensor<float, batch_size, output_features> forward(const Tensor<float, batch_size, input_features> &input){
      return log_softmax(input);
    }
  
    //backward method
    template<size_type batch_size>
    Tensor<float, batch_size, input_features> backward(
      const Tensor<float, batch_size, output_features> &gradient,
      const Tensor<float, batch_size, input_features>& input
    ){
      return logsoftmax_gradient(input) * gradient;
    }
};


#endif
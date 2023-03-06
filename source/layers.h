#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <random>


#include "tensor_algebra.h"
#include "activation_functions.h"

template<size_type input_features, size_type output_features>
struct Parameters{
  Tensor<float, input_features, output_features> weight;
  Tensor<float, input_features, output_features> weight_gradient;
  Tensor<float, output_features> bias;
  Tensor<float, output_features> bias_gradient;

  void initialize(const std::string& initializer){
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<float> distribution;
    if(initializer == "he"){
      distribution = std::normal_distribution<float>(0, std::sqrt(2.0 / input_features));
    }
    else if(initializer == "xavier"){
      distribution = std::normal_distribution<float>(0, std::sqrt(2.0 / input_features + output_features));
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

    Linear(const std::string& initializer = "he"){
      parameters.initialize(initializer);
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
      parameters.weight_gradient = matmul(transpose(input), gradient);
      parameters.bias_gradient = 0.0; // Initialize bias_gradient with zeros
      
      for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < output_features; j++){
        parameters.bias_gradient(j) += gradient(i, j);
        }
      }

      Tensor<float, batch_size, input_features> input_gradient = matmul(gradient, transpose(parameters.weight));
      return input_gradient;
    }
  
  private:
    Parameters<input_features, output_features> parameters;
};

//ReLU layer 
template<size_type batch_size, size_type output_features>
class ReLU{
  public:
    //forward method
    Tensor<float, batch_size, output_features> forward(
      const Tensor<float, batch_size, output_features> &input
    ){
      input_ = input;
      output_ = relu(input);
      return output_;
    }

    
    //backward method
    template<size_type input_features>
    Tensor<float, batch_size, input_features> backward(
      const Tensor<float, batch_size, output_features> &gradient,
      const Tensor<float, batch_size, input_features>& input
    ){
      return gradient * relu_gradient(input);
    }

    Tensor<float, batch_size, output_features> input() const {return input_;};
    Tensor<float, batch_size, output_features> output() const {return output_;};

  private:
    Tensor<float, batch_size, output_features> input_;
    Tensor<float, batch_size, output_features> output_;    
};


template<size_type batch_size, size_type output_features>
class LogSoftMax{
  public:
    //forward method
    Tensor<float, batch_size, output_features> forward(
      const Tensor<float, batch_size, output_features> &input
    ){
      input_ = input;
      output_ = log_softmax(input); 
      return output_;
    }
  
    //backward method
    template<size_type input_features>
    Tensor<float, batch_size, input_features> backward(
      const Tensor<float, batch_size, output_features> &gradient,
      const Tensor<float, batch_size, input_features>& input
    ){
      return log_softmax_gradient(input) * gradient;
    }
  
    Tensor<float, batch_size, output_features> input() const {return input_;};
    Tensor<float, batch_size, output_features> output() const {return output_;};

  private:
    Tensor<float, batch_size, output_features> input_;
    Tensor<float, batch_size, output_features> output_;    
};

#endif
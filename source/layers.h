#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <random>
#include <Fastor/Fastor.h>
#include "activation_functions.h"
#include "optimizers.h"
#include "parameters.h"

//Linear layer class

template<std::size_t input_features, std::size_t output_features>
class Linear{
  public:

    Linear(const std::string& initializer = "he")
      : parameters(initializer){}

    void set_optimizer(std::unique_ptr<Optimizer<input_features, output_features>> optimizer){
      parameters.optimizer = std::move(optimizer);
    }
    
    //forward method
    template<std::size_t batch_size>
    Fastor::Tensor<float, batch_size, output_features> forward(const Fastor::Tensor<float, batch_size, input_features> &input){
      Fastor::Tensor<float,batch_size> ones(1);
      return matmul(input, parameters.weight) + outer(ones, parameters.bias); 
    }
  
    //backward method
    template<std::size_t batch_size>
    Fastor::Tensor<float, batch_size, input_features> backward(
      const Fastor::Tensor<float, batch_size, output_features> &gradient,
      const Fastor::Tensor<float, batch_size, input_features>& input
    ){
      parameters.weight_gradient = matmul(transpose(input), gradient);
      parameters.bias_gradient = 0.0; // Initialize bias_gradient with zeros
      
      for (auto i = 0; i < batch_size; i++) {
        for (auto j = 0; j < output_features; j++){
          parameters.bias_gradient(j) += gradient(i, j);
        }
      }
      return matmul(gradient, transpose(parameters.weight));
    }

    void update(){
      parameters.update();
    }
  
  private:

    Parameters<input_features, output_features> parameters;
};


//ReLU layer 
template<std::size_t batch_size, std::size_t output_features>
class ReLU{
  public:
    //forward method
    Fastor::Tensor<float, batch_size, output_features> forward(
      const Fastor::Tensor<float, batch_size, output_features>& input
    ){
      input_ = input;
      output_ = relu(input);
      return output_;
    }

    
    //backward method
    template<std::size_t input_features>
    Fastor::Tensor<float, batch_size, input_features> backward(
      const Fastor::Tensor<float, batch_size, output_features> &gradient,
      const Fastor::Tensor<float, batch_size, input_features>& input
    ){
      return gradient * relu_gradient(input);
    }

    Fastor::Tensor<float, batch_size, output_features> input() const {return input_;};
    Fastor::Tensor<float, batch_size, output_features> output() const {return output_;};

  private:

    Fastor::Tensor<float, batch_size, output_features> input_;
    Fastor::Tensor<float, batch_size, output_features> output_;    
};


template<std::size_t batch_size, std::size_t output_features>
class LogSoftMax{
  public:
    //forward method
    Fastor::Tensor<float, batch_size, output_features> forward(
      const Fastor::Tensor<float, batch_size, output_features> &input
    ){
      input_ = input;
      output_ = log_softmax(input); 
      return output_;
    }
  
    //backward method
    template<std::size_t input_features>
    Fastor::Tensor<float, batch_size, input_features> backward(
      const Fastor::Tensor<float, batch_size, output_features> &gradient,
      const Fastor::Tensor<float, batch_size, input_features>& input
    ){
      return log_softmax_gradient(input) * gradient;
    }
  
    Fastor::Tensor<float, batch_size, output_features> input() const {return input_;};
    Fastor::Tensor<float, batch_size, output_features> output() const {return output_;};

  private:
    Fastor::Tensor<float, batch_size, output_features> input_;
    Fastor::Tensor<float, batch_size, output_features> output_;    
};

#endif
#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <iostream>
#include <random>
#include <memory>
#include <Fastor/Fastor.h>
#include "optimizers.h"

template<std::size_t input_features, std::size_t output_features>
class Optimizer;

template<std::size_t input_features, std::size_t output_features>
struct Parameters{

    Fastor::Tensor<float, input_features, output_features> weight;
    Fastor::Tensor<float, input_features, output_features> weight_gradient;
    Fastor::Tensor<float, output_features> bias;
    Fastor::Tensor<float, output_features> bias_gradient;
    
    std::unique_ptr<Optimizer<input_features, output_features>> optimizer;

    Parameters(float learning_rate, const std::string& initializer = "he"){
      std::random_device rd;
      std::mt19937 generator(rd());
      std::normal_distribution<float> distribution;

      switch(initializer){
        case "he":
          distribution = std::normal_distribution<float>(0, std::sqrt(2.0 / input_features));
        break;

        case "xavier":
          distribution = std::normal_distribution<float>(0, std::sqrt(2.0 / input_features + output_features));  
        break;

        default:
          std::cout << "Invalid initializer" << std::endl;
        break;
      }

      bias = 0.0;
      for(auto i = 0; i < input_features; ++i){
        for(auto j = 0; j < output_features; ++j){
          weight(i, j) = distribution(generator);
        }
      }
    }

    void update(){
      optimizer->update(*this);
    }

};

#endif
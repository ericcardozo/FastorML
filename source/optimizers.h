#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <iostream>
#include <random>
#include <memory>
#include <Fastor/Fastor.h>
#include "parameters.h"

template<std::size_t input_features, std::size_t output_features>
class Optimizer{
  public:
    virtual void update(Parameters<input_features, output_features>& parameters) = 0;
    virtual ~Optimizer() = default;
};

template<std::size_t input_features, std::size_t output_features>
class SGD : public Optimizer<input_features, output_features>{
  public:
    SGD(float learning_rate) : learning_rate_(learning_rate) {}
    void update(Parameters<input_features, output_features>& parameters) override {
      parameters.weight -= learning_rate_ * parameters.weight_gradient;
      parameters.bias -= learning_rate_ * parameters.bias_gradient;
    }

private:
  float learning_rate_;
};

#endif
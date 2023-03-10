#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "activation_functions.h"

template<std::size_t batch_size, std::size_t output_features>
Fastor::Tensor<float, batch_size, output_features> one_hot_encoding(const std::vector<int> targets){
  Fastor::Tensor<float, batch_size, output_features> one_hot_targets(0);
  for(auto i = 0; i < batch_size; i++){
    one_hot_targets(i,targets[i]) = 1;
  }
  return one_hot_targets;
}

//nll_loss function

template<std::size_t batch_size, std::size_t output_features>
float nll_loss(
  const Fastor::Tensor<float, batch_size, output_features>& input,
  const Fastor::Tensor<float, batch_size, output_features>& one_hot_targets
){
  float output = -inner(input,one_hot_targets);
  return output / batch_size;
}

template<std::size_t batch_size, std::size_t output_features>
Fastor::Tensor<float, batch_size, output_features > nll_loss_gradient(
  const Fastor::Tensor<float, batch_size, output_features>& input,
  const Fastor::Tensor<float, batch_size, output_features>& one_hot_targets
){
  return (exp(logsoftmax(input)) - one_hot_targets) / batch_size;
}

template<std::size_t batch_size, std::size_t output_features>
class NLLLoss{
  public:
    NLLLoss(){}
    float forward(
      const Fastor::Tensor<float, batch_size, output_features>& input,
      const Fastor::Tensor<float, batch_size, output_features>& one_hot_targets
    ){
      return nll_loss(input, one_hot_targets);
    }
    Fastor::Tensor<float, batch_size, output_features> backward(
      const Fastor::Tensor<float, batch_size, output_features>& input,
      const Fastor::Tensor<float, batch_size, output_features>& one_hot_targets
    ){
      return nll_loss_gradient(input, one_hot_targets);
    }
  private:
};

#endif
#include<iostream>
#include<vector>
#include<random>
#include<memory>
#include <Fastor/Fastor.h>



template<std::size_t batch_size, std::size_t output_features>
Fastor::Tensor<float, batch_size, output_features> relu(
  const Fastor::Tensor<float, batch_size, output_features> &input){
  return max(input, 0);
}

template<std::size_t batch_size, std::size_t output_features>
Fastor::Tensor<float, batch_size, output_features> relu_gradient(
  const Fastor::Tensor<float, batch_size, output_features> &input){
  return input > 0;
}


//LogSoftMax

template<std::size_t batch_size, std::size_t output_features>
Fastor::Tensor<float, batch_size, output_features> logsoftmax(
  const Fastor::Tensor<float, batch_size, output_features>& input
){
  Fastor::Tensor<float, batch_size, output_features> output;
  for(auto i = 0; i < batch_size; i++){
    auto max_substraction = input(i,all) - max(input(i,all));
    float log_sum_exp = log(sum(exp(max_substraction)));
    output(i,all) = input(i,all) - log_sum_exp;
  }
  return output;
}

//LogSoftMax gradient.

template<std::size_t batch_size, std::size_t output_features>
Fastor::Tensor<float, batch_size, output_features> logsoftmax_gradient(
  const Fastor::Tensor<float, batch_size, output_features>& input,
  const Fastor::Tensor<float, batch_size, output_features>& one_hot_targets
){
  return exp(input) - one_hot_targets;  
}



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
  return output;
}


int main(){
  return 0;
}

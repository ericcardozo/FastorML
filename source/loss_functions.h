#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

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

#endif
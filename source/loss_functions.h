#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

template<size_type output_features, size_type batch_size>
Tensor<float, output_features, batch_size> one_hot_encoding(const std::vector<int> labels){
  Tensor<float, output_features, batch_size> one_hot_labels(0);
  for(auto i = 0; i < batch_size; i++){
    one_hot_labels(labels[i], i) = 1;
  }
  return one_hot_labels;
}

//nll_loss function and its gradient

template<size_type batch_size, size_type output_features>
float nll_loss(
  const Tensor<float, batch_size, output_features>& input,
  const Tensor<float, output_features, batch_size>& one_hot_labels
){
  float loss = -trace(matmul(one_hot_labels, input));
  return loss;
}

template<size_type batch_size, size_type output_features>
Tensor<float, batch_size, output_features> nll_loss_gradient(
  const Tensor<float, batch_size, output_features>& input,
  const Tensor<float, output_features, batch_size>& one_hot_labels
){
  return -transpose(one_hot_labels) + transpose(softmax(input));
}

#endif
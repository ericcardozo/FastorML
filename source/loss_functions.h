#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

template<size_type output_features, size_type batch_size>
Tensor<float, output_features, batch_size> one_hot_encoding(const std::vector<int> targets){
  Tensor<float, output_features, batch_size> one_hot_targets(0);
  for(auto i = 0; i < batch_size; i++){
    one_hot_labels(targets[i], i) = 1;
  }
  return one_hot_labels;
}

//nll_loss function and its gradient

template<size_type batch_size, size_type output_features>
float nll_loss(
  const Tensor<float, batch_size, output_features>& input,
  const Tensor<float, output_features, batch_size>& one_hot_targets
){
  float loss = -trace(matmul(one_hot_targets, input));
  return loss;
}

template<size_type batch_size, size_type output_features>
Tensor<float, batch_size, output_features> nll_loss_gradient(
  const Tensor<float, batch_size, output_features>& input,
  const Tensor<float, output_features, batch_size>& one_hot_targets
){
  return -transpose(one_hot_targets) + transpose(softmax(input));
}

//A class implementing the NLLLoss forward and backward method.

class NLLLoss{
  public:
    //forward method
    template<size_type batch_size, size_type output_features>
    float forward(
      const Tensor<float, batch_size, output_features> &input,
      const Tensor<float, output_features, batch_size>& one_hot_targets
    ){
      return nll_loss(input, one_hot_targets);
    }
  
    //backward method
    template<size_type batch_size, size_type input_features ,size_type output_features>
    Tensor<float, batch_size, input_features> backward(
      const Tensor<float, batch_size, output_features> &gradient,
      const Tensor<float, batch_size, input_features>& input,
      const Tensor<float, output_features, batch_size>& one_hot_targets
    ){
      return nll_loss_gradient(input, one_hot_targets) * gradient;
    }
};

#endif
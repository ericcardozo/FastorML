#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

template<std::size_t output_features, std::size_t batch_size>
Fastor::Tensor<float, output_features, batch_size> one_hot_encoding(const std::vector<int> targets){
  Fastor::Tensor<float, output_features, batch_size> one_hot_targets(0);
  for(auto i = 0; i < batch_size; i++){
    one_hot_targets(targets[i], i) = 1;
  }
  return one_hot_targets;
}

//nll_loss function and its gradient

template<std::size_t batch_size, std::size_t output_features>
float nll_loss(
  const Fastor::Tensor<float, batch_size, output_features>& input,
  const Fastor::Tensor<float, output_features, batch_size>& one_hot_targets
){
  float loss = -trace(matmul(one_hot_targets, input));
  return loss;
}

template<std::size_t batch_size, std::size_t output_features>
Fastor::Tensor<float, batch_size, output_features> nll_loss_gradient(
  const Fastor::Tensor<float, batch_size, output_features>& input,
  const Fastor::Tensor<float, output_features, batch_size>& one_hot_targets
){
  return -transpose(one_hot_targets) + transpose(softmax(input));
}

//A class implementing the NLLLoss forward and backward method.

class NLLLoss{
  public:

    NLLLoss() = default;
    //forward method
    template<std::size_t batch_size, std::size_t output_features>
    float forward(
      const Fastor::Tensor<float, batch_size, output_features> &input,
      const Fastor::Tensor<float, output_features, batch_size>& one_hot_targets
    ){
      return nll_loss(input, one_hot_targets);
    }
  
    //backward method
    template<std::size_t batch_size, std::size_t input_features ,std::size_t output_features>
    Fastor::Tensor<float, batch_size, input_features> backward(
      const Fastor::Tensor<float, batch_size, output_features> &gradient,
      const Fastor::Tensor<float, batch_size, input_features>& input,
      const Fastor::Tensor<float, output_features, batch_size>& one_hot_targets
    ){
      return nll_loss_gradient(input, one_hot_targets) * gradient;
    }
};

#endif
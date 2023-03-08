#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

//ReLU activation function and its gradient

template<std::size_t input_features, std::size_t output_features>
Fastor::Tensor<float, input_features, output_features> relu(const Fastor::Tensor<float, input_features, output_features> &input){
  return max(input, 0);
}

template<std::size_t input_features, std::size_t output_features>
Fastor::Tensor<float, input_features, output_features> relu_gradient(const Fastor::Tensor<float, input_features, output_features> &input){
  return input > 0;
}

//LogSoftmax activation function and its gradient


template<std::size_t batch_size, std::size_t output_features>
Fastor::Tensor<float, batch_size, output_features> log_softmax(const Fastor::Tensor<float, batch_size, output_features> &input){
  Fastor::Tensor<float, batch_size, output_features> output;
  for(auto i = 0; i < batch_size; i++){
    auto max_value = max(input(i,all));
    auto shifted_values = input(i,all) - max_value;
    auto log_sum_exp = log(sum(exp(shifted_values)));
  
    output(i,all) = shifted_values - log_sum_exp;
  }
  return output;
}

template<std::size_t batch_size, std::size_t output_features>
Fastor::Tensor<float, batch_size, output_features> logsoftmax_gradient(const Fastor::Tensor<float, batch_size, output_features> &input, const Fastor::Tensor<float, batch_size, output_features> &grad_output){
  Fastor::Tensor<float, batch_size, output_features> output;
  for(auto i = 0; i < batch_size; i++){
    auto max_value = max(input(i,all));
    auto shifted_values = input(i,all) - max_value;
    auto exp_shifted_values = exp(shifted_values);
    auto sum_exp_shifted_values = sum(exp_shifted_values);
    auto softmax_output = exp_shifted_values / sum_exp_shifted_values;
  
    output(i,all) = grad_output(i,all) - matmul(grad_output(i,all), softmax_output);
  }
  return output;
}

// Softmax activation function and its gradient

template<std::size_t batch_size, std::size_t output_features>
Fastor::Tensor<float, batch_size, output_features> softmax(const Fastor::Tensor<float, batch_size, output_features> &input){
  return exp(log_softmax(input));
}

template<std::size_t batch_size, std::size_t output_features>
Fastor::Tensor<float, batch_size, output_features> softmax_gradient(const Fastor::Tensor<float, batch_size, output_features> &input, const Fastor::Tensor<float, batch_size, output_features> &grad_output){
  Fastor::Tensor<float, batch_size, output_features> output;
  for(auto i = 0; i < batch_size; i++){
    auto max_value = max(input(i,all));
    auto shifted_values = input(i,all) - max_value;
    auto exp_shifted_values = exp(shifted_values);
    auto sum_exp_shifted_values = sum(exp_shifted_values);
    auto softmax_output = exp_shifted_values / sum_exp_shifted_values;

    auto jacobian = diag(softmax_output) - outer(softmax_output, softmax_output);
    output(i,all) = matmul(grad_output(i,all), jacobian);
  }
  return output;
}



#endif
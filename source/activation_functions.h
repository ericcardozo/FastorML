#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

//ReLU activation function and its gradient

template<size_type input_features, size_type output_features>
Tensor<float, input_features, output_features> relu(const Tensor<float, input_features, output_features> &input){
  return max(input, 0);
}

template<size_type input_features, size_type output_features>
Tensor<float, input_features, output_features> relu_gradient(const Tensor<float, input_features, output_features> &input){
  return input > 0;
}

//LogSoftmax activation function and its gradient


template<size_type batch_size, size_type output_features>
Tensor<float, batch_size, output_features> log_softmax(const Tensor<float, batch_size, output_features> &input){
  Tensor<float, batch_size, output_features> output;
  for(auto i = 0; i < batch_size; i++){
    auto max_value = max(input(i,all));
    auto shifted_values = input(i,all) - max_value;
    auto log_sum_exp = log(sum(exp(shifted_values)));
  
    output(i,all) = shifted_values - log_sum_exp;
  }
  return output;
}

template<size_type batch_size, size_type output_features>
Tensor<float, batch_size, output_features> logsoftmax_gradient(const Tensor<float, batch_size, output_features> &input, const Tensor<float, batch_size, output_features> &grad_output){
  Tensor<float, batch_size, output_features> output;
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

template<size_type batch_size, size_type output_features>
Tensor<float, batch_size, output_features> softmax(const Tensor<float, batch_size, output_features> &input){
  return exp(log_softmax(input));
}

template<size_type batch_size, size_type output_features>
Tensor<float, batch_size, output_features> softmax_gradient(const Tensor<float, batch_size, output_features> &input, const Tensor<float, batch_size, output_features> &grad_output){
  Tensor<float, batch_size, output_features> output;
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
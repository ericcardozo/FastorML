#include<iostream>
using size_type = std::size_t;
#include<random>
#include<vector>

#include "source/layers.h"
#include "source/loss_functions.h"

int main(){
  std::vector<int> labels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  Tensor<float, 32, 784> input(0);
  input.random();

  Linear<784,256> linear1("he");
  ReLU relu1;
  
  Linear<256,10> linear2("xavier");
  LogSoftMax logsoftmax;

  Tensor<float,32,256> output = linear1.forward(input);
  output = relu1.forward(output);
  Tensor<float,32,10> output2 = linear2.forward(output);
  output2 = logsoftmax.forward(output2);

  std::cout << output2;
}
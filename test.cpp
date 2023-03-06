#include<iostream>
using size_type = std::size_t;
#include<random>
#include<vector>

#include "data/data_reader.h"
#include "source/tensor_algebra.h"
#include "source/dataset.h"
#include "source/layers.h"
#include "source/loss_functions.h"


int main(){
  auto [features, targets] = read("data/mnist_test.csv");
  Dataset<32,784> dataset(features, targets, true, true);

  for(auto batch : dataset){
    for(auto target : batch.targets){
      std::cout << target << std::endl;
    }
  }

  Tensor<float, 32, 784> input(0);
  input.random();

  Linear<784, 216> linear_layer1;
  ReLU<32,216> relu1;
  Linear<216,10> linear_layer2;
  LogSoftMax<32,10> log_softmax;
  NLLLoss nll_loss;

/*

  Tensor<float, 32, 216> output1 = linear_layer1.forward(input);
  Tensor<float, 32, 216> output2 = relu1.forward(output1);
  Tensor<float, 32, 10> output3 = linear_layer2.forward(output2);
  Tensor<float, 32, 10> output4 = log_softmax.forward(output3);/*
  Tensor<float, 10, 32> one_hot_labels = one_hot_encoding<10, 32>(labels);
  float loss = nll_loss.forward(output4, one_hot_labels);
  Tensor<float, 32, 10> ones(1);
  Tensor<float, 32, 10> gradient = nll_loss.backward(ones, output4, one_hot_labels);
  Tensor<float, 32, 216> gradient1 = linear_layer2.backward(gradient, output2);
  */
}
#include<iostream>
#include<random>
#include<vector>
#include<memory>
#include<Fastor/Fastor.h>

  using Fastor::all;
  using Fastor::matmul;
  using Fastor::outer;
  using Fastor::transpose;
  using Fastor::sum;
  using Fastor::exp;
  using Fastor::log;

#include "data/data_reader.h"
#include "source/dataset.h"
#include "source/optimizers.h"
#include "source/parameters.h"
#include "source/layers.h"
#include "source/loss_functions.h"


int main(){
  auto [features, targets] = read("data/mnist_test.csv");
  Dataset<32,784> dataset(features, targets, true);

  Linear<784, 128> linear("He");
  ReLU<32, 128> relu;
  Linear<128, 10> linear2("Xavier");
  LogSoftMax<32, 10> logsoftmax;
  NLLLoss<32,10> loss;

  linear.set_optimizer(std::move(std::make_unique<SGD<784,128>>(0.01)));
  linear2.set_optimizer(std::move(std::make_unique<SGD<128,10>>(0.01)));

  for (auto i = 0; i < 5; i++){
    for(auto batch : dataset){
      auto [input, targets] = batch;
      auto one_hot_targets = one_hot_encoding<32,10>(targets);
      
      auto output = linear.forward(input);
      auto output2 = relu.forward(output);
      auto output3 = linear2.forward(output2);
      auto output4 = logsoftmax.forward(output3);
      auto loss_value = loss.forward(output4, one_hot_targets);
      auto lossgradient = loss.backward(output4, one_hot_targets);
      auto gradient2 = linear2.backward(lossgradient, output2);
      auto gradient3 = relu.backward(gradient2, output);
      auto gradient4 = linear.backward(gradient3, input);
      std::cout << loss_value << std::endl;
      linear.update();
      linear2.update();

      dataset.shuffle();
      }
  }

  return 0; ////It Works!!!!!!!!!!!!!!!!!!!
}
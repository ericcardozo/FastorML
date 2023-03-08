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

  Fastor::Tensor<float, 32, 784> input = dataset.begin()->features;
  std::vector<int> labels = dataset.begin()->targets;

  Linear<784, 128> linear("he");
  ReLU<32, 128> relu;
  Linear<128, 10> linear2("xavier");
  LogSoftMax<32, 10> logsoftmax;
  NLLLoss nllloss;

  linear.set_optimizer(std::move(std::make_unique<SGD<784,128>>(0.01)));
  linear2.set_optimizer(std::move(std::make_unique<SGD<784,128>>(0.01)));

  for (auto i = 0; i < 10; i++){
    auto output = linear.forward(input);
    auto output2 = relu.forward(output);
    auto output3 = linear2.forward(output2);
    auto output4 = logsoftmax.forward(output3);
    auto loss = nllloss.forward(output4, labels);

    std::cout << "Loss: " << loss << std::endl;

    auto gradient = nllloss.backward();
    auto gradient2 = linear2.backward(gradient, output2);
    auto gradient3 = relu.backward(gradient2);
    auto gradient4 = linear.backward(gradient3, input);

    linear.update();
    linear2.update();
  }
}
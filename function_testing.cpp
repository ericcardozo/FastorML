#include<iostream>
#include<Fastor/Fastor.h>
using namespace Fastor;
#include "source/activation_functions.h"
#include "source/loss_functions.h"
#include "source/layers.h"

//logsoftmax ok!
//relu ok!
//nll_loss ok!
//one_hot_encoding ok!


int main(){
  Fastor::Tensor<float, 2, 3> x = {{1.242, 2.4360, -1.5399},
                                   {-0.5412, 0.0326, 0.5927}};
  std::vector<int> targets = {0, 1};
  auto one_hot_targets = one_hot_encoding<2, 3>(targets);
  LogSoftMax<2, 3> logsoftmax;
  auto output = logsoftmax.backward(x,one_hot_targets);
  print(output);
  
  //here i test the gradient of the nll_loss function
  //the gradient of the nll_loss function is:






  return 0;
}
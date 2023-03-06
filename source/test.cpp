#include<iostream>
  using size_type = std::size_t;
#include<vector>
#include "../data/data_reader.h"
#include "tensor_algebra.h"
#include<random>

int main(){
  Tensor<float, 1, 3> a(1);
  Tensor<float, 3> b = {1,2,3};
  std::cout << a;
  std::cout << b;
}
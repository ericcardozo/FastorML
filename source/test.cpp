#include<iostream>
#include<vector>
#include<random>
#include<memory>
#include <Fastor/Fastor.h>


int main(){
  Fastor::Tensor<float, 2, 2> a;
  a.eye();
  print(a);
  return 0;
}

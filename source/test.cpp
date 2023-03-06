#include<iostream>
  using size_type = std::size_t;
#include<vector>
#include "../data/data_reader.h"
#include "tensor_algebra.h"
#include <iostream>
  using size_type = std::size_t;
#include<random>
#include<vector>



template<size_type batch_size, size_type features>
class DataLoader{
  public:

    struct Batch{
      size_type size() const {return labels.size();};
      Batch(){labels.reserve(batch_size);};
      std::vector<int> labels;
      Tensor<float, batch_size, features> images;
    };

    using iterator = typename std::vector<Batch>::iterator;
    using const_iterator = typename std::vector<Batch>::const_iterator;

    DataLoader(
      const DataSet& data,
      bool normalize = true,
      bool shuffle = true
    ){
      for(auto i = 0; i < data.size() / batch_size; i++){
        Batch batch;
        for(auto j = 0; j < batch_size; j++){
          batch.labels.push_back(data.labels[i * batch_size + j]);
          batch.images(j,all) = Tensor<float,features>(data.images[i * batch_size + j]);
          if(normalize){
            batch.images(j,all) = batch.images(j,all) / 255;
          }//see this later for more general normalization.
        }
        batches.push_back(batch);
      }
      if(shuffle){
        std::random_device rd;
        std::mt19937 generator(rd());
        std::shuffle(batches.begin(), batches.end(), generator);
      }
    }

    iterator begin(){return batches.begin();}
    iterator end(){return batches.end();}
    const_iterator begin() const {return batches.begin();}
    const_iterator end() const {return batches.end();}

  private:

    std::vector<Batch> batches;
};

int main(){
  Dataset<32, 784> data(read("../data/mnist_test.csv"));
  std::size_t index = 0;
  std::cout << dataloader.begin()->labels[1];
}
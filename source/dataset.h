#ifndef DATASET_H
#define DATASET_H

#include <iostream>

template<std::size_t batch_size, std::size_t feature_size>
class Dataset{
  public:

    struct Batch{
      Batch() = default;
      Fastor::Tensor<float, batch_size, feature_size> features;
      std::vector<int> targets;
    };  


    using iterator = typename std::vector<Batch>::iterator;
    using const_iterator = typename std::vector<Batch>::const_iterator;

    iterator begin(){return dataset_.begin();}
    iterator end(){return dataset_.end();}
    const_iterator begin() const {return dataset_.begin();}
    const_iterator end() const {return dataset_.end();}

    Dataset(
      const std::vector<std::vector<float>>& features,
      const std::vector<int>& targets,
      bool normalize = true
    ){
      for(auto i = 0; i < features.size()/batch_size; i++){
        Batch batch;
        for(auto j = 0; j < batch_size; j++){
          batch.features(j,all) = Fastor::Tensor<float,feature_size>(features[i * batch_size + j]);
          batch.targets.push_back(targets[i * batch_size + j]);
        }
        if(normalize){//fix this later with a proper normalizer function.
          batch.features /= 255;
        }
        dataset_.emplace_back(std::move(batch));
      }
    }

    void shuffle(){
      std::random_device rd;
      std::mt19937 generator(rd());
      std::shuffle(dataset_.begin(), dataset_.end(), generator);
    }

  private:
    std::vector<Batch> dataset_;
};

#endif
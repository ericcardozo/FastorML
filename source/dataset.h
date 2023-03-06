#ifndef DATASET_H
#define DATASET_H

//std::pair<std::vector<std::vector<float>>, std::vector<int>>

template<size_type batch_size, size_type feature_size>
class Dataset{
  public:
    using Batch = std::pair<Tensor<float, batch_size, feature_size>, std::vector<int>>;
    using iterator = typename std::vector<Batch>::iterator;
    using const_iterator = typename std::vector<Batch>::const_iterator;

    iterator begin(){return batches.begin();}
    iterator end(){return batches.end();}
    const_iterator begin() const {return batches.begin();}
    const_iterator end() const {return batches.end();}

    //add a normalizer function here!

    Dataset(const std::pair<std::vector<std::vector<float>>, std::vector<int>>& data,
            bool shuffle = true){
      for(auto i = 0; i < data.first.size() / batch_size; i++){
        Batch batch;
        batch.second.reserve(batch_size);
        for(auto j = 0; j < batch_size; j++){
          batch.first(j,all) = Tensor<float,feature_size>(data.first[i * batch_size + j]);
          batch.second.push_back(data.second[i * batch_size + j]);
        }
        batches.emplace_back(std::move(batch));
      }
      if(shuffle){
        std::random_device rd;
        std::mt19937 generator(rd());
        std::shuffle(batches.begin(), batches.end(), generator);
      }
    }


  private:
    std::vector<Batch> batches;
};

#endif
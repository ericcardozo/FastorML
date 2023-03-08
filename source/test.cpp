#include<iostream>
  using std::size_t = std::size_t;
#include<vector>
#include<random>
#include<memory>
//#include "Fastor::Tensor_algebra.h"
#include <Fastor/Fastor.h>
using namespace Fastor;

template<std::size_t input_features, std::size_t output_features>
struct Parameters;

/*
template<class Derived>
class Optimizer{
  public:
    Derived& self(){return static_cast<Derived&>(*this);}
    const Derived& self() const {return static_cast<const Derived&>(*this);}

    template<std::size_t input_size, std::size_t output_size>
    void update(Parameters<input_size,output_size>& parameters){
      self().update(parameters);
    }
};*/

class Optimizer{
  public:
    virtual void update(Parameters& parameters) = 0;
    virtual ~Optimizer() = default;
};

template<std::size_t input_features, std::size_t output_features>
struct Parameters{
    Fastor::Tensor<float, input_features, output_features> weight;
    Fastor::Tensor<float, input_features, output_features> weight_gradient;
    Fastor::Tensor<float, output_features> bias;
    Fastor::Tensor<float, output_features> bias_gradient;


    Parameters(const std::string& initializer = "he")
    : bias(0,0) {

      std::random_device rd;
      std::mt19937 generator(rd());
      std::normal_distribution<float> distribution;

      switch(initializer){
        case "he":
          distribution = std::normal_distribution<float>(0, std::sqrt(2.0 / input_features));
        break;

        case "xavier":
          distribution = std::normal_distribution<float>(0, std::sqrt(2.0 / input_features + output_features));  
        break;

        default:
          std::cout << "Invalid initializer" << std::endl;
        break;
      }

      for(auto i = 0; i < input_features; ++i){
        for(auto j = 0; j < output_features; ++j){
          weight(i, j) = distribution(generator);
        }
      }
    }

    void update(){
      optimizer->update(*this);
    }
};

/*
class SGD : public Optimizer<SGD>{
  public:
    SGD(float learning_rate) : learning_rate_(learning_rate) {}

    template<std::size_t input_size, std::size_t output_size>
    void update(Parameters<input_size,output_size>& parameters){
      parameters.weight -= learning_rate_ * parameters.weight_gradient;
      parameters.bias -= learning_rate_ * parameters.bias_gradient;
    }

  private:
    float learning_rate_;
};
*/

class SGD : public Optimizer{
  public:
    SGD(float learning_rate) : learning_rate_(learning_rate) {}

    void update(Parameters& parameters){
      parameters.weight -= learning_rate_ * parameters.weight_gradient;
      parameters.bias -= learning_rate_ * parameters.bias_gradient;
    }

  private:
    float learning_rate_;
};

//Linear layer class

template<std::size_t input_features, std::size_t output_features>
class Linear{
  public:

    Linear(float learning_rate, const std::string& initializer = "he")
      : parameters(learning_rate, initializer){}
    
    void set_optimizer(std::shared_ptr<Optimizer> optimizer){
      parameters.optimizer = optimizer;
    }
    //forward method
    template<std::size_t batch_size>
    Fastor::Tensor<float, batch_size, output_features> forward(const Fastor::Tensor<float, batch_size, input_features> &input){
      Fastor::Tensor<float,batch_size> ones(1);
      return matmul(input, parameters.weight) + outer(ones, parameters.bias); 
    }
  
    //backward method
    template<std::size_t batch_size>
    Fastor::Tensor<float, batch_size, input_features> backward(
      const Fastor::Tensor<float, batch_size, output_features> &gradient,
      const Fastor::Tensor<float, batch_size, input_features>& input
    ){
      parameters.weight_gradient = matmul(transpose(input), gradient);
      parameters.bias_gradient = 0.0; // Initialize bias_gradient with zeros
      
      for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < output_features; j++){
        parameters.bias_gradient(j) += gradient(i, j);
        }
      }

      Fastor::Tensor<float, batch_size, input_features> input_gradient = matmul(gradient, transpose(parameters.weight));
      return input_gradient;
    }
  
  private:
    Parameters<input_features, output_features> parameters;
};


int main(){
  Linear<2,3> linear(0.01);
  linear.set_optimizer(std::make_shared<SGD>(0.01));
  Fastor::Tensor<float, 2, 2> input = {{1, 2}, {3, 4}};
  Fastor::Tensor<float, 2, 3> output = linear.forward(input);
  std::cout << output << std::endl;
  return 0;
}

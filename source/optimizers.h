#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

class Optimizer{
  public:
    virtual void update(Parameters& parameters) = 0;
    virtual ~Optimizer() = default;
};

class SGD : public Optimizer{
  public:
    SGD(float learning_rate) : learning_rate_(learning_rate) {}
    void update(Parameters& parameters) override {
      parameters.weight -= learning_rate_ * parameters.weight_gradient;
      parameters.bias -= learning_rate_ * parameters.bias_gradient;
    }

private:
  float learning_rate_;
};

#endif
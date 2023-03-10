import torch
import numpy as np

import numpy as np

import torch

def log_softmax_naive(input):
    # Subtract the maximum value along the second dimension of the input tensor
    input_max, _ = torch.max(input, dim=1, keepdim=True)
    input_exp = torch.exp(input - input_max)
    
    # Compute the sum of the exponentiated input values along the second dimension
    input_exp_sum = torch.sum(input_exp, dim=1, keepdim=True)
    
    # Compute the log softmax of the input tensor
    log_softmax = input - input_max - torch.log(input_exp_sum)
    
    return log_softmax

def nllloss_naive(log_probs, target):
    # Select the log probability corresponding to the target label for each sample
    log_prob_target = torch.gather(log_probs, 1, target.view(-1,1))
    
    # Compute the negative log likelihood loss
    nllloss = -torch.mean(log_prob_target)
    
    return nllloss

def nll_loss_grad(input, target):
    # compute softmax of input
    exp_input = np.exp(input)
    softmax = exp_input / np.sum(exp_input, axis=1, keepdims=True)
    
    # compute one-hot target vector
    num_classes = input.shape[1]
    one_hot_target = np.zeros_like(softmax)
    one_hot_target[np.arange(len(target)), target] = 1
    
    # compute gradient of loss function
    grad = softmax - one_hot_target
    grad /= len(target)  # divide by batch size to compute average gradient
    
    return grad

    
def log_softmax_grad(input):
    # compute softmax of input
    input_max = np.max(input, axis=1, keepdims=True)
    exp_input = np.exp(input - input_max)
    softmax = exp_input / np.sum(exp_input, axis=1, keepdims=True)
    
    # compute gradient of log softmax
    grad = np.zeros_like(input)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            for k in range(input.shape[1]):
                if j == k:
                    grad[i,j] += softmax[i,k] * (1 - softmax[i,j])
                else:
                    grad[i,j] -= softmax[i,k] * softmax[i,j]
    
    return grad

import torch

# Define the input tensor and target tensor
input = torch.tensor([[1.242, 2.4360, -1.5399], [-0.5412, 0.0326, 0.5927]], requires_grad=True)
target = torch.tensor([0, 1])

# Compute the log softmax of the input tensor
log_probs = torch.nn.functional.log_softmax(input, dim=1)

# Compute the negative log likelihood loss of the log probabilities and target tensor
loss = torch.nn.functional.nll_loss(log_probs, target)

# Compute the gradient of the loss with respect to the log probabilities
grad_output = torch.ones_like(log_probs) * (-1)
log_probs.backward(grad_output)

# Compute the gradient of the log softmax using the log_softmax_grad function
grad_log_softmax = log_softmax_grad(input.detach().numpy())

# Print the gradient of the input tensor computed using both methods
print(grad_log_softmax)
print(input.grad)
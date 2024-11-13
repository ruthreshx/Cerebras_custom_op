#include "heaviside.h"


namespace custom_namespace {

  //Custom Heaviside implementation
  torch::Tensor Heaviside::compute_heaviside(const torch::Tensor& input, const torch::Tensor& values) {
  
      // Create masks for each condition
      torch::Tensor positive_mask = (input > 0);
      torch::Tensor zero_mask = (input == 0);

      // Apply the conditions:
      // If x > 0, result is 1
      // If x == 0, result is y
      // If x < 0, result is 0
      torch::Tensor result = positive_mask * torch::ones_like(input) + zero_mask * values;
      return result;
  }
}


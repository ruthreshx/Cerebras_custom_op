#include "heaviside.h"

// Heaviside implementation
namespace custom_namespace {
  torch::Tensor Heaviside::compute_heaviside(const torch::Tensor& input, const torch::Tensor& values) {
  
      // Apply condition based on the input tensor's value
      torch::Tensor result = torch::where(input < 0, torch::zeros_like(input), torch::ones_like(input));
      result = torch::where(input == 0, values * torch::ones_like(input), result);
      return result;
  }
}


#include <elu.h>

namespace custom_namespace {
  // Custom ELU function implementation
  torch::Tensor ELU::compute_elu(const torch::Tensor& input, double alpha) {
      // Initialize the result tensor
      torch::Tensor result = torch::zeros_like(input);
  
      // For x >= 0, ELU(x) = x
      result = torch::where(input >= 0, input, result);
  
      // For x < 0, ELU(x) = alpha * (exp(x) - 1)
      result = torch::where(input < 0, alpha * (torch::exp(input) - 1), result);
  
      return result;
  }
}
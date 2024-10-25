#include <entr.h>

namespace custom_namespace {
  // Custom entr function implementation
  torch::Tensor Entr::compute_entr(const torch::Tensor& input) {
      // Create a tensor filled with -infinity for input < 0
      torch::Tensor neg_inf = torch::full_like(input, -std::numeric_limits<double>::infinity());
  
      // Apply the entr operation
      torch::Tensor result = torch::zeros_like(input);  // Initialize result tensor
  
      // Apply the formula: result = -x * log(x) where x > 0
      result = torch::where(input > 0, -input * torch::log(input), result);
  
      // Set the result to 0 where input == 0
      result = torch::where(input == 0, torch::zeros_like(input), result);
  
      // Set the result to -infinity where input < 0
      result = torch::where(input < 0, neg_inf, result);
  
      return result;
  }
}

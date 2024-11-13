#include <elu.h>

namespace custom_namespace {

  // Custom ELU function implementation
  torch::Tensor ELU::compute_elu(const torch::Tensor& input, double alpha) {

    // Create masks for the conditions
    torch::Tensor positive_mask = (input > 0);
    torch::Tensor negative_mask = (input <= 0);

    // Apply ELU formula: input if input > 0, alpha * (exp(input) - 1) if input <= 0
    torch::Tensor positive_result = positive_mask * input;
    torch::Tensor negative_result = negative_mask * (alpha * (torch::exp(input) - 1));

    // Combine the results
    torch::Tensor result = positive_result + negative_result;

    return result;
  
  }
}
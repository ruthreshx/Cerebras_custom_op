#include <entr.h>

namespace custom_namespace {
  // Custom entr function implementation
  torch::Tensor Entr::compute_entr(const torch::Tensor& input) {

    // Create masks for each condition
    torch::Tensor positive_mask = (input > 0);
    torch::Tensor zero_mask = (input == 0);
    torch::Tensor negative_mask = (input < 0);

    // Compute -input * log(input) for positive values
    torch::Tensor positive_result = positive_mask * (-input * torch::log(input));

    // Zero where input is exactly zero
    torch::Tensor zero_result = zero_mask * torch::zeros_like(input);

    // -inf for negative values
    torch::Tensor negative_result = negative_mask * -std::numeric_limits<float>::infinity();

    // Combine the results
    torch::Tensor result = positive_result + zero_result + negative_result;
    return result;

  }
}

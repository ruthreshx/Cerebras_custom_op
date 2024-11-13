#include <entr.h>

namespace custom_namespace {
  // Custom entr function implementation
  torch::Tensor Entr::compute_entr(const torch::Tensor& input) {

    // Create the result tensor by initializing it to zeros (for x == 0 cases)
    torch::Tensor result = torch::zeros_like(input);

    // Calculate -x * ln(x) for x > 0
    torch::Tensor positive_x = input * (input > 0);  // Retains only positive values, others are zero
    torch::Tensor positive_result = -positive_x * torch::log(positive_x);
    
    // Combine the result, keeping zeros where x == 0
    result += positive_result;

    return result;

  }
}

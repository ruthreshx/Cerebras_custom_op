#include <logaddexp.h>

namespace custom_namespace {

  // Custom Copysign function implementation
  torch::Tensor LogAddExp::compute_logaddexp(const torch::Tensor& input, const torch::Tensor& other) {

    // Compute the element-wise maximum of two tensors
    torch::Tensor max_val = torch::max(input, other);

    // Compute the exp between the diff of two tensors
    torch::Tensor diff_exp = torch::exp(-torch::abs(input - other));

    // Add the max_val and log result of computed exp difference
    return max_val + torch::log1p(diff_exp);
  
  }
}
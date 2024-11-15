#include <copysign.h>

namespace custom_namespace {

  // Custom Copysign function implementation
  torch::Tensor CopySign::compute_copysign(const torch::Tensor& input, const torch::Tensor& other) {

    // Ensure both tensors have the same shape or are broadcastable
    TORCH_CHECK(input.sizes() == other.sizes() || 
                input.sizes().size() == other.sizes().size(),
                "Input and sign_tensor must have the same shape or be broadcastable.");

    // Get the magnitude of input (absolute value)
    torch::Tensor abs_input = torch::abs(input);

    // Extract the sign of sign_tensor using signbit
    torch::Tensor sign_mask = torch::signbit(other);

    // Apply sign_mask: if sign_mask is true (1), negate the magnitude; otherwise, keep it positive
    torch::Tensor result = abs_input.where(sign_mask, -abs_input);

    return result;
  
  }
}
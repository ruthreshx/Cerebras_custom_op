#include <copysign.h>

namespace custom_namespace {

  // Custom Copysign function implementation
  torch::Tensor CopySign::compute_copysign(const torch::Tensor& input, const torch::Tensor& other) {

    // Ensure both tensors have the same shape or are broadcastable
    TORCH_CHECK(input.sizes() == other.sizes() || 
                input.sizes().size() == other.sizes().size(),
                "Input and sign_tensor must have the same shape or be broadcastable.");

    // // Get the magnitude of input (absolute value)
    // torch::Tensor abs_input = torch::abs(input);

    // // Extract the sign of sign_tensor using signbit
    // torch::Tensor sign_mask = torch::signbit(other);

    // // Apply sign_mask: if sign_mask is true (1), negate the magnitude; otherwise, keep it positive
    // torch::Tensor result = abs_input.where(sign_mask, -abs_input);

    // Calculate the sign of each element in tensor B
    // torch::signbit returns -1 for negative values and 1 for positive values
    torch::Tensor sign_of_B = torch::signbit(other).to(torch::kFloat);

    // negative values become -1, positive values become +1
    torch::Tensor sign_adjustment = (-2 * sign_of_B) + 1;

    // Apply the sign of B to A by multiplying the absolute value of A with the calculated sign adjustment
    torch::Tensor result = torch::abs(input) * sign_adjustment;

    // Return the final result tensor
    return result;
  
  }
}
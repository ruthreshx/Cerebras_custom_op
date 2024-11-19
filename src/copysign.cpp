#include <copysign.h>

namespace custom_namespace {

  // Custom Copysign function implementation
  torch::Tensor CopySign::compute_copysign(const torch::Tensor& input, const torch::Tensor& other) {

    // Ensure both tensors have the same shape or are broadcastable
    TORCH_CHECK(input.sizes() == other.sizes() || 
                input.sizes().size() == other.sizes().size(),
                "Input and sign_tensor must have the same shape or be broadcastable.");


    // // Calculate the sign of each element in tensor B
    // // torch::signbit returns -1 for negative values and 1 for positive values
    // torch::Tensor sign_of_B = torch::signbit(other).to(torch::kFloat);

    // // negative values become -1, positive values become +1
    // torch::Tensor sign_adjustment = (-2 * sign_of_B) + 1;

    // // Apply the sign of B to A by multiplying the absolute value of A with the calculated sign adjustment
    // torch::Tensor result = torch::abs(input) * sign_adjustment;

    // // Return the final result tensor
    // return result;


    // Ensure both tensors are of floating-point type
    TORCH_CHECK(input.is_floating_point(), "Magnitude source must be a floating-point tensor.");
    TORCH_CHECK(other.is_floating_point(), "Sign source must be a floating-point tensor.");

    // Handle 32-bit (float) and 64-bit (double) floating-point tensors
    // if (input.scalar_type() == torch::kFloat) {
    // 32-bit floating-point
    auto mag_bits = input.bitwise_and(0x7FFFFFFF);  // Clear sign bit
    auto sign_bits = other.bitwise_and(0x80000000);     // Extract sign bit
    return mag_bits.bitwise_or(sign_bits).view_as(input);
    // }
  
  }
}
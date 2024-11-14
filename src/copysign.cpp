#include <copysign.h>

namespace custom_namespace {

  // Custom Copysign function implementation
  torch::Tensor CopySign::compute_copysign(const torch::Tensor& input, const torch::Tensor& other) {

        // Separate the integer part from the fractional part of each element
    torch::Tensor integer_part = torch::floor(input);  // Floor of each element
    torch::Tensor fractional_part = input - integer_part;

    // Check where there is a fractional part (i.e., non-zero fraction)
    torch::Tensor has_fraction = fractional_part > 0;

    // Increment integer_part by 1 where there is a non-zero fractional part
    return integer_part + has_fraction;
  
  }
}
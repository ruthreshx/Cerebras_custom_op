#include <addmv.h>

namespace custom_namespace {

  // Custom addmv function implementation
  torch::Tensor AddMV::compute_addmv(const torch::Tensor& input, const torch::Tensor& matrix, const torch::Tensor& vector, double beta, double alpha) {

    // Check dimensions
    if (matrix.dim() != 2 || vector.dim() != 1 || input.dim() != 1) {
        throw std::invalid_argument("Matrix must be 2-D, vector must be 1-D, and input must be 1-D.");
    }
    
    // Check size compatibility
    if (matrix.size(1) != vector.size(0)) {
        throw std::invalid_argument("The number of columns in matrix must match the size of vector.");
    }
    if (matrix.size(0) != input.size(0)) {
        throw std::invalid_argument("The number of rows in matrix must match the size of input.");
    }

    // Handle alpha == 0 && beta == 0
    if (alpha == 0 && beta == 0){
      return torch::zeros_like(input);
    }
    else if(alpha == 0){ // Handle alpha == 0 
      return beta * input;
    }
    else if(beta == 0){ // Handle beta == 0 
      return torch::mv(matrix, vector) * alpha;
    }
    
    // Cout = β input+α (mat@vec)
    return alpha * torch::mv(matrix, vector) + beta * input;

  }
}
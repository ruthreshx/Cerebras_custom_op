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

    // Handle alpha == 0 & beta == 0 or beta == 1
    if (alpha == 0) {
        // Skip matrix-vector multiplication; use only scaled input (if beta > 0)
        if (beta == 0) {
            return torch::zeros_like(input);  // If beta == 0, the result is zeros
        } else if (beta == 1) {
            return input;  // If beta == 1, the result is just the input
        } else {
            return beta * input;  // For other beta values, scale input by beta
        }
    } 
    else {

        // Perform matrix-vector multiplication (mat @ vec)
        torch::Tensor mat_vec_product =  torch::mv(matrix, vector);

        // Scale the result of matrix-vector multiplication by alpha if alpha is not 1
        torch::Tensor scaled_mat_vec_product = (alpha == 1) ? mat_vec_product : alpha * mat_vec_product;

        if (beta == 0) {
            return torch::mv(matrix, vector);  // No input scaling, just matrix-vector product
        } else if (beta == 1) {
            return input + torch::mv(matrix, vector);  // If beta == 1, add input to the product
        } else {
            // out=β input + α (mat @ vec)
            return beta * input + torch::mv(matrix, vector);  // For other beta values, scale input by beta
        }
    }

  }
}
#include <addmm.h>

namespace custom_namespace {

  torch::Tensor custom_mm(torch::Tensor mat1, torch::Tensor mat2) {

    // Check if both matrices are 2D
    if (mat1.dim() != 2 || mat2.dim() != 2) {
        throw std::invalid_argument("Both inputs must be 2D matrices");
    }

    // Check if inner dimensions of the matrices match (for matrix multiplication)
    if (mat1.size(1) != mat2.size(0)) {
        throw std::invalid_argument("Inner dimensions must match");
    }

    // Get the dimensions of the matrices
    int64_t m = mat1.size(0);  // Rows of mat1
    int64_t n = mat2.size(1);  // Columns of mat2

    // Initialize an empty tensor to store the result of matrix multiplication
    torch::Tensor result = torch::zeros({m, n}, mat1.options());

    // Perform matrix multiplication by calculating the dot product
    // of rows of mat1 with columns of mat2.
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            // The element at result[i, j] is the dot product of row i of mat1 and column j of mat2
            result[i][j] = torch::dot(mat1[i], mat2.transpose(0, 1)[j]);
        }
    }

    // Return the result of the matrix multiplication
    return result;
  }

  // Custom AddMM function implementation
  torch::Tensor AddMM::compute_addmm(const torch::Tensor& input, const torch::Tensor& mat1, const torch::Tensor& mat2, double beta, double alpha) {

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

        // Perform matrix multiplication (mat1 @ mat2)
        torch::Tensor mat_product =  custom_mm(mat1, mat2);

        // Scale the result of matrix-vector multiplication by alpha if alpha is not 1
        torch::Tensor scaled_mat_product = (alpha == 1) ? mat_product : alpha * mat_product;

        if (beta == 0) {
            return scaled_mat_product;  // No input scaling, just matrix-vector product
        } else if (beta == 1) {
            return input + scaled_mat_product;  // If beta == 1, add input to the product
        } else {
            // out=β input + α (mat1 @ mat2)
            return beta * input + scaled_mat_product;  // For other beta values, scale input by beta
        }
    }

  }
}
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

    // Handle alpha == 0 && beta == 0
    if (alpha == 0 && beta == 0){
      return torch::zeros_like(input);
    }
    else if(alpha == 0){ // Handle alpha == 0 
      return beta * input;
    }
    else if(beta == 0){ // Handle beta == 0 
      return custom_mm(mat1, mat2) * alpha;
    }

    // out=β input+α (mat1 @ mat2)
    return alpha * custom_mm(mat1, mat2) + beta * input;

  }
}
#include <addbmm.h>

namespace custom_namespace {

  torch::Tensor custom_bmm(torch::Tensor A, torch::Tensor B){

     // Ensure dimensions are compatible
    auto batch_size = A.size(0);
    auto M = A.size(1);
    auto N = A.size(2);
    auto P = B.size(2);

    TORCH_CHECK(B.size(0) == batch_size, "A and B must have the same batch size.");
    TORCH_CHECK(B.size(1) == N, "Inner dimensions of A and B must match.");

    // Perform batched matrix multiplication without loops
    // Expand A and B for broadcasting
    torch::Tensor A_expanded = A.unsqueeze(-1);  // Shape: (batch_size, M, N, 1)
    torch::Tensor B_expanded = B.unsqueeze(1);   // Shape: (batch_size, 1, N, P)

    // Element-wise multiplication and summation over batch dimension
    torch::Tensor product = (A_expanded * B_expanded).sum(2);  // Shape: (batch_size, M, P)

    return product;
  }


  // Custom function implementation
  torch::Tensor AddBMM::compute_addbmm(const torch::Tensor& input, const torch::Tensor& batch1, const torch::Tensor& batch2, double alpha, double beta) {

    // Ensure dimensions are compatible
    TORCH_CHECK(batch1.dim() == 3 && batch2.dim() == 3, "batch1 and batch2 must be 3-dimensional tensors.");
    TORCH_CHECK(batch1.size(0) == batch2.size(0), "batch1 and batch2 must have the same batch size.");
    TORCH_CHECK(batch1.size(2) == batch2.size(1), "batch1's last dimension must match batch2's second-to-last dimension.");
    TORCH_CHECK(input.size(0) == batch1.size(1) && input.size(1) == batch2.size(2),
                "input tensor must match the dimensions of the result of batch1 @ batch2.");

      // Handle alpha == 0 & beta == 0 or beta == 1
    if (alpha == 0) {
        // Skip matrix-vector multiplication; use only scaled input (if beta > 0)
        if (beta == 0) {
            return torch::zeros_like(input);  // If beta == 0, the result is zeros
        } 
        if (beta == 1) {
            return input;  // If beta == 1, the result is just the input
        } 

        return beta * input;  // For other beta values, scale input by beta

    } 
    else {

        // Perform matrix multiplication (mat1 @ mat2)
        torch::Tensor mat_product =  custom_bmm(batch1, batch2);

        // Scale the result of matrix-vector multiplication by alpha if alpha is not 1
        torch::Tensor batch_mat_product = (alpha == 1) ? mat_product : alpha * mat_product;

        if (beta == 0) {
            return batch_mat_product;  // No input scaling, just matrix-vector product
        }
        if (beta == 1) {
            return input + batch_mat_product;  // If beta == 1, add input to the product
        }
        // out=β input + α (mat1 @ mat2)
        return beta * input + batch_mat_product;  // For other beta values, scale input by beta
    }
  }
}
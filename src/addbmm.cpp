#include <addbmm.h>

namespace custom_namespace {
  // Custom function implementation
  torch::Tensor AddBMM::compute_addbmm(const torch::Tensor& input, const torch::Tensor& batch1, const torch::Tensor& batch2, double alpha, double beta) {

    // Ensure dimensions are compatible
    TORCH_CHECK(batch1.dim() == 3 && batch2.dim() == 3, "batch1 and batch2 must be 3-dimensional tensors.");
    TORCH_CHECK(batch1.size(0) == batch2.size(0), "batch1 and batch2 must have the same batch size.");
    TORCH_CHECK(batch1.size(2) == batch2.size(1), "batch1's last dimension must match batch2's second-to-last dimension.");
    TORCH_CHECK(input.size(0) == batch1.size(1) && input.size(1) == batch2.size(2),
                "input tensor must match the dimensions of the result of batch1 @ batch2.");

    // if both alpha and beta equals 0, result is 0
    if(alpha == 0 and beta == 0){
        return torch::zeros_like(input);
    }
    if(alpha == 0){
        return beta * input;
    }
    // first dimension of the input matrices gives the no of iterations (batches) to run
    int batch_size = batch1.size(0);

    //output tensor with 2nd dimension of matrix 1 and 3rd dimension of matrix 2 to store the result of batchwise matrix multiplication
    auto output = torch::zeros({batch1.size(1),batch2.size(2)});

    // scale the batched product result with alpha and add it to the previous batch result
    for(int i =0;i<batch_size;i++){
        auto result = torch::matmul(batch1[i], batch2[i]);
        output = output + (alpha * result);
    }

    return beta * input + output;
  
  }
}
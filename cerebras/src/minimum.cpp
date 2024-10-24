#include "minimum.h"

namespace custom_namespace {
    torch::Tensor Minimum::compute_min(const torch::Tensor& tensor1, const torch::Tensor& tensor2) {

    // Ensure that tensor1 and tensor2 have the same shape
        if (tensor1.sizes() != tensor2.sizes()) {
            throw std::invalid_argument("Input tensors must have the same shape.");
        }
    
    // Apply condition based on the input tensor's value 
    torch::Tensor result = torch::where(tensor1 < tensor2, tensor1, tensor2);
    return result;

    }
}

#include "minimum.h"

namespace custom_namespace {
    torch::Tensor Minimum::compute_min(const torch::Tensor& tensor1, const torch::Tensor& tensor2) {

    // Ensure that tensor1 and tensor2 have the same shape
        if (tensor1.sizes() != tensor2.sizes()) {
            throw std::invalid_argument("Input tensors must have the same shape.");
        }
    
        // Calculate the difference between tensors
        torch::Tensor diff = tensor1 - tensor2;

        // Compute the element-wise minimum using (a + b - |a - b|) / 2
        torch::Tensor result = (tensor1 + tensor2 - torch::abs(diff)) / 2;

        return result;

    }
}

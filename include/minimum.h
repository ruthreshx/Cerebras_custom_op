#ifndef MINIMUM_H

#include <torch/torch.h>

namespace custom_namespace {
    class Minimum {
    public:
        Minimum() {}

        // Function to compute element-wise minimum of two tensors
        torch::Tensor compute_min(const torch::Tensor& tensor1, const torch::Tensor& tensor2);
    };
}

#endif

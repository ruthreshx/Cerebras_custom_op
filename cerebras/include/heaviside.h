#ifndef MINIMUM_H

#include <torch/torch.h>

namespace custom_namespace {
    class Heaviside {
    public:
        Heaviside() {}

        // Function to compute element-wise minimum of two tensors
        torch::Tensor compute_heaviside(const torch::Tensor& input, const torch::Tensor& values);
    };
}

#endif

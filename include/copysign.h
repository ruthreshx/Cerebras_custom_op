#ifndef COPYSIGN_H

#include <torch/torch.h>


namespace custom_namespace {
    class CopySign {
    public:
        CopySign() {}

        // Custom copysign function implementation
        torch::Tensor compute_copysign(const torch::Tensor& input, const torch::Tensor& other);
  };
}

#endif // COPYSIGN_H

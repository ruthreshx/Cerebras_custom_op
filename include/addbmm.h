#ifndef ADDBMM_H

#include <torch/torch.h>

namespace custom_namespace {
    class AddBMM {
    public:
          AddBMM() {}

          // Custom ELU function implementation
          torch::Tensor compute_addbmm(const torch::Tensor& input, const torch::Tensor& batch1, const torch::Tensor& batch2, double alpha, double beta);
  };
}
#endif // ADDBMM_H

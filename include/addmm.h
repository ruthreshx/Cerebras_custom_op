#ifndef ADDMM_H

#include <torch/torch.h>

namespace custom_namespace {
    class AddMM {
    public:
          AddMM() {}

          // Custom AddMM function implementation
          torch::Tensor compute_addmm(const torch::Tensor& input, const torch::Tensor& mat1, const torch::Tensor& mat2, double beta, double alpha);
  };
}
#endif // ADDMM_H

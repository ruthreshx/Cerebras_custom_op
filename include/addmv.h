#ifndef ADDMV_H

#include <torch/torch.h>

namespace custom_namespace {
    class AddMV {
    public:
          AddMV() {}

          // Custom ELU function implementation
          torch::Tensor compute_addmv(const torch::Tensor& input, const torch::Tensor& matrix, const torch::Tensor& vector, double beta, double alpha);
  };
}
#endif // ADDMV_H

#ifndef ELU_H

#include <torch/torch.h>

namespace custom_namespace {
    class ELU {
    public:
          ELU() {}

          // Custom ELU function implementation
          torch::Tensor compute_elu(const torch::Tensor& input, double alpha);
  };
}
#endif // ELU_H

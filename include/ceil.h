#ifndef CEIL_H

#include <torch/torch.h>

namespace custom_namespace {
    class Ceil {
    public:
          Ceil() {}

          // Custom Ceil function implementation
          torch::Tensor compute_ceil(const torch::Tensor& input);
  };
}
#endif // CEIL_H

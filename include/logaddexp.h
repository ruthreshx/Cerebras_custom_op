#ifndef LOGADDEXP_H

#include <torch/torch.h>
#include <limits>  // For -infinity

namespace custom_namespace {
    class LogAddExp {
    public:
        LogAddExp() {}

        // Custom LogAddExp function implementation
        torch::Tensor compute_logaddexp(const torch::Tensor& input, const torch::Tensor& other);
  };
}

#endif // LOGADDEXP_H

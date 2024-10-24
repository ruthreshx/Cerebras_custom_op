#ifndef ENTR_H

#include <torch/torch.h>
#include <iostream>
#include <limits>  // For -infinity

namespace custom_namespace {
    class Entr {
    public:
        Entr() {}

        // Custom Entr function implementation
        torch::Tensor compute_entr(const torch::Tensor& input);
  };
}

#endif // ENTR_H

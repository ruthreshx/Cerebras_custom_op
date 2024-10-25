// test_elu.cpp
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "elu.h"

namespace custom_namespace {

    // Test for the compute_elu method with positive alpha
    TEST(EluTest, PositiveAlpha) {
        custom_namespace::ELU elu_instance;
        torch::Tensor input = torch::tensor({-1.0, 2.0, 1.0, 20.0});
        double alpha = 1.0;

        // Call compute_elu
        torch::Tensor result = elu_instance.compute_elu(input, alpha);

        // Expected output for ELU with alpha = 1.0 on [-1, 0, 1, 2]
        torch::Tensor expected = torch::elu(input, alpha);

        // Test if the result is close to the expected output
        ASSERT_TRUE(torch::allclose(result, expected, 1e-4));
    }

    // Test for the compute_elu method with a negative input tensor
    TEST(EluTest, NegativeInputs) {
        custom_namespace::ELU elu_instance;
        torch::Tensor input = torch::tensor({-1.5, -0.5, -0.1, 0.0});
        double alpha = 1.0;

        // Call compute_elu
        torch::Tensor result = elu_instance.compute_elu(input, alpha);

        // Expected output for ELU with alpha = 1.0 on negative inputs
        torch::Tensor expected = torch::elu(input, alpha);

        // Test if the result matches the expected output
        ASSERT_TRUE(torch::allclose(result, expected, 1e-4));
    }

}  // namespace custom_namespace

// test_heaviside.cpp
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "heaviside.h"

namespace custom_namespace {

    // Test for positive inputs to compute_heaviside
    TEST(HeavisideTest, PositiveInput) {
        custom_namespace::Heaviside heaviside_instance;
        torch::Tensor input = torch::tensor({1.0, 2.0, 3.0});
        torch::Tensor values = torch::tensor({-1.0, -2.0, -3.0});
        
        // Call compute_heaviside
        torch::Tensor result = heaviside_instance.compute_heaviside(input, values);

        // Expected output: tensor of ones since input is positive
        torch::Tensor expected = torch::heaviside(input, values);

        ASSERT_TRUE(torch::allclose(result, expected, 1e-4));
    }

    // Test for zero input with default h value
    TEST(HeavisideTest, ZeroInputDefaultH) {
        custom_namespace::Heaviside heaviside_instance;
        torch::Tensor input = torch::tensor({0.0});
        torch::Tensor values = torch::tensor({0.0});
        
        // Call compute_heaviside with default h (0.5)
        torch::Tensor result = heaviside_instance.compute_heaviside(input, values);

        // Expected output: tensor of 0.5 for input of zero
        torch::Tensor expected = torch::heaviside(input, values);

        ASSERT_TRUE(torch::allclose(result, expected, 1e-4));
    }

    // Test for zero input with custom h value
    TEST(HeavisideTest, ZeroInputCustomH) {
        custom_namespace::Heaviside heaviside_instance;
        torch::Tensor input = torch::tensor({2.0});
        torch::Tensor values = torch::tensor({0.7});
        
        // Call compute_heaviside with h = 0.7
        torch::Tensor result = heaviside_instance.compute_heaviside(input, values);

        // Expected output: tensor of 0.7 for input of zero
        torch::Tensor expected = torch::heaviside(input, values);

        ASSERT_TRUE(torch::allclose(result, expected, 1e-4));
    }

    // Test for negative inputs to compute_heaviside
    TEST(HeavisideTest, NegativeInput) {
        custom_namespace::Heaviside heaviside_instance;
        torch::Tensor input = torch::tensor({-1.0, 2.0, -3.0});
       torch::Tensor values = torch::tensor({-2.0, 4.0, 6.0});
        
        // Call compute_heaviside
        torch::Tensor result = heaviside_instance.compute_heaviside(input, values);

        // Expected output: tensor of zeros since input is negative
        torch::Tensor expected = torch::heaviside(input, values);

        ASSERT_TRUE(torch::allclose(result, expected, 1e-4));
    }

} // namespace custom_namespace

// test_minimum.cpp
#include <gtest/gtest.h>
#include "minimum.h"  // Include the custom Minimum class header

// Test case for tensors with matching shapes and element-wise minimum
TEST(MinimumTest, MatchingShape) {
    torch::Tensor tensor1 = torch::tensor({1.0, 5.0, 3.0});
    torch::Tensor tensor2 = torch::tensor({4.0, 2.0, 6.0});
    torch::Tensor expected = torch::minimum(tensor1, tensor2);

    custom_namespace::Minimum min_instance;  // Create an instance of Minimum
    auto result = min_instance.compute_min(tensor1, tensor2);
    EXPECT_TRUE(torch::allclose(result, expected));  // Verifies that result matches expected tensor
}

// The rest of the tests can follow the same pattern:
TEST(MinimumTest, MismatchedShape) {
    torch::Tensor tensor1 = torch::tensor({1.0, 5.0, 3.0});
    torch::Tensor tensor2 = torch::tensor({4.0, 2.0});

    custom_namespace::Minimum min_instance;
    EXPECT_THROW(min_instance.compute_min(tensor1, tensor2), std::invalid_argument);
}

TEST(MinimumTest, NegativeValues) {
    torch::Tensor tensor1 = torch::tensor({-1.0, -5.0, 3.0});
    torch::Tensor tensor2 = torch::tensor({-4.0, 2.0, -6.0});
    torch::Tensor expected = torch::minimum(tensor1, tensor2);
  

    custom_namespace::Minimum min_instance;
    auto result = min_instance.compute_min(tensor1, tensor2);
    EXPECT_TRUE(torch::allclose(result, expected, 1e-3));
}

TEST(MinimumTest, EqualTensors) {
    torch::Tensor tensor1 = torch::tensor({3.0, 5.0, 1.0});
    torch::Tensor tensor2 = torch::tensor({3.0, 5.0, 1.0});
    torch::Tensor expected = torch::minimum(tensor1, tensor2);

    custom_namespace::Minimum min_instance;
    auto result = min_instance.compute_min(tensor1, tensor2);
    EXPECT_TRUE(torch::allclose(result, expected));
}
